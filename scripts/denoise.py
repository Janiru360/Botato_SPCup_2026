import os
import sys
import glob
import contextlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm
from torch.cuda.amp import autocast
import numpy as np
from skimage.restoration import denoise_wavelet
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 0. ENVIRONMENT & PATH DISCOVERY
# ==========================================
# Resolve absolute paths relative to where the script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NOISY_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '../competition_data/submissions/noisy'))
OUT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '../competition_data/submissions/denoised'))
WEIGHTS_DIR = os.path.join(SCRIPT_DIR, 'weights')

# Append local SCUNet repo to path (Ensures fully offline execution)
SCUNET_PATH = os.path.join(SCRIPT_DIR, 'SCUNet')
if not os.path.exists(SCUNET_PATH):
    raise FileNotFoundError("SCUNet directory not found. Please ensure the SCUNet repository is placed inside the 'scripts/' directory.")
sys.path.append(SCUNET_PATH)

try:
    from models.network_scunet import SCUNet
except ImportError:
    raise ImportError("Failed to import SCUNet. Check if SCUNet/models/network_scunet.py exists.")

# Ensure output directory exists
os.makedirs(OUT_DIR, exist_ok=True)

# Define exact weight paths
RES_W = os.path.join(WEIGHTS_DIR, 'botato_restormer_final_460.pth')
SCU_W = os.path.join(WEIGHTS_DIR, 'scunet_color_real_psnr.pth')

if not os.path.exists(RES_W) or not os.path.exists(SCU_W):
    raise FileNotFoundError(f"Model weights missing! Please download them from the provided Google Drive link and place them in {WEIGHTS_DIR}")

# Hardware Fallback (Rubric Compliant)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USE_HALF = DEVICE.type == 'cuda'
print(f"Hardware Detected: {DEVICE.type.upper()} | FP16 Precision: {USE_HALF}")

# ==========================================
# 1. ARCHITECTURAL DEFINITIONS (Restormer)
# ==========================================
class MDTA(nn.Module):
    def __init__(self, channels, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(1, num_heads, 1, 1))
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.qkv_dwconv = nn.Conv2d(channels * 3, channels * 3, kernel_size=3, stride=1, padding=1, groups=channels * 3, bias=False)
        self.project_out = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv_dwconv(self.qkv(x)).chunk(3, dim=1)
        q, k, v = q.view(b, self.num_heads, c // self.num_heads, h * w), k.view(b, self.num_heads, c // self.num_heads, h * w), v.view(b, self.num_heads, c // self.num_heads, h * w)
        attn = torch.softmax(torch.matmul(F.normalize(q, dim=-1), F.normalize(k, dim=-1).transpose(-2, -1)) * self.temperature, dim=-1)
        return self.project_out(torch.matmul(attn, v).contiguous().view(b, c, h, w))

class GDFN(nn.Module):
    def __init__(self, channels, expansion_factor):
        super().__init__()
        hidden_channels = int(channels * expansion_factor)
        self.project_in = nn.Conv2d(channels, hidden_channels * 2, kernel_size=1, bias=False)
        self.dwconv = nn.Conv2d(hidden_channels * 2, hidden_channels * 2, kernel_size=3, stride=1, padding=1, groups=hidden_channels * 2, bias=False)
        self.project_out = nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False)
    def forward(self, x):
        x1, x2 = self.dwconv(self.project_in(x)).chunk(2, dim=1)
        return self.project_out(F.gelu(x1) * x2)

class TransformerBlock(nn.Module):
    def __init__(self, channels, num_heads, expansion_factor=2.66):
        super().__init__()
        self.norm1, self.attn = nn.LayerNorm(channels), MDTA(channels, num_heads)
        self.norm2, self.ffn = nn.LayerNorm(channels), GDFN(channels, expansion_factor)
    def forward(self, x):
        x = x + self.attn(self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous())
        return x + self.ffn(self.norm2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous())

class Restormer(nn.Module):
    def __init__(self, img_channel=3, dim=48, num_blocks=[2, 3, 3, 4], num_heads=[1, 2, 4, 8]):
        super().__init__()
        self.intro = nn.Conv2d(img_channel, dim, kernel_size=3, padding=1)
        self.enc1 = nn.Sequential(*[TransformerBlock(dim, num_heads[0]) for _ in range(num_blocks[0])])
        self.down1 = nn.Conv2d(dim, dim*2, kernel_size=4, stride=2, padding=1)
        self.enc2 = nn.Sequential(*[TransformerBlock(dim*2, num_heads[1]) for _ in range(num_blocks[1])])
        self.down2 = nn.Conv2d(dim*2, dim*4, kernel_size=4, stride=2, padding=1)
        self.enc3 = nn.Sequential(*[TransformerBlock(dim*4, num_heads[2]) for _ in range(num_blocks[2])])
        self.down3 = nn.Conv2d(dim*4, dim*8, kernel_size=4, stride=2, padding=1)
        self.bottleneck = nn.Sequential(*[TransformerBlock(dim*8, num_heads[3]) for _ in range(num_blocks[3])])
        self.up3 = nn.ConvTranspose2d(dim*8, dim*4, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(*[TransformerBlock(dim*4, num_heads[2]) for _ in range(num_blocks[2])])
        self.up2 = nn.ConvTranspose2d(dim*4, dim*2, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(*[TransformerBlock(dim*2, num_heads[1]) for _ in range(num_blocks[1])])
        self.up1 = nn.ConvTranspose2d(dim*2, dim, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(*[TransformerBlock(dim, num_heads[0]) for _ in range(num_blocks[0])])
        self.ending = nn.Conv2d(dim, img_channel, kernel_size=3, padding=1)
    def forward(self, x):
        x1, x2, x3 = self.enc1(self.intro(x)), self.enc2(self.down1(self.enc1(self.intro(x)))), self.enc3(self.down2(self.enc2(self.down1(self.enc1(self.intro(x))))))
        return self.ending(self.dec1(self.up1(self.dec2(self.up2(self.dec3(self.up3(self.bottleneck(self.down3(x3))) + x3)) + x2)) + x1)) + x

# ==========================================
# 2. MODEL INITIALIZATION
# ==========================================
print("\n--- LOADING MODELS ---")
model_r = Restormer().to(DEVICE)
model_r.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(RES_W, map_location=DEVICE).items()}, strict=True)
model_r.eval()

model_s = SCUNet(in_nc=3, config=[4, 4, 4, 4, 4, 4, 4], dim=64).to(DEVICE)
model_s.load_state_dict(torch.load(SCU_W, map_location=DEVICE), strict=True)
model_s.eval()

if USE_HALF: 
    model_r, model_s = model_r.half(), model_s.half()

def apply_8way_tta(mod, tensor, is_scunet=False):
    outputs = []
    _, _, h, w = tensor.shape
    pad_h = (64 - h % 64) % 64 if is_scunet else (16 - h % 16) % 16
    pad_w = (64 - w % 64) % 64 if is_scunet else (16 - w % 16) % 16
    if pad_h != 0 or pad_w != 0: 
        tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode='reflect')
    
    ctx = autocast() if USE_HALF else contextlib.nullcontext()
    with ctx:
        for rot in [0, 1, 2, 3]:
            for flip in [False, True]:
                x = tensor
                if rot > 0: x = torch.rot90(x, k=rot, dims=[2, 3])
                if flip: x = torch.flip(x, dims=[3])
                pred = mod(x)
                if flip: pred = torch.flip(pred, dims=[3])
                if rot > 0: pred = torch.rot90(pred, k=-rot, dims=[2, 3])
                outputs.append(pred)
            
    out = torch.mean(torch.stack(outputs), dim=0)
    return out[:, :, :h, :w] if (pad_h != 0 or pad_w != 0) else out

# ==========================================
# 3. EVALUATION
# ==========================================
def main():
    print(f"\n--- SCANNING FOR IMAGES ---")
    noisy_images = glob.glob(os.path.join(NOISY_DIR, '*_noise.png'))
    
    if not noisy_images:
        print(f"Error: No images found in {NOISY_DIR}")
        return
        
    print(f"Found {len(noisy_images)} images. Commencing Hybrid Pipeline...")

    with torch.no_grad():
        for img_path in tqdm(noisy_images, desc="Denoising"):
            filename = os.path.basename(img_path)
            base_name = filename.replace('_noise.png', '.png')
            
            # Load Image
            img_raw = TF.to_tensor(Image.open(img_path).convert("RGB"))
            img_np = img_raw.permute(1, 2, 0).numpy()
            
            img_dl = img_raw.unsqueeze(0).to(DEVICE)
            if USE_HALF: img_dl = img_dl.half()
            
            # Parallel Stream A: Deep Learning Ensemble (90% / 10%)
            pred_r_gpu = apply_8way_tta(model_r, img_dl, False).float().cpu()
            pred_s_gpu = apply_8way_tta(model_s, img_dl, True).float().cpu()
            pred_dl = (pred_r_gpu * 0.90) + (pred_s_gpu * 0.10)
            
            # Parallel Stream B: Classical 'sym8' Filter
            wavelet_np = denoise_wavelet(img_np, channel_axis=-1, wavelet='sym8', method='BayesShrink', mode='soft', rescale_sigma=True)
            pred_classical = torch.from_numpy(wavelet_np).permute(2, 0, 1).unsqueeze(0).float()
            
            # Final Blend (95% DL / 5% Classical)
            blend = (pred_dl * 0.95) + (pred_classical * 0.05)
            blend = torch.nan_to_num(blend.squeeze(0), nan=0.0, posinf=1.0, neginf=0.0).clamp(0, 1)
            
            # Save output
            out_file = os.path.join(OUT_DIR, base_name)
            TF.to_pil_image(blend).save(out_file)
            
    print(f"\n✅ Success! All denoised images saved to {OUT_DIR}")

if __name__ == '__main__':
    main()