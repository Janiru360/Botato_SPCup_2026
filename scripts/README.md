# Scripts

Use this space to place your code/model/algorithm and document how to run it.

> **Important — Submission Freeze**
>
> After the preliminary-round submission deadline, no new commits may be
> made to the submitted private GitHub repository.
>
> The Git commit SHA stated in this README and recorded by the organizers
> will be treated as the official submitted code version.
>
> If external model weights/checkpoints are used, their Google Drive link,
> expected local path, and SHA-256 checksum must also be stated here.
> The submitted model/checkpoint must not be modified or replaced after
> the deadline.

## How to Run

1. **Clone the Repository:** Open your terminal and clone the repository using Git, then navigate into the project directory:
   ```bash
   git clone <REPOSITORY_URL>
   cd <REPOSITORY_NAME>
   ```

2. **Environment Setup:** Ensure you have Python 3.8+ installed on your system.

3. **Install Dependencies:** Navigate to this `scripts/` directory in your terminal and install the required packages using:
   ```bash
   cd scripts
   pip install -r requirements.txt
   ```

4. **Download and Place SCUNet Codebase:** Follow the instructions in the [SCUNet Setup](#scunet-setup) section below to download and position the required model architecture folder.

5. **Download and Place Model Weights:** Follow the instructions in the [Pretrained Models](#pretrained-models) section below to download and correctly position the `.pth` files.

6. **Prepare Data:** Ensure the 20 target noisy images (e.g., `461_noise.png`) are located in the required `../competition_data/submissions/noisy/` directory relative to the repository root.

7. **Execution:** Run the denoising script from within the `scripts/` directory:
   ```bash
   python denoise.py
   ```

## SCUNet Setup

Because the SCUNet codebase is no longer tracked as a submodule, you need to manually download it.

### 1. Download the Codebase

Download the zipped SCUNet archive (e.g., `SCUNet.zip`) from the following Google Drive folder:
*   **Download Link:** [SPCup_FinalWeights Folder](https://drive.google.com/drive/folders/1jeMf5u829oAC6IbL3PGVAA8VItAdKjem?usp=sharing)

### 2. Extract and Place the Folder

Extract the contents of the zip file. Move the extracted `SCUNet` folder directly into your `scripts` directory. 

## Pretrained Models

Before running the project, you need to manually download the model checkpoints and place them in the correct directory.

### 1. Download the Weights

Download the `botato_final_weights.zip` archive from the same Google Drive folder:
*   **Download Link:** [SPCup_FinalWeights Folder](https://drive.google.com/drive/folders/1jeMf5u829oAC6IbL3PGVAA8VItAdKjem?usp=sharing)

### 2. Extract and Place the Files

Extract the contents of `botato_final_weights.zip`. Create a `weights` folder inside your `scripts` directory (if it doesn't already exist) and move both `.pth` files into it. 

Your final file structure (after both SCUNet and the weights are placed) must look exactly like this:

```text
├── scripts/
│   ├── SCUNet/
│   │   ├── ... (SCUNet model files)
│   ├── weights/
│   │   ├── botato_restormer_final_460.pth
│   │   └── scunet_color_real_psnr.pth
```

## Requirements

1. **Software / Packages:**
All software dependencies are listed in `requirements.txt`. Key libraries include `torch`, `torchvision`, `scikit-image`, `opencv-python`, `einops`, `thop`, and `pytorch-msssim`.
*(Note: The SCUNet architecture codebase must be manually downloaded and placed inside the `scripts/SCUNet/` directory as per the instructions above).*

2. **Hardware:**
The codebase features an automated hardware manager. If a CUDA-compatible GPU is detected, inference will automatically run using FP16 (Half) precision for speed and memory efficiency. If no GPU is available, the script gracefully falls back to FP32 execution on the CPU to prevent out-of-memory crashes and ensure successful offline evaluation.

## Usage

Run the solution directly via the terminal:
```bash
python denoise.py
```

The script requires zero command-line arguments. It automatically dynamically maps the input and output directories based on the official repository structure:

*   **Inputs expected from:** `../competition_data/submissions/noisy/`
*   **Outputs written to:** `../competition_data/submissions/denoised/`

## Output

The script processes all `*_noise.png` files found in the input directory. The denoised outputs are saved directly into the target output directory using the strict `<id>.png` naming convention (e.g., processing `461_noise.png` results in a fully denoised `461.png`).

## Official Submission Information

**Git Commit SHA:**
`<commit SHA>`

### **Model Checkpoint: Model 1 (Global Stream): Restormer**
* **Model Checkpoint:** `botato_restormer_final_460.pth`
* **Model Drive Link:** [SPCup_FinalWeights Folder](https://drive.google.com/drive/folders/1jeMf5u829oAC6IbL3PGVAA8VItAdKjem?usp=sharing)
* **Expected Model Path:** `scripts/weights/botato_restormer_final_460.pth`
* **Model SHA-256:** `974ba021133fa22c7f60cce7c2add48dcf0fd7e61216476c54b2f1c87d72d3c7`

### **Model 2 (Local Stream): SCUNet**
* **Model Checkpoint:** `scunet_color_real_psnr.pth`
* **Model Drive Link:** [SPCup_FinalWeights Folder](https://drive.google.com/drive/folders/1jeMf5u829oAC6IbL3PGVAA8VItAdKjem?usp=sharing)
* **Expected Model Path:** `scripts/weights/scunet_color_real_psnr.pth`
* **Model SHA-256:** `fa78899ba2caec9d235a900e91d96c689da71c42029230c2028b00f09f809c2e`

Model Drive Link:
`<link or N/A>`

Expected Model Path:
`<path or N/A>`

Model SHA-256:
`<SHA-256 or N/A>`