# Neural Localizer Fields Evaluation on Drive & Act Dataset

<p align="center">
  <img src="demo_smplest.gif" alt="Demo GIF">
</p>

## Overview
This repository contains the evaluation code for the paper [**"Neural Localizer Fields for Continuous 3D Human Pose and Shape Estimation"**](https://arxiv.org/abs/2407.07532) on a custom dataset. The code computes the MPJPE (Mean Per Joint Position Error) using the Neural Localizer Fields (NLF) model, projects the 3D SMPL model to 2D, and visualizes the results by saving the images as well as generating a video.  
A sample output video can be seen [on youtube](https://driveandact.com/) (Note: The model is not fine-tuned for the Drive&Act dataset, so the results are not meant as a benchmark).

For comparison, previews of other methods can be also seen on youtube:
- [SMPLest-X](https://youtu.be/rc0tO6B85pU)
- [CLIFF](https://youtu.be/-a_40oJfg9Y)

More information on statistical body models can be found in the [accompanying text](human_body_models.pdf).

## Table of Contents
1. [Installation](#installation)
2. [Datasets & Model Checkpoints](#datasets--model-checkpoints)
3. [Configuration](#configuration)
4. [Running the Code](#running-the-code)
5. [Project Structure](#project-structure)
6. [Known Issues](#known-issues)
7. [References / Citations](#references--citations)
8. [License](#license)

---

## Installation


### Installation using `uv`

This project is configured to use the [**uv**](https://github.com/astral-sh/uv) package manager to manage dependencies. If you are new to `uv`, follow the steps below to set up the environment and run the code:

1. **Clone the Repository**
   ```bash
   git clone https://github.com/<username>/<repo_name>.git
   cd <repo_name>
   ```
   Make sure your current directory contains the `pyproject.toml` file.

2. **Install `uv`**
   If you do not already have `uv` installed:
   ```bash
   pip install uv
   ```

3. **Initialize the Project**
   ```bash
   uv init
   ```
   - This command will detect your `pyproject.toml` and prepare the project structure for `uv`.
   - You should see `uv` creating or updating files such as the `.uv/` folder for local config.

4. **Sync/Install Dependencies**
   ```bash
   uv sync
   ```
   - `uv` reads the `[project]` and `[project].dependencies` sections of your `pyproject.toml`, resolves them, and installs everything into a dedicated virtual environment (in `.uv/venv/` by default).
   - After this step, all dependencies listed in `pyproject.toml` should be installed.

5. **Activate the `uv` Virtual Environment** *(optional but recommended)*
   ```bash
   source .uv/venv/bin/activate
   ```
   or on Windows:
   ```bash
   .uv\venv\Scripts\activate
   ```
   Now, any Python scripts you run will use the environment that `uv` created.

#### Troubleshooting
- If you encounter an error about missing dependencies, verify that `uv sync` completed successfully.  
- If you prefer installing dependencies via a different method (e.g., pip or conda), you can extract them from `pyproject.toml` and install them in your own environment. 

Once completed, you should have:
1. A local `uv`-managed environment containing all required packages.
3. The ability to run the script via the `python -m src.main ...` command, read below.

---

### Alternatively (via requirements.txt or conda)
If you do not wish to use `uv`, you can extract the dependencies 
from the `[project].dependencies` section of the `pyproject.toml` and place 
them in a `requirements.txt`, or set them up in a conda environment. The project 
should remain compatible with a standard `pip install -r requirements.txt` approach. 

A minimal dependencies excerpt might look like:
```bash
chumpy==0.70
einops==0.8.1
json-tricks==3.17.3
matplotlib==3.7.5
numpy==1.23.5
opencv-python==4.8.1.78
pillow>=9.1.1
pyopengl>=3.1.0
pyrender==0.1.45
smplx==0.1.28
timm==1.0.14
torch==1.13.1
torchvision==0.14.1
torchaudio==0.13.1
torchgeometry>=0.1.2
tqdm==4.67.1
trimesh==4.6.2
ultralytics==8.3.75
yacs>=0.1.8
gpt-repository-loader>=0.10.0
numba>=0.60.0
boxlib
```
(You may need to select the correct PyTorch binaries for your CUDA version.)

---

## Datasets & Model Checkpoints

### Drive&Act Dataset
- Download the **Drive&Act A-Column driver dataset** from [the official website](https://driveandact.com/dataset/a_column_driver.zip) under “a_column_driver.zip.”  
  Note: This dataset may have usage restrictions. See the [Drive&Act license or usage terms](https://driveandact.com) for more details.

- Download the **corresponding pose annotations** from [this link](https://driveandact.com/dataset/iccv_openpose_3d.zip).

- You will need to extract frames from the video. See the annotations file for the annotated frame indices.


### SMPL Model
- Obtain the SMPL model (in `.pkl` format) from the [official SMPL website](https://smpl.is.tue.mpg.de/login.php).

### NLF Model
- Download the precompiled NLF TorchScript file from [this link](https://bit.ly/nlf_l_pt).  

Once you have unzipped these files, place them in the appropriate folders and specify their paths in your `config.yaml` (see below).

---

## Configuration

You can find (and edit) the main configuration file at `configs/config.yaml`. It contains the following entries:

```yaml
files:
  annotations_file_csv: "path/to/pose_annotations.csv"
  video_file: "path/to/video.mp4"
  timestamps_file: "path/to/video_timestamps.timestamp"
  calibration_file: "path/to/camera_calibration_file.json"
  output_folder: "path/to/output_folder"
  image_folder: "path/to/folder/with/input/images/from/video"

smpl:
  smpl_model_path: "path/to/smpl_model.pkl"

nlf:
  nlf_model_path: "path/nlf/compiled_model.torchscript"

video_processing:
  image_start: 10000
  length: 200
  batch_size: 8

parallel:
  max_workers: 4
```
Adjust these fields to match your local environment. For instance, if your frames start at 000000.jpg and you want to process frames from index 15000 to 15100, set `image_start` to `15000` and `length` to `100`.

---

## Running the Code

1. **Ensure** the `config.yaml` is updated with your dataset, SMPL model, and TorchScript checkpoint paths.
2. **Activate** your environment (whether the `uv` environment or another one).
3. **Run**:
   ```bash
   python -m src.main --config configs/config.yaml
   ```
   This will load the NLF model, read your frames, compute 3D pose, project it back to 2D, measure MPJPE, and store images plus a final video in your specified `output_folder`.

#### Example Quick Test
If you just want to test a smaller subset, you can set:
```yaml
video_processing:
  image_start: 25000
  length: 10
  batch_size: 2
```
This processes frames 25000 to 25009 in increments of 2 images at a time.

---

## Project Structure

The final structure of the repository should look as follows:

```
NeuralLocalizerFields/
  ├── configs/
  │    └── config.yaml
  ├── src/
  │    ├── main.py
  │    ├── visualization_utils.py
  │    └── ...
  ├── data/
  │    ├── some_folder_for_extracted_frames/
  │    └── ...
  ├── human_models/
  │    ├── smpl
  │    │    ├── SMPL_MALE.pkl
  │    │    └── ...
  ├── models/
  │    ├── nlf_s_multi.torchscript
  │    └── ...
  ├── README.md
  └── ...
```
- **configs/** contains your YAML configuration.
- **src/** holds the main scripts (`main.py`, etc.).
- **data/** is where you’d place your dataset files (or symlink them from another location).
- **models/** might contain your downloaded or compiled NLF TorchScript and SMPL files.
- The final output frames and videos would be placed in the folder specified by `output_folder` in the config.

---
## Known Issues
1. **TorchScript Model Crash**  
   - There is a bug in the compiled TorchScript models that occasionally causes the model to crash.  
   - *Mitigation:* If a crash occurs, the code will skip the problematic batch and continues processing the next one. The skipped frames are still outputted, but without the pose estimation.

2. **Drive&Act Dataset Constraints**  
   - The code is tailored to the [Drive&Act dataset](https://driveandact.com/). You may need to adapt certain indexing logic if you want to apply it to a different dataset structure.

3. **Single Person Assumption**  
   - The code currently assumes only one person is present in each frame. If you require multi-person support, you can modify the for-loops that handle bounding boxes or persons.

---

## References / Citations
This code is free to use without restrictions. However, if you use the Neural Localizer Fields approach in your work, 
cite the [original paper](https://arxiv.org/abs/2407.07532). The dataset used is from 
[Drive&Act](https://driveandact.com/), so also cite the relevant Drive&Act publication or license as applicable.


---

## License
This evaluation code is released under the [MIT License](LICENSE). Please also respect the 
licensing terms of the Drive&Act dataset, [SMPL model](https://smpl.is.tue.mpg.de/), etc.
