# Installation
The requirements for the environment are based on [Mask2Former](https://github.com/facebookresearch/Mask2Former/blob/main/INSTALL.md)'s environment.

## Example conda environment setup
Our code is developed based on **PyTorch 2.5.1** with **CUDA 11.8**:
```bash
# Create and activate conda environment
conda create -n s4m_env python=3.10.9
conda activate s4m_env

# Install PyTorch with CUDA 11.8
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118
```

Install **Detectron2** and compile the custom CUDA ops for **Mask2Former**:
```bash
# Install Detectron2
python3 -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

# Clone Mask2Former and build custom ops
git clone https://github.com/facebookresearch/Mask2Former && cd Mask2Former
python3 -m pip install -r requirements.txt

cd mask2former/modeling/pixel_decoder/ops
chmod +x make.sh
sh make.sh

# Return to the parent directory
cd ../../../../../
```

After installing Detectron2 and Mask2Former, clone this repository and install its dependencies:
```bash 
git clone https://github.com/cvlab-kaist/S4M.git && cd S4M
python3 -m pip install -r requirements.txt
python3 -m pip install git+https://github.com/cocodataset/panopticapi.git
```

Install SAM2 and download a [model_checkpoint](https://github.com/facebookresearch/sam2#model-description). Please save the SAM2 checkpoints under `./model_weights` (we recommend that you download san2_hiera_large.pt, as this is the checkpoint we primarily use!).

```bash
cd sam2
pip install -e .
```




