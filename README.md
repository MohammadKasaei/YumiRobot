# YumiRobot

# Installation and Setup

## Clone the Repository:

```
git clone git@github.com:MohammadKasaei/YumiRobot.git
cd YumiRobot
```
## Set Up a Virtual Environment (optional):

```
python -m venv yumi
source yumi/bin/activate  # On Windows use `yumi\Scripts\activate`
```
## Install Dependencies:
Before running the script, make sure you have execute permissions. Run the following command:
```
chmod +x install_dependencies.sh
```
To install all the dependencies, simply run:
```
./install_dependencies.sh
```
Wait for the script to complete. Once done, all the required dependencies should be installed in your environment.



# Dependencies
The code requires `python>=3.10`, as well as `pytorch>=2.0.1` and 'CUDA Version: 12.0'

Install pytorch:
```
pip install torch torchvision 
```

Install Segment Anything:
```
pip install git+https://github.com/facebookresearch/segment-anything.git
```


The following dependencies are necessary for mask post-processing, 

```
pip install opencv-python pycocotools numpy matplotlib onnxruntime onnx 
```

Also install open3d and scikit-image:
```
pip install open3d scikit-image
```


Click the links below to download the checkpoint for the corresponding model type.

- **`default` or `vit_h`: [ViT-H SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)**
- `vit_l`: [ViT-L SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth)
- `vit_b`: [ViT-B SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)


After downloading, put the models into `sam_harmony/models/` folder. By defualt we used `vit_b` model, but it can be replaced by the others.


# Compile and Running
