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



# Download SAM Models

Click the links below to download the checkpoint for the corresponding model type.

- **`default` or `vit_h`: [ViT-H SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)**
- `vit_l`: [ViT-L SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth)
- `vit_b`: [ViT-B SAM model.](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)


After downloading, put the models into `sam_harmony/models/` folder. By defualt we used `vit_b` model, but it can be replaced by the others.


# Running
After installing all the dependencies and downloading the models and put them into the `sam_harmony/models/` folder, you can run the simulator. We also provide a version that does not have the vision pipeline, so you dont need to download the models.


# Vision-based grasping from the box
```
python -m tests.harmony_test_vision
```
Once everything successfully installed, you'll see a box containing four racks randomly place in front of the simulated Yumi robot and the robot is performing rack extraction task.

![alt](images/yumi_with_vision.gif)

The vision pipeline is composed of four stages:

![alt](images/grasp-pipeline.png)

# Blind grasping from the box
```
python -m tests.harmony_test_vision
```
Once everything successfully installed, you'll see a box containing four racks placed in front of the simulated Yumi robot in a fixed position, and the robot is performing rack extraction task.


![alt](images/yumi_without_vision.gif)
