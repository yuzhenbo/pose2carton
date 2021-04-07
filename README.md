# Pose2Carton
A educational project (e.g., using 3D pose to control 3D carton) for learning 3D vision (application of human mesh recovery) or imitation learning

# Requirements
* code only tested on linux system (ubuntu 16.04)
* open3d==0.12.0
* tqdm

`pip install open3d==0.12.0 tqdm` (anaconda recommended, python3)


# Environment Tutorial for MAYA
* Download: [Tutorial1](https://blog.csdn.net/otter1010/article/details/111396928), [Tutorial2](https://knowledge.autodesk.com/zh-hans/support/maya/learn-explore/caas/simplecontent/content/installing-maya-2020-ubuntu.html)
* Install: [Tutorial](https://blog.csdn.net/White_Idiot/article/details/78253004)

# FBX from the internet
[Tutorial](doc/fbx_from_the_internet.md)

# Dataset requirements 
Update soon

# Code structure
* [`transfer.py`](transfer.py): the main mapping file
* [`vis.py`](vis.py): visualize the mapping sequence of the corresponding mesh
* [`pose_samples`](pose_samples/): some samples of SMPL model for one frame
* [`obj_seq_id`](obj_seq_id/): some samples of SMPL model for temporal sequence


# Method
![image](img/pipeline.png)

# Project Result
![image](img/pose2carton.png)

# Visulization
Run vis.py (to get more clear visualization, press ctrl + 1 / ctrl + 2 / ctrl + 3 ...)

![image](img/vis.png)


# LICENSE
The code is under Apache-2.0 License.

# For EE228 students from SJTU
Please read course project [requirements](doc/EE228.md). 
