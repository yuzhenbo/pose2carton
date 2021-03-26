# Pose2Carton
A educational project (e.g., using 3D pose to control 3D carton) for learning 3D vision (application of human mesh recovery) or imitation learning

# Requirements
* code only tested on linux system (ubuntu 16.04)
* open3d==0.12.0
* tqdm
`pip install open3d==0.12.0 tqdm` (anaconda recommended, python3)

# Code structure
* [`transfer.py`](transfer.py): the main mapping file
* [`vis.py`](vis.py): visualize the mapping sequence of the corresponding mesh
* [`pose_samples`](pose_samples/): some samples of SMPL model for one frame
* [`obj_seq_id`](obj_seq_id/): some samples of SMPL model for temporal sequence


To Do
More details will be updated soon.
----------------------------------------------------------------

环境依赖
open3d==0.12.0 
tqdm

`pip install open3d==0.12.0 tqdm` (anaconda recommended, python3)

* 运行 fbx_parser 和 maya_save_fbx 脚本需要 mayapy 环境(先安装，配置maya)

* transfer.py 进行匹配和迁移

* vis.py 对匹配得到的一个序列 mesh 进行可视化(需要先运行transfer.py 中的transfer_one_sequence函数) 如果需要更好的视觉效果，press ctrl + 1 / ctrl + 2 / ctrl + 3 ...

* pose_samples/ 随即生成的一些人体姿态, SMPL模型

* info_seq_id.pkl 包含某一视频序列的人体姿态信息

* obj_seq_id/ 包含视频序列中的人体mesh数据

* obj_seq_id/, pose_samples/, fbx/ 下的obj建议用meshlab查看

* fbx/ 下的fbx建议用blender, maya查看

submission : 最终你需要提交 results 文件夹压缩包, vis.py 生成的可视化录屏

----
详细稍后会补充说明
待更新