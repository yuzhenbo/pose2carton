***

环境依赖

open3d==0.9.0.0

tqdm

h5py

`pip install open3d==0.9.0.0`

(anaconda recommended, python3)

(使用 `python -c "import open3d as o3d`进行测试，如不报错则说明安装成功)
 
`pip install tqdm`

(如果使用anaconda进行包管理，比较新的版本里会自带)

`pip install h5py`

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