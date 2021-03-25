#!/usr/bin/env python
# coding=utf-8

import os 
import os.path as osp 
import glob 
import shutil 
import random 
import argparse 

random.seed(2021)

parser = argparse.ArgumentParser()
parser.add_argument('--num_groups', default=30)
args = parser.parse_args()

num_groups = int(args.num_groups)
group_root = "groups"
if osp.exists(group_root): 
    shutil.rmtree(group_root)

os.makedirs(group_root, exist_ok=True)
materials = [name for name in os.listdir('.') if name != 'fbx' and name != group_root]

fbx_files = glob.glob("fbx/*.fbx")
fbx_files = [fbx_file for fbx_file in fbx_files if osp.exists(fbx_file.replace(".fbx", ".obj")) and osp.exists(fbx_file.replace(".fbx", ".txt"))]
random.shuffle(fbx_files)
group_len = len(fbx_files) // num_groups

print('assigning {} files into {} groups'.format(len(fbx_files), num_groups))

for i in range(num_groups): 
    start = group_len * i 
    end = group_len * (i+1)
    group_dir = osp.join(group_root, 'group_' + str(i))
    os.makedirs(group_dir, exist_ok=True)
    # for material in materials: 
    #     if osp.isdir(material): 
    #         if osp.exists(osp.join(group_dir, material)): 
    #             shutil.rmtree(osp.join(group_dir, material))
    #         shutil.copytree(material, osp.join(group_dir, material))
    #     else:
    #         shutil.copy(material, osp.join(group_dir, material))
    #     print('copy {} to {}'.format(material, osp.join(group_dir, material)))

    group_fbx_dir = osp.join(group_dir, "fbx")
    os.makedirs(group_fbx_dir, exist_ok=True)
    for fbx_file in fbx_files[start:end]: 
        shutil.copy(fbx_file, osp.join(group_fbx_dir, osp.basename(fbx_file)))
        shutil.copy(fbx_file.replace(".fbx", ".obj"), osp.join(group_fbx_dir, osp.basename(fbx_file).replace(".fbx", ".obj")))
        shutil.copy(fbx_file.replace(".fbx", ".txt"), osp.join(group_fbx_dir, osp.basename(fbx_file).replace(".fbx", ".txt")))

