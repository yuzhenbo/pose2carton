#!/usr/bin/env python
# coding=utf-8

import time
import os 
import os.path as osp 
import warnings
import glob
import numpy as np
import cv2
import open3d as o3d 
from tqdm import tqdm

use_online_model = False
if not use_online_model and int(o3d.__version__.split('.')[1]) < 11:
    warnings.warn('You are using open3d below 0.11.0, which may cause black images in saving screen captures' + \
                     'you can use a video recorder to save the visualization, or switch to a newer version')
elif use_online_model:
    warnings.warn("You are using 3d model downloaded elsewhere; to see the texture with open3d, you need to" + \
                  "ensure that your open3d is below 0.11.0(e.g. 0.10.0); but this will cause open3d fail to save the" + \
                  "visualization, so you shall need to use other video recording tool; Also, rendering process may be" + \
                  "a bit slow, wait with patience to prove your effort and get the bonus :)" )


savedir = "vis/"
os.makedirs(savedir, exist_ok=True)
save_video_path = osp.join(savedir, "vis.mp4")
human_obj_dir = "./obj_seq_5"

visualizer = o3d.visualization.Visualizer()
visualizer.create_window('open3d')
ctr = visualizer.get_view_control()

model_obj_dir = human_obj_dir + '_3dmodel'

human_obj_files = sorted(glob.glob(osp.join(human_obj_dir, "*.obj")), key=lambda x: int(osp.basename(x).split('.')[0]))
model_obj_files = sorted(glob.glob(osp.join(model_obj_dir, "*.obj")), key=lambda x: int(osp.basename(x).split('.')[0]))
mesh = o3d.io.read_triangle_mesh(model_obj_files[0])
mesh.vertex_colors = o3d.utility.Vector3dVector(np.asarray(mesh.vertex_colors) / 3)
dist = mesh.get_max_bound() - mesh.get_min_bound()
mesh.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices) / dist)
mesh.translate(-mesh.get_center())

human_mesh = o3d.io.read_triangle_mesh(human_obj_files[0])
human_mesh.translate(-human_mesh.get_center() + [1.5, 0.0, 0.0])
human_mesh.vertex_colors = o3d.utility.Vector3dVector(np.asarray(human_mesh.vertex_colors) / 3)
visualizer.add_geometry(mesh)
visualizer.add_geometry(human_mesh)
tbar = tqdm(range(1, len(model_obj_files)))

def set_color(mesh):
    vertices = np.asarray(mesh.vertices)
    min_vals = vertices.min(axis=0, keepdims=True)
    max_vals = vertices.max(axis=0, keepdims=True)
    colors = (vertices - min_vals) / (max_vals - min_vals)
    # colors = colors[..., ::-1]
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    return mesh

def compose_video(): 
    _files = sorted(glob.glob(osp.join(savedir, "*.png")), key=lambda x: int(osp.basename(x).split('.')[0]))
    h, w = 480, 640
    video_writer = cv2.VideoWriter(save_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 24, (w, h))
    for _file in tqdm(_files, total=len(_files)): 
        img = cv2.imread(_file)
        img = cv2.resize(img, (w, h))
        video_writer.write(img)
    video_writer.release()
    print('save video to', save_video_path)
    

for idx in tbar: 
    human_obj_file = human_obj_files[idx]
    model_obj_file = model_obj_files[idx]
    mesh.vertices = o3d.io.read_triangle_mesh(model_obj_file).vertices
    if idx == 0:
        dist = mesh.get_max_bound() - mesh.get_min_bound()

    mesh.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices) / dist)
    human_mesh.vertices = o3d.io.read_triangle_mesh(human_obj_file).vertices

    mesh.translate(-mesh.get_center())
    human_mesh.translate(-human_mesh.get_center() + [1.5, 0.0, 0.0])

    # we already have texture for online model, skip
    if not use_online_model:
        mesh = set_color(mesh)
    human_mesh = set_color(human_mesh)
    visualizer.update_geometry(mesh)
    visualizer.update_geometry(human_mesh)
    if not use_online_model:
        time.sleep(0.02)
    visualizer.poll_events()

    # visualizer.capture_screen_image(osp.join("vis", "{}.png".format(idx)), do_render=True)
    vis_save_path = osp.join(savedir, str(idx) + ".png")
    # img = np.asarray(visualizer.capture_screen_float_buffer()).copy()
    visualizer.capture_screen_image(vis_save_path, True)

visualizer.run()
visualizer.destroy_window()
compose_video()


