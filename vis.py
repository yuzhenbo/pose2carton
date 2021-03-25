#!/usr/bin/env python
# coding=utf-8

import time
import os 
import os.path as osp 
import glob
import numpy as np
import open3d as o3d 
from tqdm import tqdm

visualizer = o3d.visualization.Visualizer()
visualizer.create_window('open3d')

human_obj_dir = "./obj_seq_5"
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

for idx in tbar: 
    human_obj_file = human_obj_files[idx]
    model_obj_file = model_obj_files[idx]
    mesh.vertices = o3d.io.read_triangle_mesh(model_obj_file).vertices
    dist = mesh.get_max_bound() - mesh.get_min_bound()
    mesh.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices) / dist)
    mesh.translate(-mesh.get_center())

    human_mesh.vertices = o3d.io.read_triangle_mesh(human_obj_file).vertices
    human_mesh.translate(-human_mesh.get_center() + [1.5, 0.0, 0.0])

    mesh = set_color(mesh)
    human_mesh = set_color(human_mesh)
    visualizer.update_geometry(mesh)
    visualizer.update_geometry(human_mesh)
    time.sleep(0.02)
    visualizer.poll_events()
    # visualizer.capture_screen_image(osp.join("vis", "{}.png".format(idx)), do_render=True)

visualizer.run()
visualizer.destroy_window()


