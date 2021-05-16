#!/usr/bin/env python
# coding=utf-8

import numpy as np 

class TriangleMesh:
    def __init__(self, filename): 
        self.vertices, self.triangles = self.load(filename)

    def load(self, filename):
        fp = open(filename, "r")
        lines = fp.read().strip().split('\n')
        vid = 0 
        fid = 0
        vertices = []
        triangles = []
        for line in lines: 
            if not line.startswith("v") and not line.startswith("f"):
                continue 
            if line.startswith("v"):
                vertices.append(line.split(' ')[1:])
                vid += 1  
            else:
                triangles.append(line.split(' ')[1:])
                fid += 1
        fp.close()
        vertices = np.array(vertices).reshape(-1, 3).astype(np.float32)
        triangles = np.array(triangles).reshape(-1, 3).astype(np.int)
        return vertices, triangles
