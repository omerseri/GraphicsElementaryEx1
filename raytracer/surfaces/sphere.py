import numpy as np
from . import Surface
from constants import EPSILON

class Sphere(Surface):
    def __init__(self, position, radius, material_index):
        self.position = np.array(position)
        self.radius = float(radius)
        self.material_index = material_index

    def intersect(self, ray_origin, ray_dir):
        # Math: |O + tV - C|^2 = r^2
        L = self.position - ray_origin
        tca = np.dot(L, ray_dir)
        if tca < 0:
            return float('inf'), None
        d2 = np.dot(L, L) - tca * tca
        if d2 > self.radius * self.radius:
            return float('inf'), None
        thc = np.sqrt(self.radius * self.radius - d2)
        t0 = tca - thc
        t1 = tca + thc
        t = t0 if t0 > EPSILON else t1
        if t < EPSILON:
            return float('inf'), None
        hit_point = ray_origin + t * ray_dir
        normal = hit_point - self.position
        normal = normal / np.linalg.norm(normal)
        return t, normal