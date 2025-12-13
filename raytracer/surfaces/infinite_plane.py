import numpy as np
from . import Surface
from constants import EPSILON

class InfinitePlane(Surface):
    def __init__(self, normal, offset, material_index):
        self.normal = np.array(normal)
        self.offset = float(offset)
        self.material_index = material_index

    def intersect(self, ray_origin, ray_dir):
        # Math: (P . N) = offset
        # (O + tD) . N = offset
        denom = np.dot(ray_dir, self.normal)

        # Avoid division by zero (parallel ray)
        if abs(denom) < 1e-6:
            return float('inf'), None

        t = (self.offset - np.dot(ray_origin, self.normal)) / denom
        
        if t > EPSILON:
            return t, self.normal
            
        return float('inf'), None