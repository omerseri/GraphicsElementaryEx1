import numpy as np
from . import Surface
from constants import EPSILON

class Sphere(Surface):
    def __init__(self, position, radius, material_index):
        self.position = np.array(position)
        self.radius = float(radius)
        self.material_index = material_index

    def intersect(self, ray_origin, ray_dir):
        L = self.position - ray_origin
        tca = np.dot(L, ray_dir)
        
        d2 = np.dot(L, L) - tca * tca
        if d2 > self.radius * self.radius:
            return float('inf'), None
            
        thc = np.sqrt(self.radius * self.radius - d2)
        
        t0 = tca - thc
        t1 = tca + thc
        
        # If t0 is negative (behind us), try t1 (in front of us)
        if t0 > EPSILON:
            return t0, self.normalize(ray_origin + t0 * ray_dir - self.position)
        elif t1 > EPSILON:
            # We are inside the sphere, normal points inward
            return t1, -self.normalize(ray_origin + t1 * ray_dir - self.position)
            
        return float('inf'), None