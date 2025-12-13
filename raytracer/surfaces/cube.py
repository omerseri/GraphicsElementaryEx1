import numpy as np
from . import Surface
from constants import EPSILON

class Cube(Surface):
    def __init__(self, position, scale, material_index):
        self.position = np.array(position)
        self.scale = float(scale)
        self.material_index = material_index

    def intersect(self, ray_origin, ray_dir):
        # Slabs method
        center = self.position
        half_scale = self.scale / 2.0
        
        p_min = center - half_scale
        p_max = center + half_scale

        t_min = -float('inf')
        t_max = float('inf')
        
        normal = np.zeros(3)
        
        for i in range(3): # x, y, z axes
            # Check for parallel rays
            if abs(ray_dir[i]) < 1e-9:
                if ray_origin[i] < p_min[i] or ray_origin[i] > p_max[i]:
                    return float('inf'), None
            else:
                t1 = (p_min[i] - ray_origin[i]) / ray_dir[i]
                t2 = (p_max[i] - ray_origin[i]) / ray_dir[i]
                
                # We want t1 to be the entry, t2 to be the exit
                if t1 > t2:
                    t1, t2 = t2, t1
                
                if t1 > t_min:
                    t_min = t1
                    # Update normal based on the axis we just hit
                    current_normal = np.zeros(3)
                    current_normal[i] = -1 if ray_dir[i] > 0 else 1
                    normal = current_normal
                    
                if t2 < t_max:
                    t_max = t2
                    
                if t_min > t_max:
                    return float('inf'), None
                    
        if t_min > EPSILON:
            return t_min, normal
        
        return float('inf'), None