import numpy as np

def normalize(v):
    norm = np.linalg.norm(v)
    return v if norm == 0 else v / norm

class Camera:
    def __init__(self, position, look_at, up_vector, screen_distance, screen_width):
        # Data conversion from raw lists to numpy
        self.position = np.array(position)
        self.look_at = np.array(look_at)
        self.up_vector = np.array(up_vector)
        self.screen_distance = float(screen_distance)
        self.screen_width = float(screen_width)
        
        # Precompute Basis Vectors
        self.W = normalize(self.look_at - self.position)
        self.U = normalize(np.cross(self.up_vector, self.W))
        self.V = normalize(np.cross(self.W, self.U))
        
        self.screen_center = self.position + self.W * self.screen_distance

    def get_ray(self, x, y, width, height):
        aspect_ratio = float(width) / height
        screen_height = self.screen_width / aspect_ratio
        
        # Map pixels to Screen Plane
        normalized_x = (x / float(width)) - 0.5
        normalized_y = ((height - y) / float(height)) - 0.5
        
        world_x = normalized_x * self.screen_width
        world_y = normalized_y * screen_height
        
        pixel_pos = self.screen_center + (self.U * world_x) + (self.V * world_y)
        
        direction = normalize(pixel_pos - self.position)
        return self.position, direction