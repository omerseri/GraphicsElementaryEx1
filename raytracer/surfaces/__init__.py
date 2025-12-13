import numpy as np

class Surface:
    def intersect(self, ray_origin, ray_dir):
        raise NotImplementedError("Subclasses must implement intersect method")

    def normalize(self, v):
        norm = np.linalg.norm(v)
        return v if norm == 0 else v / norm