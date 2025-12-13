class Surface:
    def intersect(self, ray_origin, ray_dir):
        """
        Calculates intersection with a ray.
        :param ray_origin: numpy array (3,)
        :param ray_dir: numpy array (3,) - normalized
        :return: tuple (t, normal)
                 t: distance to intersection (float), or inf if no hit
                 normal: surface normal at intersection (numpy array), or None
        """
        raise NotImplementedError("Subclasses must implement intersect method")