import argparse
from PIL import Image
import numpy as np
import time

from constants import EPSILON
from camera import Camera
from light import Light
from material import Material
from scene_settings import SceneSettings
from surfaces import Surface 
from surfaces.cube import Cube
from surfaces.infinite_plane import InfinitePlane
from surfaces.sphere import Sphere

def normalize(v):
    norm = np.linalg.norm(v)
    return v if norm == 0 else v / norm

def parse_scene_file(file_path):
    objects = []
    camera = None
    scene_settings = None
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            obj_type = parts[0]
            params = [float(p) for p in parts[1:]]
            if obj_type == "cam":
                camera = Camera(params[:3], params[3:6], params[6:9], params[9], params[10])
            elif obj_type == "set":
                scene_settings = SceneSettings(params[:3], params[3], params[4])
            elif obj_type == "mtl":
                material = Material(params[:3], params[3:6], params[6:9], params[9], params[10])
                objects.append(material)
            elif obj_type == "sph":
                sphere = Sphere(params[:3], params[3], int(params[4]))
                objects.append(sphere)
            elif obj_type == "pln":
                plane = InfinitePlane(params[:3], params[3], int(params[4]))
                objects.append(plane)
            elif obj_type == "box":
                cube = Cube(params[:3], params[3], int(params[4]))
                objects.append(cube)
            elif obj_type == "lgt":
                light = Light(params[:3], params[3:6], params[6], params[7], params[8])
                objects.append(light)
            else:
                raise ValueError("Unknown object type: {}".format(obj_type))
    return camera, scene_settings, objects

def save_image(image_array, file_name="output.png"):
    image = Image.fromarray(np.uint8(image_array))
    image.save(file_name)

def find_nearest_object(ray_origin, ray_dir, surfaces):
    nearest_t = float('inf')
    nearest_obj = None
    nearest_normal = None
    
    # Go over all objects in scene amd calculate intersections.
    for obj in surfaces:
        t, normal = obj.intersect(ray_origin, ray_dir)
        if t < nearest_t:
            nearest_t = t
            nearest_obj = obj
            nearest_normal = normal

    return nearest_t, nearest_obj, nearest_normal

def calculate_soft_shadow(hit_point, light, surfaces, root_shadow_rays):
    N = root_shadow_rays
    light_pos = np.array(light.position)
    to_light = light_pos - hit_point
    light_dir = normalize(to_light)

    # Creating local coordinate system for light source.
    helper = np.array([0, 1, 0])
    if abs(np.dot(helper, light_dir)) > 0.99:
        helper = np.array([1, 0, 0])
    
    light_u = normalize(np.cross(helper, light_dir))
    light_v = normalize(np.cross(light_dir, light_u))

    rect_width = light.radius
    cell_size = rect_width / N
    start_point = light_pos - (light_u * rect_width / 2) - (light_v * rect_width / 2)

    # Creating grid of indecies
    i_indices, j_indices = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
    
    # Creating random values for each cell all at once.
    rand_u = np.random.random((N, N))
    rand_v = np.random.random((N, N))
    
    # Calculating offset for each sample.
    offsets_u = (i_indices + rand_u) * cell_size
    offsets_v = (j_indices + rand_v) * cell_size
    
    # Calculating each sample point.
    sample_points = (start_point + 
                     offsets_u[:, :, np.newaxis] * light_u + 
                     offsets_v[:, :, np.newaxis] * light_v)
    
    # Transform into list (N*N, 3)
    flat_samples = sample_points.reshape(-1, 3)
    shadow_vecs = flat_samples - hit_point
    dists_to_samples = np.linalg.norm(shadow_vecs, axis=1)
    shadow_dirs = shadow_vecs / dists_to_samples[:, np.newaxis]
    
    rays_hit_light = 0.0
    # Find the nearest object for each shadow ray.
    for idx in range(N * N):
        nearest_t, _, _ = find_nearest_object(hit_point, shadow_dirs[idx], surfaces)
        if nearest_t >= dists_to_samples[idx] - EPSILON: 
            rays_hit_light += 1.0

    return rays_hit_light / (N * N)

def cast_ray(ray_origin, ray_dir, surfaces, materials, lights, settings, recursion_level):
    if recursion_level > settings.max_recursions:
        return np.array(settings.background_color)

    # Find the which object the current ray hits.
    t, hit_obj, normal = find_nearest_object(ray_origin, ray_dir, surfaces)

    # The ray doesn't hit any object, return the background color.
    if hit_obj is None:
        return np.array(settings.background_color)

    hit_point = ray_origin + t * ray_dir
    hit_point_offset = hit_point + normal * EPSILON 

    mat = materials[hit_obj.material_index - 1]
    
    diffuse_final = np.zeros(3)
    specular_final = np.zeros(3)

    for light in lights:
        light_pos = np.array(light.position)
        L_vec = light_pos - hit_point
        L_dir = normalize(L_vec)

        # Calculate soft shadoes to find light intensity.
        light_intensity_factor = calculate_soft_shadow(hit_point_offset, light, surfaces, int(settings.root_number_shadow_rays))
        intensity = (1 - light.shadow_intensity) + (light.shadow_intensity * light_intensity_factor)
        
        # Object is not obscured.
        if intensity > 0:
            light_color = np.array(light.color)
            N_dot_L = max(0, np.dot(normal, L_dir))
            # Find diffuse contribution.
            diffuse_contribution = light_color * np.array(mat.diffuse_color) * N_dot_L
            
            R_dir = normalize(2 * np.dot(normal, L_dir) * normal - L_dir)
            V_dir = normalize(-ray_dir)
            R_dot_V = max(0, np.dot(R_dir, V_dir))
            
            # Find specular contribution
            specular_factor = pow(R_dot_V, mat.shininess) * light.specular_intensity
            specular_contribution = light_color * np.array(mat.specular_color) * specular_factor

            diffuse_final += diffuse_contribution * intensity
            specular_final += specular_contribution * intensity

    # Calculate reflection color.
    reflection_color = np.zeros(3)
    if np.linalg.norm(mat.reflection_color) > 0:
        ref_dir = normalize(ray_dir - 2 * np.dot(ray_dir, normal) * normal)
        rec_color = cast_ray(hit_point_offset, ref_dir, surfaces, materials, lights, settings, recursion_level + 1)
        reflection_color = rec_color * np.array(mat.reflection_color)

    # Calculate transsparency color.
    transparency_color = np.zeros(3)
    if mat.transparency > 0:
        pass_through_origin = hit_point + ray_dir * EPSILON 
        transparency_color = cast_ray(pass_through_origin, ray_dir, surfaces, materials, lights, settings, recursion_level + 1)

    # Calculate final color.
    color = (transparency_color * mat.transparency) + \
            (diffuse_final + specular_final) * (1 - mat.transparency) + \
            reflection_color
    return color

def main():
    parser = argparse.ArgumentParser(description='Python Ray Tracer')
    parser.add_argument('scene_file', type=str, help='Path to the scene file')
    parser.add_argument('output_image', type=str, help='Name of the output image file')
    parser.add_argument('--width', type=int, default=500, help='Image width')
    parser.add_argument('--height', type=int, default=500, help='Image height')
    args = parser.parse_args()

    camera, scene_settings, objects = parse_scene_file(args.scene_file)

    materials = [obj for obj in objects if isinstance(obj, Material)]
    lights = [obj for obj in objects if isinstance(obj, Light)]
    surfaces = [obj for obj in objects if isinstance(obj, Surface)]

    width, height = args.width, args.height
    final_image = np.zeros((height, width, 3))

    # print(f"Starting render on a single processor...")
    # start_time = time.time()

    for y in range(height):
        for x in range(width):
            ray_origin, ray_dir = camera.get_ray(x, y, width, height)
            pixel_color = cast_ray(ray_origin, ray_dir, surfaces, materials, lights, scene_settings, 0)
            final_image[y, x] = pixel_color
        
    #     if y % 50 == 0:
    #         print(f"Row {y}/{height} done...")

    # end_time = time.time()
    # print(f"Render time: {end_time - start_time:.4f} seconds")
    final_image=np.clip(final_image, 0,1)*255

    save_image(final_image, args.output_image)

if __name__ == '__main__':
    main()
