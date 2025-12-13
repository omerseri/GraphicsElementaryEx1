import argparse
from PIL import Image
import numpy as np
import random
import multiprocessing

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
    
    for obj in surfaces:
        t, normal = obj.intersect(ray_origin, ray_dir)
        if t < nearest_t:
            nearest_t = t
            nearest_obj = obj
            nearest_normal = normal

    return nearest_t, nearest_obj, nearest_normal

def calculate_soft_shadow(hit_point, light, surfaces, root_shadow_rays):
    light_pos = np.array(light.position)
    to_light = light_pos - hit_point
    light_dir = normalize(to_light)

    helper = np.array([0, 1, 0])
    if abs(np.dot(helper, light_dir)) > 0.99:
        helper = np.array([1, 0, 0])
    
    light_u = normalize(np.cross(helper, light_dir))
    light_v = normalize(np.cross(light_dir, light_u))

    rect_width = light.radius
    cell_size = rect_width / root_shadow_rays
    
    rays_hit_light = 0.0
    total_rays = root_shadow_rays * root_shadow_rays
    
    start_point = light_pos - (light_u * rect_width / 2) - (light_v * rect_width / 2)

    for i in range(root_shadow_rays):
        for j in range(root_shadow_rays):
            rand_u = random.random()
            rand_v = random.random()
            
            cell_center_u = (i + rand_u) * cell_size
            cell_center_v = (j + rand_v) * cell_size
            
            sample_point = start_point + (light_u * cell_center_u) + (light_v * cell_center_v)
            
            shadow_vec = sample_point - hit_point
            dist_to_sample = np.linalg.norm(shadow_vec)
            shadow_dir = normalize(shadow_vec)
            
            nearest_t, _, _ = find_nearest_object(hit_point, shadow_dir, surfaces)
            
            if nearest_t >= dist_to_sample - EPSILON: 
                rays_hit_light += 1.0

    return rays_hit_light / total_rays

def cast_ray(ray_origin, ray_dir, surfaces, materials, lights, settings, recursion_level):
    if recursion_level > settings.max_recursions:
        return np.array(settings.background_color)

    t, hit_obj, normal = find_nearest_object(ray_origin, ray_dir, surfaces)

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

        light_intensity_factor = calculate_soft_shadow(hit_point_offset, light, surfaces, int(settings.root_number_shadow_rays))
        intensity = (1 - light.shadow_intensity) + (light.shadow_intensity * light_intensity_factor)
        
        if intensity > 0:
            light_color = np.array(light.color)
            N_dot_L = max(0, np.dot(normal, L_dir))

            # I_diff = K_d * I_p * (N · L)
            diffuse_contribution = light_color * np.array(mat.diffuse_color) * N_dot_L
            
        
            R_dir = normalize(2 * np.dot(normal, L_dir) * normal - L_dir)
            V_dir = normalize(-ray_dir)
            R_dot_V = max(0, np.dot(R_dir, V_dir))
            
            # I_spec = K_s * I_p * (R · V)^n * specular_intensity
            specular_factor = pow(R_dot_V, mat.shininess) * light.specular_intensity
            specular_contribution = light_color * np.array(mat.specular_color) * specular_factor

            diffuse_final += diffuse_contribution * intensity
            specular_final += specular_contribution * intensity

    reflection_color = np.zeros(3)
    if np.linalg.norm(mat.reflection_color) > 0:
        # R = V - 2(V · N)N
        ref_dir = normalize(ray_dir - 2 * np.dot(ray_dir, normal) * normal)
        rec_color = cast_ray(hit_point_offset, ref_dir, surfaces, materials, lights, settings, recursion_level + 1)
        reflection_color = rec_color * np.array(mat.reflection_color)

    transparency_color = np.zeros(3)
    if mat.transparency > 0:
        pass_through_origin = hit_point + ray_dir * EPSILON 
        transparency_color = cast_ray(pass_through_origin, ray_dir, surfaces, materials, lights, settings, recursion_level + 1)

    color = (transparency_color * mat.transparency) + \
            (diffuse_final + specular_final) * (1 - mat.transparency) + \
            reflection_color
    return np.clip(color, 0, 1)

def render_chunk(y_start, y_end, width, height, camera, surfaces, materials, lights, scene_settings):
    # Create a local buffer for this chunk of the image
    chunk_height = y_end - y_start
    chunk_image = np.zeros((chunk_height, width, 3))
    
    for i in range(chunk_height):
        y = y_start + i
        for x in range(width):
            ray_origin, ray_dir = camera.get_ray(x, y, width, height)
            
            pixel_color = cast_ray(ray_origin, ray_dir, surfaces, materials, lights, scene_settings, 0)
            chunk_image[i, x] = 255 * pixel_color
            
    return (y_start, chunk_image)

def main():
    parser = argparse.ArgumentParser(description='Python Ray Tracer')
    parser.add_argument('scene_file', type=str, help='Path to the scene file')
    parser.add_argument('output_image', type=str, help='Name of the output image file')
    parser.add_argument('--width', type=int, default=500, help='Image width')
    parser.add_argument('--height', type=int, default=500, help='Image height')
    args = parser.parse_args()

    camera, scene_settings, objects = parse_scene_file(args.scene_file)

    # Filter objects by type
    materials = [obj for obj in objects if isinstance(obj, Material)]
    lights = [obj for obj in objects if isinstance(obj, Light)]
    surfaces = [obj for obj in objects if isinstance(obj, Surface)]

    # Determine how many CPUs we have
    num_processes = multiprocessing.cpu_count()

    # Calculate chunk size (how many rows per process)
    chunk_size = args.height // num_processes
    
    # Create a list of arguments for each process
    # Each process gets a different y_start and y_end
    tasks = []
    for i in range(num_processes):
        y_start = i * chunk_size
        # The last process takes any remaining rows (in case height isn't divisible by num_processes)
        if i == num_processes - 1:
            y_end = args.height
        else:
            y_end = (i + 1) * chunk_size
        
        tasks.append((y_start, y_end, args.width, args.height, camera, surfaces, materials, lights, scene_settings))

    # Create the Pool and Run tasks
    with multiprocessing.Pool(processes=num_processes) as pool:
        # starmap unpacks the arguments tuple into the function arguments
        results = pool.starmap(render_chunk, tasks)

    # Sort results by y_start to ensure correct order
    results.sort(key=lambda x: x[0])
    
    # Concatenate all the chunk arrays into one final image
    final_image = np.concatenate([r[1] for r in results], axis=0)
    save_image(final_image, args.output_image)

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()