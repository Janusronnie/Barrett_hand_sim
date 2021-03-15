import sdfield_generator
import os
import numpy as np


def cal_dist(point_loc=[0.0, 0.0, 0.0]):
    point_local = list(np.array(point_loc) + np.array([-0.5, -0.5, -0.1]))
    if os.path.exists("sdf_obj_mesh/object.obj"):
        if os.path.exists("demo_sdf_files/object.pickle"):
            loaded_sdf = sdfield_generator.SDF_Generator("sdf_obj_mesh/object.obj",
                                                         file_name="demo_sdf_files/object.pickle")
            dist = loaded_sdf.interpolate_sdf_from_point(point_local)
            # loaded_sdf.visualize_sdf()
            return dist
        else:
            sdf = sdfield_generator.SDF_Generator("sdf_obj_mesh/object.obj", resolution=64, padding=1)
            distance_array, sdf_origin, sdf_spacing, sdf_resolution = sdf.calculate_sdf_array()
            sdf.create_sdf_file("demo_sdf_files/object.pickle")
            dist = sdf.interpolate_sdf_from_point(point_local)
            # sdf.visualize_sdf()
            return dist
    else:
        raise ValueError


if __name__ == "__main__":
    print(cal_dist())