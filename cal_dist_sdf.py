import sdfield_generator
import os


def cal_dist(point_loc=[0.0, 0.0, 0.0]):

    if os.path.exists("obj_mesh/object.obj"):
        if os.path.exists("demo_sdf_files/object.pickle"):
            loaded_sdf = sdfield_generator.SDF_Generator("obj_mesh/object.obj",
                                                         file_name="demo_sdf_files/object.pickle")
            distance_array, sdf_origin, sdf_spacing, sdf_resolution = loaded_sdf.get_sdf_properties()
            dist = loaded_sdf.interpolate_sdf_from_point(point_loc)
            # loaded_sdf.visualize_sdf()
            return dist
        else:
            sdf = sdfield_generator.SDF_Generator("obj_mesh/object.obj", resolution=64, padding=1)
            distance_array, sdf_origin, sdf_spacing, sdf_resolution = sdf.calculate_sdf_array()
            sdf.create_sdf_file("demo_sdf_files/object.pickle")
            dist = sdf.interpolate_sdf_from_point(point_loc)
            # sdf.visualize_sdf()
            return dist
    else:
        raise ValueError


if __name__ == "__main__":

    nearest_distance = cal_dist(point_loc=[0.0, 0.0, 0.084])
    print(nearest_distance)

