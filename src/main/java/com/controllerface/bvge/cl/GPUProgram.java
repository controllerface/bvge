package com.controllerface.bvge.cl;

import org.jocl.cl_kernel;
import org.jocl.cl_program;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static com.controllerface.bvge.cl.CLUtils.read_src;
import static com.controllerface.bvge.cl.GPU.*;

public abstract class GPUProgram
{
    /**
     * Helper functions. Program implementations can use these functions to build out a program,
     * in addition to the main source file of the program. This allows for utility functions to be
     * reused and also kept separate from the core kernel code, which helps keep these functions
     * focused on a single task. See the function source files themselves for usage information.
     */
    public static String func_is_in_bounds             = read_src("functions/is_in_bounds.cl");
    public static String func_get_extents              = read_src("functions/get_extents.cl");
    public static String func_get_key_for_point        = read_src("functions/get_key_for_point.cl");
    public static String func_calculate_key_index      = read_src("functions/calculate_key_index.cl");
    public static String func_exclusive_scan           = read_src("functions/exclusive_scan.cl");
    public static String func_do_bounds_intersect      = read_src("functions/do_bounds_intersect.cl");
    public static String func_project_polygon          = read_src("functions/project_polygon.cl");
    public static String func_project_circle           = read_src("functions/project_circle.cl");
    public static String func_polygon_distance         = read_src("functions/polygon_distance.cl");
    public static String func_edge_contact             = read_src("functions/edge_contact.cl");
    public static String func_rotate_point             = read_src("functions/rotate_point.cl");
    public static String func_angle_between            = read_src("functions/angle_between.cl");
    public static String func_closest_point_circle     = read_src("functions/closest_point_circle.cl");
    public static String func_matrix_transform         = read_src("functions/matrix_transform.cl");
    public static String func_calculate_centroid       = read_src("functions/calculate_centroid.cl");
    public static String func_polygon_collision        = read_src("functions/polygon_collision.cl");
    public static String func_circle_collision         = read_src("functions/circle_collision.cl");
    public static String func_polygon_circle_collision = read_src("functions/polygon_circle_collision.cl");

    public static String prag_int32_base_atomics = read_src("pragma/int32_base_atomics.cl");


    protected cl_program program;

    protected Map<Kernel, cl_kernel> kernels = new HashMap<>();

    protected List<String> source_files = new ArrayList<>();

    protected abstract void init();

    public Map<Kernel, cl_kernel> kernels()
    {
        return kernels;
    }

    protected void add_src(String src)
    {
        source_files.add(src);
    }

    protected void make_program()
    {
        this.program = cl_p(this.source_files);
    }

    protected void make_kernel(Kernel kernel_name)
    {
        this.kernels.put(kernel_name, cl_k(program, kernel_name.name()));
    }
}
