package com.controllerface.bvge.cl;

import org.jocl.CL;
import org.jocl.cl_kernel;
import org.jocl.cl_program;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static com.controllerface.bvge.cl.CLUtils.read_src;
import static com.controllerface.bvge.cl.GPU.*;
import static org.jocl.CL.clReleaseProgram;

/**
 * An abstraction for general-purpose GPU programs. Implementations of various programs that
 * execute on the GPU (as opposed to the CPU) extend this class and load the necessary source
 * files that define the program. Unlike code that runs on the CPU as part of the main program
 * (in this case, Java) GPU programs are written in C, which is compiled and executed at runtime.
 * Communication across the CPU/GPU boundary is achieved using the Open CL API. In order to
 * send data to and from the GPU, data buffers are used which allow host code to push raw data
 * into the GPU, as well as read it back. In the simplest scenario, CPU code creates a buffer
 * of data that needs to be accessible to a GPU kernel function and transfers it in to the GPU
 * before calling the kernel. Afterward, any values that were defined as output variables can
 * be transferred back out to the CPU.
 */
public abstract class GPUProgram
{
    /**
     * Constant values that can be used within kernels. There is a general expectation that the CPU
     * code will define matching constants for easy interoperability.
     * todo: generate the GPU code from the CPU constants instead of duplicating the code in both places
     */
    protected static String const_hull_flags              = read_src("constants/hull_flags.cl");

    /**
     * Helper functions. Program implementations can use these functions to build out a program,
     * in addition to the main source file of the program. This allows for utility functions to be
     * reused and also kept separate from the core kernel code, which helps keep these functions
     * focused on a single task. See the function source files themselves for usage information.
     */
    protected static String func_angle_between            = read_src("functions/angle_between.cl");
    protected static String func_calculate_centroid       = read_src("functions/calculate_centroid.cl");
    protected static String func_calculate_key_index      = read_src("functions/calculate_key_index.cl");
    protected static String func_circle_collision         = read_src("functions/circle_collision.cl");
    protected static String func_closest_point_circle     = read_src("functions/closest_point_circle.cl");
    protected static String func_do_bounds_intersect      = read_src("functions/do_bounds_intersect.cl");
    protected static String func_edge_contact             = read_src("functions/edge_contact.cl");
    protected static String func_exclusive_scan           = read_src("functions/exclusive_scan.cl");
    protected static String func_get_extents              = read_src("functions/get_extents.cl");
    protected static String func_get_key_for_point        = read_src("functions/get_key_for_point.cl");
    protected static String func_is_in_bounds             = read_src("functions/is_in_bounds.cl");
    protected static String func_matrix_transform         = read_src("functions/matrix_transform.cl");
    protected static String func_polygon_circle_collision = read_src("functions/polygon_circle_collision.cl");
    protected static String func_polygon_collision        = read_src("functions/polygon_collision.cl");
    protected static String func_polygon_distance         = read_src("functions/polygon_distance.cl");
    protected static String func_project_circle           = read_src("functions/project_circle.cl");
    protected static String func_project_polygon          = read_src("functions/project_polygon.cl");
    protected static String func_rotate_point             = read_src("functions/rotate_point.cl");

    /**
     * Simple header line to ensure atomics are enabled. May not be required in more modern drivers, but
     * worth adding as a best practice.
     */
    protected static String prag_int32_base_atomics       = read_src("pragma/int32_base_atomics.cl");

    /**
     * This is the backing Open CL program the implementation class wraps.
     */
    protected cl_program program;

    /**
     * After init is called, this will contain all the Open CL kernels that are defined in the program
     */
    protected Map<Kernel, cl_kernel> kernels = new HashMap<>();

    /**
     * Contains the raw source data of the program, in compilation order.
     */
    protected List<String> src = new ArrayList<>();

    /**
     * Signals the program implementation to compile itself, and load any kernels into the kernel map.
     */
    protected abstract void init();

    /**
     * Accessor for loaded kernels.
     *
     * @return the map of the currently loaded kernels.
     */
    public Map<Kernel, cl_kernel> kernels()
    {
        return kernels;
    }

    /**
     * Compiles this program, making the kernels it provides ready for use in an Open CL context.
     */
    protected void make_program()
    {
        this.program = gpu_p(this.src);
    }

    /**
     * Loads a kernel from this program into the internal kernels map, making it ready for use in
     * an Open CL context. It is the responsibility of the implementation to ensure that the loaded
     * program has a defined kernel entry point that matches the name of the Kernel object that is
     * passed in.
     *
     * @param kernel Kernel enum type to be loaded.
     */
    protected void load_kernel(Kernel kernel)
    {
        this.kernels.put(kernel, CLUtils.cl_k(program, kernel.name()));
    }

    /**
     * Release the resources associated with this program and the kernels that were loaded from it.
     */
    public void destroy()
    {
        clReleaseProgram(program);
        for (cl_kernel clKernel : kernels.values())
        {
            CL.clReleaseKernel(clKernel);
        }
    }
}
