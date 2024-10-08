package com.controllerface.bvge.gpu.cl.programs;

import com.controllerface.bvge.game.Constants;
import com.controllerface.bvge.gpu.GPU;
import com.controllerface.bvge.gpu.GPUResource;
import com.controllerface.bvge.gpu.cl.kernels.CL_Kernel;
import com.controllerface.bvge.gpu.cl.kernels.KernelType;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

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
public abstract class GPUProgram implements GPUResource
{
    /**
     * Constant values that can be used within kernels. There is a general expectation that the CPU
     * code will define matching constants for easy interoperability.
     * todo: generate the GPU code from the CPU constants instead of duplicating the code in both places
     */
    protected static String const_entity_flags             = Constants.entity_flags_src();
    protected static String const_hull_flags               = Constants.hull_flags_src();
    protected static String const_point_flags              = Constants.point_flags_src();
    protected static String const_edge_flags               = Constants.edge_flags_src();
    protected static String const_hit_thresholds           = GPU.CL.read_src("constants/hit_thresholds.cl");
    protected static String const_identity_matrix          = GPU.CL.read_src("constants/identity_matrix.cl");

    /**
     * Helper functions. Program implementations can use these functions to build out a program,
     * in addition to the main source file of the program. This allows for utility functions to be
     * reused and also kept separate from the core kernel code, which helps keep these functions
     * focused on a single task. See the function source files themselves for usage information.
     */
    protected static String func_angle_between             = GPU.CL.read_src("functions/angle_between.cl");
    protected static String func_calculate_centroid        = GPU.CL.read_src("functions/calculate_centroid.cl");
    protected static String func_calculate_key_index       = GPU.CL.read_src("functions/calculate_key_index.cl");
    protected static String func_circle_collision          = GPU.CL.read_src("functions/circle_collision.cl");
    protected static String func_closest_point_circle      = GPU.CL.read_src("functions/closest_point_circle.cl");
    protected static String func_do_bounds_intersect       = GPU.CL.read_src("functions/do_bounds_intersect.cl");
    protected static String func_edge_contact              = GPU.CL.read_src("functions/edge_contact.cl");
    protected static String func_exclusive_scan            = GPU.CL.read_src("functions/exclusive_scan.cl");
    protected static String func_get_extents               = GPU.CL.read_src("functions/get_extents.cl");
    protected static String func_get_key_for_point         = GPU.CL.read_src("functions/get_key_for_point.cl");
    protected static String func_is_in_bounds              = GPU.CL.read_src("functions/is_in_bounds.cl");
    protected static String func_matrix_mul_affine         = GPU.CL.read_src("functions/matrix_multiply_affine.cl");
    protected static String func_matrix_multiply           = GPU.CL.read_src("functions/matrix_multiply.cl");
    protected static String func_matrix_transform          = GPU.CL.read_src("functions/matrix_transform.cl");
    protected static String func_polygon_circle_collision  = GPU.CL.read_src("functions/polygon_circle_collision.cl");
    protected static String func_polygon_collision         = GPU.CL.read_src("functions/polygon_collision.cl");
    protected static String func_sensor_collision          = GPU.CL.read_src("functions/sensor_collision.cl");
    protected static String func_block_collision           = GPU.CL.read_src("functions/block_collision.cl");
    protected static String func_polygon_distance          = GPU.CL.read_src("functions/polygon_distance.cl");
    protected static String func_point_polygon_containment = GPU.CL.read_src("functions/point_polygon_containment.cl");
    protected static String func_pos_vector_to_matrix      = GPU.CL.read_src("functions/translation_vector_to_matrix.cl");
    protected static String func_project_circle            = GPU.CL.read_src("functions/project_circle.cl");
    protected static String func_project_polygon           = GPU.CL.read_src("functions/project_polygon.cl");
    protected static String func_quaternion_lerp           = GPU.CL.read_src("functions/quaternion_lerp.cl");
    protected static String func_rot_quaternion_to_matrix  = GPU.CL.read_src("functions/rotation_quaternion_to_matrix.cl");
    protected static String func_rotate_point              = GPU.CL.read_src("functions/rotate_point.cl");
    protected static String func_scl_vector_to_matrix      = GPU.CL.read_src("functions/scaling_vector_to_matrix.cl");
    protected static String func_vector_lerp               = GPU.CL.read_src("functions/vector_lerp.cl");

    /**
     * Simple header line to ensure atomics are enabled. May not be required in more modern drivers, but
     * worth adding as a best practice.
     */
    protected static String prag_int32_base_atomics       = GPU.CL.read_src("pragma/int32_base_atomics.cl");

    /**
     * This is the backing Open CL program the implementation class wraps.
     */
    protected CL_Program program;

    /**
     * After init is called, this will contain all the Open CL kernels that are defined in the program
     */
    protected Map<KernelType, CL_Kernel> kernels = new HashMap<>();

    /**
     * Contains the raw source data of the program, in compilation order.
     */
    protected List<String> src = new ArrayList<>();

    /**
     * Signals the program implementation to compile itself, and load any kernels into the kernel map.
     */
    public abstract GPUProgram init();

    /**
     * Compiles this program, making the kernels it provides ready for use in an Open CL context.
     */
    protected void make_program()
    {
        this.program = GPU.CL.new_program(GPU.compute.context, GPU.compute.device, this.src);
    }

    /**
     * Loads a kernel from this program into the internal kernels map, making it ready for use in
     * an Open CL context. It is the responsibility of the implementation to ensure that the loaded
     * program has a defined kernel entry point that matches the name of the Kernel object that is
     * passed in.
     *
     * @param kernel Kernel enum type to be loaded.
     */
    protected void load_kernel(KernelType kernel)
    {
        this.kernels.put(kernel, GPU.CL.new_kernel(program, kernel.name()));
    }

    /**
     * Release the resources associated with this program and the kernels that were loaded from it.
     */
    public void release()
    {
        program.release();
        for (var kernel : kernels.values())
        {
            kernel.release();
        }
    }

    public CL_Kernel get_kernel(KernelType kernel)
    {
        return kernels.get(kernel);
    }
}
