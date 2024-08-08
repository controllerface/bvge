package com.controllerface.bvge.gpu.cl.kernels.crud;

import com.controllerface.bvge.gpu.GPU;
import com.controllerface.bvge.gpu.cl.contexts.CL_CommandQueue;
import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;
import com.controllerface.bvge.gpu.cl.kernels.KernelArg;
import com.controllerface.bvge.gpu.cl.kernels.KernelType;
import com.controllerface.bvge.gpu.cl.programs.GPUProgram;

import static com.controllerface.bvge.gpu.cl.buffers.CL_DataTypes.*;

public class CreatePoint_k extends GPUKernel
{
    public static final String kernel_source = GPU.CL.crud_create_k_src(KernelType.create_point, Args.class);

    public enum Args implements KernelArg
    {
        points                     (cl_float4.buffer_name()),
        point_vertex_references    (cl_int.buffer_name()),
        point_hull_indices         (cl_int.buffer_name()),
        point_hit_counts           (cl_short.buffer_name()),
        point_flags                (cl_int.buffer_name()),
        point_bone_tables          (cl_int4.buffer_name()),
        target                     (cl_int.name()),
        new_point                  (cl_float4.name()),
        new_point_vertex_reference (cl_int.name()),
        new_point_hull_index       (cl_int.name()),
        new_point_hit_count        (cl_short.name()),
        new_point_flags            (cl_int.name()),
        new_point_bone_table       (cl_int4.name()),

        ;

        private final String cl_type;
        Args(String clType) { cl_type = clType; }
        public String cl_type() { return cl_type; }
    }

    public CreatePoint_k(CL_CommandQueue command_queue_ptr, GPUProgram program)
    {
        super(command_queue_ptr, program.get_kernel(KernelType.create_point));
    }
}
