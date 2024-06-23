package com.controllerface.bvge.cl.kernels.crud;

import com.controllerface.bvge.cl.CLUtils;
import com.controllerface.bvge.cl.kernels.GPUKernel;
import com.controllerface.bvge.cl.kernels.Kernel;

public class CreatePoint_k extends GPUKernel
{
    public static final String kernel_source = CLUtils.crud_create_k_src(Kernel.create_point, Args.class);

    public enum Args implements KernelArg
    {
        points                     (Type.buffer_float4),
        point_vertex_references    (Type.buffer_int),
        point_hull_indices         (Type.buffer_int),
        point_hit_counts           (Type.buffer_short),
        point_flags                (Type.buffer_int),
        point_bone_tables          (Type.buffer_int4),
        target                     (Type.arg_int),
        new_point                  (Type.arg_float4),
        new_point_vertex_reference (Type.arg_int),
        new_point_hull_index       (Type.arg_int),
        new_point_hit_count        (Type.arg_short),
        new_point_flags            (Type.arg_int),
        new_point_bone_table       (Type.arg_int4),

        ;

        private final String cl_type;
        Args(String clType) { cl_type = clType; }
        public String cl_type() { return cl_type; }
    }

    public CreatePoint_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
