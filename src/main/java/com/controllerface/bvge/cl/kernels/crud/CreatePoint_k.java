package com.controllerface.bvge.cl.kernels.crud;

import com.controllerface.bvge.cl.CLUtils;
import com.controllerface.bvge.cl.kernels.GPUKernel;
import com.controllerface.bvge.cl.kernels.Kernel;

public class CreatePoint_k extends GPUKernel
{
    public static final String kernel_source = CLUtils.crud_k_src(Kernel.create_point, Args.class);

    public enum Args implements KernelArg
    {
        points                     (Type.float4_buffer),
        point_vertex_references    (Type.int_buffer),
        point_hull_indices         (Type.int_buffer),
        point_hit_counts           (Type.short_buffer),
        point_flags                (Type.int_buffer),
        point_bone_tables          (Type.int4_buffer),
        target                     (Type.int_arg),
        new_point                  (Type.float4_arg),
        new_point_vertex_reference (Type.int_arg),
        new_point_hull_index       (Type.int_arg),
        new_point_hit_count        (Type.short_arg),
        new_point_flags            (Type.int_arg),
        new_point_bone_table       (Type.int4_arg),

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
