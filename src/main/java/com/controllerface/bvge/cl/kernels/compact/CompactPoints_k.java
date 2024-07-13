package com.controllerface.bvge.cl.kernels.compact;

import com.controllerface.bvge.cl.CLUtils;
import com.controllerface.bvge.cl.kernels.GPUKernel;
import com.controllerface.bvge.cl.kernels.Kernel;
import com.controllerface.bvge.cl.kernels.crud.KernelArg;

public class CompactPoints_k extends GPUKernel
{
    public static final String kernel_source = CLUtils.compact_k_src(Kernel.compact_points, Args.class);

    public enum Args implements KernelArg
    {
        point_shift             (Type.buffer_int),
        points                  (Type.buffer_float4),
        anti_gravity            (Type.buffer_float),
        anti_time               (Type.buffer_float2),
        point_vertex_references (Type.buffer_int),
        point_hull_indices      (Type.buffer_int),
        point_flags             (Type.buffer_int),
        point_hit_counts        (Type.buffer_short),
        bone_tables             (Type.buffer_int4),

        ;

        private final String cl_type;
        Args(String clType) { cl_type = clType; }
        public String cl_type() { return cl_type; }
    }

    public CompactPoints_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
