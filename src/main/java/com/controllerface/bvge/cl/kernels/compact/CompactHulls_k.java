package com.controllerface.bvge.cl.kernels.compact;

import com.controllerface.bvge.cl.CLUtils;
import com.controllerface.bvge.cl.kernels.GPUKernel;
import com.controllerface.bvge.cl.kernels.Kernel;
import com.controllerface.bvge.cl.kernels.crud.KernelArg;

public class CompactHulls_k extends GPUKernel
{
    public static final String kernel_source = CLUtils.compact_k_src(Kernel.compact_hulls, Args.class);

    public enum Args implements KernelArg
    {
        hull_shift          (Type.buffer_int),
        hulls               (Type.buffer_float4),
        hull_scales         (Type.buffer_float2),
        hull_mesh_ids       (Type.buffer_int),
        hull_uv_offsets     (Type.buffer_int),
        hull_rotations      (Type.buffer_float2),
        hull_frictions      (Type.buffer_float),
        hull_restitutions   (Type.buffer_float),
        hull_integrity      (Type.buffer_int),
        hull_bone_tables    (Type.buffer_int2),
        hull_entity_ids     (Type.buffer_int),
        hull_flags          (Type.buffer_int),
        hull_point_tables   (Type.buffer_int2),
        hull_edge_tables    (Type.buffer_int2),
        hull_aabb           (Type.buffer_float4),
        hull_aabb_index     (Type.buffer_int4),
        hull_aabb_key_table (Type.buffer_int2),

        ;

        private final String cl_type;
        Args(String clType) { cl_type = clType; }
        public String cl_type() { return cl_type; }
    }

    public CompactHulls_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
