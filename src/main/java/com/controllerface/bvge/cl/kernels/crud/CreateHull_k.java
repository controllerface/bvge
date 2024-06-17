package com.controllerface.bvge.cl.kernels.crud;

import com.controllerface.bvge.cl.CLUtils;
import com.controllerface.bvge.cl.kernels.GPUKernel;
import com.controllerface.bvge.cl.kernels.Kernel;

public class CreateHull_k extends GPUKernel
{
    public static final String kernel_source = CLUtils.crud_k_src(Kernel.create_hull, Args.class);

    public enum Args implements KernelArg
    {
        hulls              (Type.float4_buffer),
        hull_scales        (Type.float2_buffer),
        hull_rotations     (Type.float2_buffer),
        hull_frictions     (Type.float_buffer),
        hull_restitutions  (Type.float_buffer),
        hull_point_tables  (Type.int2_buffer),
        hull_edge_tables   (Type.int2_buffer),
        hull_bone_tables   (Type.int2_buffer),
        hull_entity_ids    (Type.int_buffer),
        hull_flags         (Type.int_buffer),
        hull_mesh_ids      (Type.int_buffer),
        hull_uv_offsets    (Type.int_buffer),
        hull_integrity     (Type.int_buffer),
        target             (Type.int_arg),
        new_hull           (Type.float4_arg),
        new_hull_scale     (Type.float2_arg),
        new_rotation       (Type.float2_arg),
        new_friction       (Type.float_arg),
        new_restitution    (Type.float_arg),
        new_point_table    (Type.int2_arg),
        new_edge_table     (Type.int2_arg),
        new_bone_table     (Type.int2_arg),
        new_entity_id      (Type.int_arg),
        new_flags          (Type.int_arg),
        new_hull_mesh_id   (Type.int_arg),
        new_hull_uv_offset (Type.int_arg),
        new_hull_integrity (Type.int_arg),

        ;

        private final String cl_type;
        Args(String clType) {cl_type = clType;}
        public String cl_type() { return cl_type; }
    }

    public CreateHull_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
