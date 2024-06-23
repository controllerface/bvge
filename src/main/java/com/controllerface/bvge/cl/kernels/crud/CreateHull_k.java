package com.controllerface.bvge.cl.kernels.crud;

import com.controllerface.bvge.cl.CLUtils;
import com.controllerface.bvge.cl.kernels.GPUKernel;
import com.controllerface.bvge.cl.kernels.Kernel;

public class CreateHull_k extends GPUKernel
{
    public static final String kernel_source = CLUtils.crud_create_k_src(Kernel.create_hull, Args.class);

    public enum Args implements KernelArg
    {
        hulls              (Type.buffer_float4),
        hull_scales        (Type.buffer_float2),
        hull_rotations     (Type.buffer_float2),
        hull_frictions     (Type.buffer_float),
        hull_restitutions  (Type.buffer_float),
        hull_point_tables  (Type.buffer_int2),
        hull_edge_tables   (Type.buffer_int2),
        hull_bone_tables   (Type.buffer_int2),
        hull_entity_ids    (Type.buffer_int),
        hull_flags         (Type.buffer_int),
        hull_mesh_ids      (Type.buffer_int),
        hull_uv_offsets    (Type.buffer_int),
        hull_integrity     (Type.buffer_int),
        target             (Type.arg_int),
        new_hull           (Type.arg_float4),
        new_hull_scale     (Type.arg_float2),
        new_rotation       (Type.arg_float2),
        new_friction       (Type.arg_float),
        new_restitution    (Type.arg_float),
        new_point_table    (Type.arg_int2),
        new_edge_table     (Type.arg_int2),
        new_bone_table     (Type.arg_int2),
        new_entity_id      (Type.arg_int),
        new_flags          (Type.arg_int),
        new_hull_mesh_id   (Type.arg_int),
        new_hull_uv_offset (Type.arg_int),
        new_hull_integrity (Type.arg_int),

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
