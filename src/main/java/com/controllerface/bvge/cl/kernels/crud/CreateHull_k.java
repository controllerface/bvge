package com.controllerface.bvge.cl.kernels.crud;

import com.controllerface.bvge.cl.CLUtils;
import com.controllerface.bvge.cl.kernels.GPUKernel;
import com.controllerface.bvge.cl.kernels.Kernel;
import com.controllerface.bvge.cl.kernels.KernelArg;

import static com.controllerface.bvge.cl.CLData.*;

public class CreateHull_k extends GPUKernel
{
    public static final String kernel_source = CLUtils.crud_create_k_src(Kernel.create_hull, Args.class);

    public enum Args implements KernelArg
    {
        hulls              (cl_float4.buffer_name()),
        hull_scales        (cl_float2.buffer_name()),
        hull_rotations     (cl_float2.buffer_name()),
        hull_frictions     (cl_float.buffer_name()),
        hull_restitutions  (cl_float.buffer_name()),
        hull_point_tables  (cl_int2.buffer_name()),
        hull_edge_tables   (cl_int2.buffer_name()),
        hull_bone_tables   (cl_int2.buffer_name()),
        hull_entity_ids    (cl_int.buffer_name()),
        hull_flags         (cl_int.buffer_name()),
        hull_mesh_ids      (cl_int.buffer_name()),
        hull_uv_offsets    (cl_int.buffer_name()),
        hull_integrity     (cl_int.buffer_name()),
        target             (cl_int.name()),
        new_hull           (cl_float4.name()),
        new_hull_scale     (cl_float2.name()),
        new_rotation       (cl_float2.name()),
        new_friction       (cl_float.name()),
        new_restitution    (cl_float.name()),
        new_point_table    (cl_int2.name()),
        new_edge_table     (cl_int2.name()),
        new_bone_table     (cl_int2.name()),
        new_entity_id      (cl_int.name()),
        new_flags          (cl_int.name()),
        new_hull_mesh_id   (cl_int.name()),
        new_hull_uv_offset (cl_int.name()),
        new_hull_integrity (cl_int.name()),

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
