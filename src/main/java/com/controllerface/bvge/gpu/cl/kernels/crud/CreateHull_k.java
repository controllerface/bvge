package com.controllerface.bvge.gpu.cl.kernels.crud;

import com.controllerface.bvge.gpu.GPU;
import com.controllerface.bvge.gpu.cl.contexts.CL_CommandQueue;
import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;
import com.controllerface.bvge.gpu.cl.kernels.KernelArg;
import com.controllerface.bvge.gpu.cl.kernels.KernelType;
import com.controllerface.bvge.gpu.cl.programs.GPUProgram;
import com.controllerface.bvge.memory.groups.CoreBufferGroup;

import static com.controllerface.bvge.gpu.cl.buffers.CL_DataTypes.*;
import static com.controllerface.bvge.memory.types.CoreBufferType.*;

public class CreateHull_k extends GPUKernel
{
    public static final String kernel_source = GPU.CL.crud_create_k_src(KernelType.create_hull, Args.class);

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
        Args(String clType) { cl_type = clType; }
        public String cl_type() { return cl_type; }
    }

    public CreateHull_k(CL_CommandQueue command_queue_ptr, GPUProgram program)
    {
        super(command_queue_ptr, program.get_kernel(KernelType.create_hull));
    }

    public GPUKernel init(CoreBufferGroup core_buffers)
    {
        return this.buf_arg(Args.hulls, core_buffers.buffer(HULL))
            .buf_arg(Args.hull_scales, core_buffers.buffer(HULL_SCALE))
            .buf_arg(Args.hull_rotations, core_buffers.buffer(HULL_ROTATION))
            .buf_arg(Args.hull_frictions, core_buffers.buffer(HULL_FRICTION))
            .buf_arg(Args.hull_restitutions, core_buffers.buffer(HULL_RESTITUTION))
            .buf_arg(Args.hull_point_tables, core_buffers.buffer(HULL_POINT_TABLE))
            .buf_arg(Args.hull_edge_tables, core_buffers.buffer(HULL_EDGE_TABLE))
            .buf_arg(Args.hull_bone_tables, core_buffers.buffer(HULL_BONE_TABLE))
            .buf_arg(Args.hull_entity_ids, core_buffers.buffer(HULL_ENTITY_ID))
            .buf_arg(Args.hull_flags, core_buffers.buffer(HULL_FLAG))
            .buf_arg(Args.hull_mesh_ids, core_buffers.buffer(HULL_MESH_ID))
            .buf_arg(Args.hull_uv_offsets, core_buffers.buffer(HULL_UV_OFFSET))
            .buf_arg(Args.hull_integrity, core_buffers.buffer(HULL_INTEGRITY));
    }
}
