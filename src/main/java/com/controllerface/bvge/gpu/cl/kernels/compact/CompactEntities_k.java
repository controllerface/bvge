package com.controllerface.bvge.gpu.cl.kernels.compact;

import com.controllerface.bvge.gpu.cl.buffers.ResizableBuffer;
import com.controllerface.bvge.gpu.cl.contexts.CL_CommandQueue;
import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;
import com.controllerface.bvge.gpu.cl.kernels.KernelType;
import com.controllerface.bvge.gpu.cl.programs.GPUProgram;
import com.controllerface.bvge.memory.groups.CoreBufferGroup;

import static com.controllerface.bvge.memory.types.CoreBufferType.*;

public class CompactEntities_k extends GPUKernel
{
    public enum Args
    {
        buffer_in_1,
        buffer_in_2,
        entities,
        entity_masses,
        entity_root_hulls,
        entity_model_indices,
        entity_model_transforms,
        entity_types,
        entity_flags,
        entity_animation_layers,
        entity_previous_layers,
        entity_animation_time,
        entity_previous_time,
        entity_animation_blend,
        entity_motion_states,
        entity_entity_hull_tables,
        entity_bone_tables,
        hull_bone_tables,
        hull_entity_ids,
        hull_point_tables,
        hull_edge_tables,
        points,
        point_hull_indices,
        point_bone_tables,
        entity_bone_parent_ids,
        hull_bind_pose_indices,
        edges,
        hull_bone_shift,
        point_shift,
        edge_shift,
        hull_shift,
        entity_bone_shift,
    }

    public CompactEntities_k(CL_CommandQueue command_queue_ptr, GPUProgram program)
    {
        super(command_queue_ptr, program.get_kernel(KernelType.compact_entities));
    }

    public GPUKernel init(CoreBufferGroup sector_buffers,
                          ResizableBuffer b_entity_bone_shift,
                          ResizableBuffer b_hull_bone_shift,
                          ResizableBuffer b_edge_shift,
                          ResizableBuffer b_hull_shift,
                          ResizableBuffer b_point_shift)
    {
        return this.buf_arg(Args.entities, sector_buffers.buffer(ENTITY))
            .buf_arg(Args.entity_masses, sector_buffers.buffer(ENTITY_MASS))
            .buf_arg(Args.entity_root_hulls, sector_buffers.buffer(ENTITY_ROOT_HULL))
            .buf_arg(Args.entity_model_indices, sector_buffers.buffer(ENTITY_MODEL_ID))
            .buf_arg(Args.entity_model_transforms, sector_buffers.buffer(ENTITY_TRANSFORM_ID))
            .buf_arg(Args.entity_types, sector_buffers.buffer(ENTITY_TYPE))
            .buf_arg(Args.entity_flags, sector_buffers.buffer(ENTITY_FLAG))
            .buf_arg(Args.entity_animation_layers, sector_buffers.buffer(ENTITY_ANIM_LAYER))
            .buf_arg(Args.entity_previous_layers, sector_buffers.buffer(ENTITY_PREV_LAYER))
            .buf_arg(Args.entity_animation_time, sector_buffers.buffer(ENTITY_ANIM_TIME))
            .buf_arg(Args.entity_previous_time, sector_buffers.buffer(ENTITY_PREV_TIME))
            .buf_arg(Args.entity_animation_blend, sector_buffers.buffer(ENTITY_ANIM_BLEND))
            .buf_arg(Args.entity_motion_states, sector_buffers.buffer(ENTITY_MOTION_STATE))
            .buf_arg(Args.entity_entity_hull_tables, sector_buffers.buffer(ENTITY_HULL_TABLE))
            .buf_arg(Args.entity_bone_tables, sector_buffers.buffer(ENTITY_BONE_TABLE))
            .buf_arg(Args.hull_bone_tables, sector_buffers.buffer(HULL_BONE_TABLE))
            .buf_arg(Args.hull_entity_ids, sector_buffers.buffer(HULL_ENTITY_ID))
            .buf_arg(Args.hull_point_tables, sector_buffers.buffer(HULL_POINT_TABLE))
            .buf_arg(Args.hull_edge_tables, sector_buffers.buffer(HULL_EDGE_TABLE))
            .buf_arg(Args.points, sector_buffers.buffer(POINT))
            .buf_arg(Args.point_hull_indices, sector_buffers.buffer(POINT_HULL_INDEX))
            .buf_arg(Args.point_bone_tables, sector_buffers.buffer(POINT_BONE_TABLE))
            .buf_arg(Args.entity_bone_parent_ids, sector_buffers.buffer(ENTITY_BONE_PARENT_ID))
            .buf_arg(Args.hull_bind_pose_indices, sector_buffers.buffer(HULL_BONE_BIND_POSE))
            .buf_arg(Args.edges, sector_buffers.buffer(EDGE))
            .buf_arg(Args.hull_bone_shift, b_hull_bone_shift)
            .buf_arg(Args.point_shift, b_point_shift)
            .buf_arg(Args.edge_shift, b_edge_shift)
            .buf_arg(Args.hull_shift, b_hull_shift)
            .buf_arg(Args.entity_bone_shift, b_entity_bone_shift);
    }
}
