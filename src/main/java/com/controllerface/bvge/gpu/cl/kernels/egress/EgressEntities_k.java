package com.controllerface.bvge.gpu.cl.kernels.egress;

import com.controllerface.bvge.gpu.cl.buffers.CL_Buffer;
import com.controllerface.bvge.gpu.cl.buffers.ResizableBuffer;
import com.controllerface.bvge.gpu.cl.contexts.CL_CommandQueue;
import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;
import com.controllerface.bvge.gpu.cl.kernels.KernelType;
import com.controllerface.bvge.gpu.cl.programs.GPUProgram;
import com.controllerface.bvge.memory.GPUCoreMemory;
import com.controllerface.bvge.memory.groups.UnorderedCoreBufferGroup;

import static com.controllerface.bvge.memory.types.CoreBufferType.*;

public class EgressEntities_k extends GPUKernel
{
    public enum Args
    {
        point_hull_indices_in,
        point_bone_tables_in,
        edges_in,
        edge_pins_in,
        hull_point_tables_in,
        hull_edge_tables_in,
        hull_bone_tables_in,
        hull_bind_pose_indices_in,
        entity_bone_parent_ids_in,

        entities_in,
        entity_animation_time_in,
        entity_previous_time_in,
        entity_motion_states_in,
        entity_animation_layers_in,
        entity_previous_layers_in,
        entity_hull_tables_in,
        entity_bone_tables_in,
        entity_masses_in,
        entity_root_hulls_in,
        entity_model_indices_in,
        entity_model_transforms_in,
        entity_types_in,
        entity_flags_in,

        entities_out,
        entity_animation_time_out,
        entity_previous_time_out,
        entity_motion_states_out,
        entity_animation_layers_out,
        entity_previous_layers_out,
        entity_hull_tables_out,
        entity_bone_tables_out,
        entity_masses_out,
        entity_root_hulls_out,
        entity_model_indices_out,
        entity_model_transforms_out,
        entity_types_out,
        entity_flags_out,

        new_points,
        new_edges,
        new_hulls,
        new_hull_bones,
        new_entity_bones,
        counters,
        max_entity,
    }

    public EgressEntities_k(CL_CommandQueue command_queue_ptr, GPUProgram program)
    {
        super(command_queue_ptr, program.get_kernel(KernelType.egress_entities));
    }

    public GPUKernel init(GPUCoreMemory core_memory,
                          UnorderedCoreBufferGroup sector_buffers,
                          ResizableBuffer b_point_shift,
                          ResizableBuffer b_edge_shift,
                          ResizableBuffer b_hull_shift,
                          ResizableBuffer b_hull_bone_shift,
                          ResizableBuffer b_entity_bone_shift,
                          CL_Buffer egress_sizes)
    {
        return this.buf_arg(Args.point_hull_indices_in, core_memory.get_buffer(POINT_HULL_INDEX))
            .buf_arg(Args.point_bone_tables_in, core_memory.get_buffer(POINT_BONE_TABLE))
            .buf_arg(Args.edges_in, core_memory.get_buffer(EDGE))
            .buf_arg(Args.edge_pins_in, core_memory.get_buffer(EDGE_PIN))
            .buf_arg(Args.hull_point_tables_in, core_memory.get_buffer(HULL_POINT_TABLE))
            .buf_arg(Args.hull_edge_tables_in, core_memory.get_buffer(HULL_EDGE_TABLE))
            .buf_arg(Args.hull_bone_tables_in, core_memory.get_buffer(HULL_BONE_TABLE))
            .buf_arg(Args.hull_bind_pose_indices_in, core_memory.get_buffer(HULL_BONE_BIND_POSE))
            .buf_arg(Args.entity_bone_parent_ids_in, core_memory.get_buffer(ENTITY_BONE_PARENT_ID))
            .buf_arg(Args.entities_in, core_memory.get_buffer(ENTITY))
            .buf_arg(Args.entity_animation_time_in, core_memory.get_buffer(ENTITY_ANIM_TIME))
            .buf_arg(Args.entity_previous_time_in, core_memory.get_buffer(ENTITY_PREV_TIME))
            .buf_arg(Args.entity_motion_states_in, core_memory.get_buffer(ENTITY_MOTION_STATE))
            .buf_arg(Args.entity_animation_layers_in, core_memory.get_buffer(ENTITY_ANIM_LAYER))
            .buf_arg(Args.entity_previous_layers_in, core_memory.get_buffer(ENTITY_PREV_LAYER))
            .buf_arg(Args.entity_hull_tables_in, core_memory.get_buffer(ENTITY_HULL_TABLE))
            .buf_arg(Args.entity_bone_tables_in, core_memory.get_buffer(ENTITY_BONE_TABLE))
            .buf_arg(Args.entity_masses_in, core_memory.get_buffer(ENTITY_MASS))
            .buf_arg(Args.entity_root_hulls_in, core_memory.get_buffer(ENTITY_ROOT_HULL))
            .buf_arg(Args.entity_model_indices_in, core_memory.get_buffer(ENTITY_MODEL_ID))
            .buf_arg(Args.entity_model_transforms_in, core_memory.get_buffer(ENTITY_TRANSFORM_ID))
            .buf_arg(Args.entity_types_in, core_memory.get_buffer(ENTITY_TYPE))
            .buf_arg(Args.entity_flags_in, core_memory.get_buffer(ENTITY_FLAG))
            .buf_arg(Args.entities_out, sector_buffers.buffer(ENTITY))
            .buf_arg(Args.entity_animation_time_out, sector_buffers.buffer(ENTITY_ANIM_TIME))
            .buf_arg(Args.entity_previous_time_out, sector_buffers.buffer(ENTITY_PREV_TIME))
            .buf_arg(Args.entity_motion_states_out, sector_buffers.buffer(ENTITY_MOTION_STATE))
            .buf_arg(Args.entity_animation_layers_out, sector_buffers.buffer(ENTITY_ANIM_LAYER))
            .buf_arg(Args.entity_previous_layers_out, sector_buffers.buffer(ENTITY_PREV_LAYER))
            .buf_arg(Args.entity_hull_tables_out, sector_buffers.buffer(ENTITY_HULL_TABLE))
            .buf_arg(Args.entity_bone_tables_out, sector_buffers.buffer(ENTITY_BONE_TABLE))
            .buf_arg(Args.entity_masses_out, sector_buffers.buffer(ENTITY_MASS))
            .buf_arg(Args.entity_root_hulls_out, sector_buffers.buffer(ENTITY_ROOT_HULL))
            .buf_arg(Args.entity_model_indices_out, sector_buffers.buffer(ENTITY_MODEL_ID))
            .buf_arg(Args.entity_model_transforms_out, sector_buffers.buffer(ENTITY_TRANSFORM_ID))
            .buf_arg(Args.entity_types_out, sector_buffers.buffer(ENTITY_TYPE))
            .buf_arg(Args.entity_flags_out, sector_buffers.buffer(ENTITY_FLAG))
            .buf_arg(Args.new_points, b_point_shift)
            .buf_arg(Args.new_edges, b_edge_shift)
            .buf_arg(Args.new_hulls, b_hull_shift)
            .buf_arg(Args.new_hull_bones, b_hull_bone_shift)
            .buf_arg(Args.new_entity_bones, b_entity_bone_shift)
            .buf_arg(Args.counters, egress_sizes);
    }
}
