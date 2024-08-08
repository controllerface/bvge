package com.controllerface.bvge.gpu.cl.kernels.compact;

import com.controllerface.bvge.gpu.cl.contexts.CL_CommandQueue;
import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;
import com.controllerface.bvge.gpu.cl.kernels.KernelType;
import com.controllerface.bvge.gpu.cl.programs.GPUProgram;

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
        entity_bone_shift;
    }

    public CompactEntities_k(CL_CommandQueue command_queue_ptr, GPUProgram program)
    {
        super(command_queue_ptr, program.get_kernel(KernelType.compact_entities));
    }
}
