package com.controllerface.bvge.cl.kernels.crud;

import com.controllerface.bvge.cl.CLUtils;
import com.controllerface.bvge.cl.kernels.GPUKernel;
import com.controllerface.bvge.cl.kernels.Kernel;

public class CreateEntity_k extends GPUKernel
{
    public static final String kernel_source = CLUtils.crud_k_src(Kernel.create_entity, Args.class);

    public enum Args implements KernelArg
    {
        entities                   (Type.float4_buffer),
        entity_animation_elapsed   (Type.float2_buffer),
        entity_motion_states       (Type.short2_buffer),
        entity_animation_indices   (Type.int2_buffer),
        entity_hull_tables         (Type.int2_buffer),
        entity_bone_tables         (Type.int2_buffer),
        entity_masses              (Type.float_buffer),
        entity_root_hulls          (Type.int_buffer),
        entity_model_indices       (Type.int_buffer),
        entity_model_transforms    (Type.int_buffer),
        entity_types               (Type.int_buffer),
        entity_flags               (Type.int_buffer),
        target                     (Type.int_arg),
        new_entity                 (Type.float4_arg),
        new_entity_animation_time  (Type.float2_arg),
        new_entity_animation_state (Type.short2_arg),
        new_entity_animation_index (Type.int2_arg),
        new_entity_hull_table      (Type.int2_arg),
        new_entity_bone_table      (Type.int2_arg),
        new_entity_mass            (Type.float_arg),
        new_entity_root_hull       (Type.int_arg),
        new_entity_model_id        (Type.int_arg),
        new_entity_model_transform (Type.int_arg),
        new_entity_type            (Type.int_arg),
        new_entity_flags           (Type.int_arg),

        ;

        private final String cl_type;
        Args(String clType) { cl_type = clType; }
        public String cl_type() { return cl_type; }
    }

    public CreateEntity_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
