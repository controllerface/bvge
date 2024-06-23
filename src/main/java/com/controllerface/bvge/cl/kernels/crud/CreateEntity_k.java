package com.controllerface.bvge.cl.kernels.crud;

import com.controllerface.bvge.cl.CLUtils;
import com.controllerface.bvge.cl.kernels.GPUKernel;
import com.controllerface.bvge.cl.kernels.Kernel;

public class CreateEntity_k extends GPUKernel
{
    public static final String kernel_source = CLUtils.crud_create_k_src(Kernel.create_entity, Args.class);

    public enum Args implements KernelArg
    {
        entities                   (Type.buffer_float4),
        entity_animation_elapsed   (Type.buffer_float2),
        entity_motion_states       (Type.buffer_short2),
        entity_animation_indices   (Type.buffer_int2),
        entity_hull_tables         (Type.buffer_int2),
        entity_bone_tables         (Type.buffer_int2),
        entity_masses              (Type.buffer_float),
        entity_root_hulls          (Type.buffer_int),
        entity_model_indices       (Type.buffer_int),
        entity_model_transforms    (Type.buffer_int),
        entity_types               (Type.buffer_int),
        entity_flags               (Type.buffer_int),
        target                     (Type.arg_int),
        new_entity                 (Type.arg_float4),
        new_entity_animation_time  (Type.arg_float2),
        new_entity_animation_state (Type.arg_short2),
        new_entity_animation_index (Type.arg_int2),
        new_entity_hull_table      (Type.arg_int2),
        new_entity_bone_table      (Type.arg_int2),
        new_entity_mass            (Type.arg_float),
        new_entity_root_hull       (Type.arg_int),
        new_entity_model_id        (Type.arg_int),
        new_entity_model_transform (Type.arg_int),
        new_entity_type            (Type.arg_int),
        new_entity_flags           (Type.arg_int),

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
