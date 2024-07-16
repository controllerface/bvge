package com.controllerface.bvge.cl.kernels.crud;

import com.controllerface.bvge.cl.CLUtils;
import com.controllerface.bvge.cl.kernels.GPUKernel;
import com.controllerface.bvge.cl.kernels.Kernel;
import com.controllerface.bvge.cl.kernels.KernelArg;

import static com.controllerface.bvge.cl.CLData.*;

public class CreateEntity_k extends GPUKernel
{
    public static final String kernel_source = CLUtils.crud_create_k_src(Kernel.create_entity, Args.class);

    public enum Args implements KernelArg
    {
        entities                        (cl_float4.buffer_name()),
        entity_animation_elapsed        (cl_float2.buffer_name()),
        entity_motion_states            (cl_short2.buffer_name()),
        entity_animation_layers         (cl_int2.buffer_name()),
        entity_animation_previous       (cl_int2.buffer_name()),
        entity_hull_tables              (cl_int2.buffer_name()),
        entity_bone_tables              (cl_int2.buffer_name()),
        entity_masses                   (cl_float.buffer_name()),
        entity_root_hulls               (cl_int.buffer_name()),
        entity_model_indices            (cl_int.buffer_name()),
        entity_model_transforms         (cl_int.buffer_name()),
        entity_types                    (cl_int.buffer_name()),
        entity_flags                    (cl_int.buffer_name()),
        target                          (cl_int.name()),
        new_entity                      (cl_float4.name()),
        new_entity_animation_time       (cl_float2.name()),
        new_entity_animation_state      (cl_short2.name()),
        new_entity_animation_layer      (cl_int2.name()),
        new_entity_animation_previous   (cl_int2.name()),
        new_entity_hull_table           (cl_int2.name()),
        new_entity_bone_table           (cl_int2.name()),
        new_entity_mass                 (cl_float.name()),
        new_entity_root_hull            (cl_int.name()),
        new_entity_model_id             (cl_int.name()),
        new_entity_model_transform      (cl_int.name()),
        new_entity_type                 (cl_int.name()),
        new_entity_flags                (cl_int.name()),

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
