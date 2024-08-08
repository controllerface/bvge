package com.controllerface.bvge.gpu.cl.kernels.crud;

import com.controllerface.bvge.gpu.GPU;
import com.controllerface.bvge.gpu.cl.buffers.CL_DataTypes;
import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;
import com.controllerface.bvge.gpu.cl.kernels.KernelType;
import com.controllerface.bvge.gpu.cl.kernels.KernelArg;

import static com.controllerface.bvge.memory.types.CoreBufferType.*;

public class CreateEntity_k extends GPUKernel
{
    public static final String kernel_source = GPU.CL.crud_create_k_src(KernelType.create_entity, Args.class);

    public enum Args implements KernelArg
    {
        entities                        (ENTITY.data_type().buffer_name()),
        entity_animation_time           (ENTITY_ANIM_TIME.data_type().buffer_name()),
        entity_previous_time            (ENTITY_PREV_TIME.data_type().buffer_name()),
        entity_motion_states            (ENTITY_MOTION_STATE.data_type().buffer_name()),
        entity_animation_layers         (ENTITY_ANIM_LAYER.data_type().buffer_name()),
        entity_previous_layers          (ENTITY_PREV_LAYER.data_type().buffer_name()),
        entity_hull_tables              (ENTITY_HULL_TABLE.data_type().buffer_name()),
        entity_bone_tables              (ENTITY_BONE_TABLE.data_type().buffer_name()),
        entity_masses                   (ENTITY_MASS.data_type().buffer_name()),
        entity_root_hulls               (ENTITY_ROOT_HULL.data_type().buffer_name()),
        entity_model_indices            (ENTITY_MODEL_ID.data_type().buffer_name()),
        entity_model_transforms         (ENTITY_TRANSFORM_ID.data_type().buffer_name()),
        entity_types                    (ENTITY_TYPE.data_type().buffer_name()),
        entity_flags                    (ENTITY_FLAG.data_type().buffer_name()),
        target                          (CL_DataTypes.cl_int.name()),
        new_entity                      (ENTITY.data_type().name()),
        new_entity_animation_time       (ENTITY_ANIM_TIME.data_type().name()),
        new_entity_previous_time        (ENTITY_PREV_TIME.data_type().name()),
        new_entity_animation_state      (ENTITY_MOTION_STATE.data_type().name()),
        new_entity_animation_layer      (ENTITY_ANIM_LAYER.data_type().name()),
        new_entity_previous_layer       (ENTITY_PREV_LAYER.data_type().name()),
        new_entity_hull_table           (ENTITY_HULL_TABLE.data_type().name()),
        new_entity_bone_table           (ENTITY_BONE_TABLE.data_type().name()),
        new_entity_mass                 (ENTITY_MASS.data_type().name()),
        new_entity_root_hull            (ENTITY_ROOT_HULL.data_type().name()),
        new_entity_model_id             (ENTITY_MODEL_ID.data_type().name()),
        new_entity_model_transform      (ENTITY_TRANSFORM_ID.data_type().name()),
        new_entity_type                 (ENTITY_TYPE.data_type().name()),
        new_entity_flags                (ENTITY_FLAG.data_type().name()),

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
