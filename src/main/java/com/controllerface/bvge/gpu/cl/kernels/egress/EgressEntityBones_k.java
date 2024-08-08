package com.controllerface.bvge.gpu.cl.kernels.egress;

import com.controllerface.bvge.gpu.cl.contexts.CL_CommandQueue;
import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;

public class EgressEntityBones_k extends GPUKernel
{
    public enum Args
    {
        entity_bones_in,
        entity_bone_reference_ids_in,
        entity_bone_parent_ids_in,
        entity_bones_out,
        entity_bone_reference_ids_out,
        entity_bone_parent_ids_out,
        new_entity_bones,
        max_entity_bone,
    }

    public EgressEntityBones_k(CL_CommandQueue command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
