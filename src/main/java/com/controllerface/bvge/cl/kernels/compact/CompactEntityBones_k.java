package com.controllerface.bvge.cl.kernels.compact;

import com.controllerface.bvge.cl.CLUtils;
import com.controllerface.bvge.cl.buffers.CoreBufferType;
import com.controllerface.bvge.cl.kernels.GPUKernel;
import com.controllerface.bvge.cl.kernels.Kernel;
import com.controllerface.bvge.cl.kernels.crud.KernelArg;

import static com.controllerface.bvge.cl.CLData.cl_float16;
import static com.controllerface.bvge.cl.CLData.cl_int;
import static com.controllerface.bvge.cl.buffers.CoreBufferType.*;

public class CompactEntityBones_k extends GPUKernel
{
    public static final String kernel_source = CLUtils.compact_k_src(Kernel.compact_entity_bones, Args.class);

    public enum Args implements KernelArg
    {
        entity_bone_shift           (cl_int.buffer_name()),
        entity_bones                (ENTITY_BONE.data_type().buffer_name()),
        entity_bone_reference_ids   (ENTITY_BONE_REFERENCE_ID.data_type().buffer_name()),
        entity_bone_parent_ids      (ENTITY_BONE_PARENT_ID.data_type().buffer_name()),

        ;

        private final String cl_type;
        Args(String clType) { cl_type = clType; }
        public String cl_type() { return cl_type; }
    }

    public CompactEntityBones_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
