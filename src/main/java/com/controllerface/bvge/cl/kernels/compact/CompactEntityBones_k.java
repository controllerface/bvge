package com.controllerface.bvge.cl.kernels.compact;

import com.controllerface.bvge.cl.CLUtils;
import com.controllerface.bvge.cl.kernels.GPUKernel;
import com.controllerface.bvge.cl.kernels.Kernel;
import com.controllerface.bvge.cl.kernels.crud.KernelArg;

public class CompactEntityBones_k extends GPUKernel
{
    public static final String kernel_source = CLUtils.compact_k_src(Kernel.compact_entity_bones, Args.class);

    public enum Args implements KernelArg
    {
        entity_bone_shift(Type.buffer_int),
        entity_bones(Type.buffer_float16),
        entity_bone_reference_ids(Type.buffer_int),
        entity_bone_parent_ids(Type.buffer_int),

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
