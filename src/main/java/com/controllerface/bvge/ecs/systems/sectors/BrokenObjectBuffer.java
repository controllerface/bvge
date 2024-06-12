package com.controllerface.bvge.ecs.systems.sectors;

import com.controllerface.bvge.cl.CLSize;
import com.controllerface.bvge.cl.GPGPU;
import com.controllerface.bvge.cl.GPUCoreMemory;
import com.controllerface.bvge.cl.buffers.BasicBufferGroup;
import com.controllerface.bvge.cl.buffers.BufferGroup;
import com.controllerface.bvge.cl.buffers.BufferType;
import com.controllerface.bvge.cl.kernels.EgressBroken_k;
import com.controllerface.bvge.cl.kernels.GPUKernel;
import com.controllerface.bvge.cl.kernels.Kernel;
import com.controllerface.bvge.cl.programs.GPUCrud;
import com.controllerface.bvge.cl.programs.GPUProgram;

import static com.controllerface.bvge.cl.CLUtils.arg_long;

public class BrokenObjectBuffer
{
    private final GPUProgram p_gpu_crud = new GPUCrud();

    private final GPUKernel k_egress_broken;

    private final BufferGroup broken_group;

    private final long ptr_queue;

    private final long ptr_egress_size;

    public BrokenObjectBuffer(long ptr_queue, GPUCoreMemory core_memory)
    {
        this.p_gpu_crud.init();
        this.ptr_queue  = ptr_queue;
        this.ptr_egress_size = GPGPU.cl_new_pinned_int();

        broken_group = new BasicBufferGroup(ptr_queue);
        broken_group.set_buffer(BufferType.BROKEN_POSITIONS, broken_group.new_buffer(CLSize.cl_float2, 100));
        broken_group.set_buffer(BufferType.BROKEN_MODEL_IDS, broken_group.new_buffer(CLSize.cl_int, 100));

        long k_ptr_egress_broken = this.p_gpu_crud.kernel_ptr(Kernel.egress_broken);
        k_egress_broken = new EgressBroken_k(this.ptr_queue, k_ptr_egress_broken)
            .buf_arg(EgressBroken_k.Args.entities, core_memory.buffer(BufferType.ENTITY))
            .buf_arg(EgressBroken_k.Args.entity_flags, core_memory.buffer(BufferType.ENTITY_FLAG))
            .buf_arg(EgressBroken_k.Args.entity_model_indices, core_memory.buffer(BufferType.ENTITY_MODEL_ID))
            .buf_arg(EgressBroken_k.Args.positions, broken_group.buffer(BufferType.BROKEN_POSITIONS))
            .buf_arg(EgressBroken_k.Args.model_ids, broken_group.buffer(BufferType.BROKEN_MODEL_IDS))
            .ptr_arg(EgressBroken_k.Args.counter, ptr_egress_size);
    }

    public void egress_broken(int entity_count, int egress_count)
    {
        broken_group.buffer(BufferType.BROKEN_POSITIONS).ensure_capacity(egress_count);
        broken_group.buffer(BufferType.BROKEN_MODEL_IDS).ensure_capacity(egress_count);
        k_egress_broken.call(arg_long(entity_count));
    }
}
