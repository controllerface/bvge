package com.controllerface.bvge.game.world.sectors;

import com.controllerface.bvge.cl.CLSize;
import com.controllerface.bvge.cl.GPGPU;
import com.controllerface.bvge.cl.GPUCoreMemory;
import com.controllerface.bvge.cl.buffers.BasicBufferGroup;
import com.controllerface.bvge.cl.buffers.BufferGroup;
import com.controllerface.bvge.cl.buffers.BufferType;
import com.controllerface.bvge.cl.kernels.EgressCollected_k;
import com.controllerface.bvge.cl.kernels.GPUKernel;
import com.controllerface.bvge.cl.kernels.Kernel;
import com.controllerface.bvge.cl.programs.GPUCrud;
import com.controllerface.bvge.cl.programs.GPUProgram;

import static com.controllerface.bvge.cl.CLSize.cl_int;
import static com.controllerface.bvge.cl.CLUtils.arg_long;

public class CollectedObjectBuffer
{
    private final GPUProgram p_gpu_crud = new GPUCrud();
    private final GPUKernel k_egress_collected;
    private final BufferGroup collected_group;
    private final long ptr_queue;
    private final long ptr_egress_size;

    public CollectedObjectBuffer(String name, long ptr_queue, GPUCoreMemory core_memory)
    {
        this.p_gpu_crud.init();
        this.ptr_queue  = ptr_queue;
        this.ptr_egress_size = GPGPU.cl_new_pinned_int();

        collected_group = new BasicBufferGroup(name, ptr_queue);
        collected_group.set_buffer(BufferType.COLLECTED_TYPE,  CLSize.cl_int, 100);

        long k_ptr_egress_collected = this.p_gpu_crud.kernel_ptr(Kernel.egress_collected);
        k_egress_collected = new EgressCollected_k(this.ptr_queue, k_ptr_egress_collected)
            .buf_arg(EgressCollected_k.Args.entity_flags, core_memory.get_buffer(BufferType.ENTITY_FLAG))
            .buf_arg(EgressCollected_k.Args.entity_types, core_memory.get_buffer(BufferType.ENTITY_TYPE))
            .buf_arg(EgressCollected_k.Args.types, collected_group.get_buffer(BufferType.COLLECTED_TYPE))
            .ptr_arg(EgressCollected_k.Args.counter, ptr_egress_size);
    }

    public void egress(int entity_count, int egress_count)
    {
        GPGPU.cl_zero_buffer(ptr_queue, ptr_egress_size, cl_int);
        collected_group.get_buffer(BufferType.COLLECTED_TYPE).ensure_capacity(egress_count);
        k_egress_collected.call(arg_long(entity_count));
    }

    public void unload(CollectedObjectBuffer.Raw raw, int count)
    {
        if (count > 0)
        {
            collected_group.get_buffer(BufferType.COLLECTED_TYPE).transfer_out_int(raw.types, cl_int, count);
        }
    }

    public void destroy()
    {
        p_gpu_crud.destroy();
        collected_group.destroy();
        GPGPU.cl_release_buffer(ptr_egress_size);
    }

    public static class Raw
    {
        public int[] types = new int[0];

        public void ensure_space(int count)
        {
            types      = ensure_int(types, count);
        }

        private int[] ensure_int(int[] input, int required_capacity)
        {
            return input.length >= required_capacity
                ? input
                : new int[required_capacity];
        }
    }
}
