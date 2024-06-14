package com.controllerface.bvge.ecs.systems.sectors;

import com.controllerface.bvge.cl.CLSize;
import com.controllerface.bvge.cl.GPGPU;
import com.controllerface.bvge.cl.GPUCoreMemory;
import com.controllerface.bvge.cl.buffers.BasicBufferGroup;
import com.controllerface.bvge.cl.buffers.BufferGroup;
import com.controllerface.bvge.cl.buffers.BufferType;
import com.controllerface.bvge.cl.kernels.EgressBroken_k;
import com.controllerface.bvge.cl.kernels.EgressCollected_k;
import com.controllerface.bvge.cl.kernels.GPUKernel;
import com.controllerface.bvge.cl.kernels.Kernel;
import com.controllerface.bvge.cl.programs.GPUCrud;
import com.controllerface.bvge.cl.programs.GPUProgram;

import static com.controllerface.bvge.cl.CLSize.cl_float;
import static com.controllerface.bvge.cl.CLSize.cl_int;
import static com.controllerface.bvge.cl.CLUtils.arg_long;

public class CollectedObjectBuffer
{
    private final GPUProgram p_gpu_crud = new GPUCrud();

    private final GPUKernel k_egress_collected;

    private final BufferGroup collected_group;

    private final long ptr_queue;

    private final long ptr_egress_size;

    public CollectedObjectBuffer(long ptr_queue, GPUCoreMemory core_memory)
    {
        this.p_gpu_crud.init();
        this.ptr_queue  = ptr_queue;
        this.ptr_egress_size = GPGPU.cl_new_pinned_int();

        collected_group = new BasicBufferGroup(ptr_queue);
        collected_group.set_buffer(BufferType.COLLECTED_UV_OFFSETS, collected_group.new_buffer(CLSize.cl_int, 100));
        collected_group.set_buffer(BufferType.COLLECTED_FLAG,  collected_group.new_buffer(CLSize.cl_int, 100));

        long k_ptr_egress_collected = this.p_gpu_crud.kernel_ptr(Kernel.egress_collected);
        k_egress_collected = new EgressCollected_k(this.ptr_queue, k_ptr_egress_collected)
            .buf_arg(EgressCollected_k.Args.entity_flags, core_memory.buffer(BufferType.ENTITY_FLAG))
            .buf_arg(EgressCollected_k.Args.entity_hull_tables, core_memory.buffer(BufferType.ENTITY_HULL_TABLE))
            .buf_arg(EgressCollected_k.Args.hull_flags, core_memory.buffer(BufferType.HULL_FLAG))
            .buf_arg(EgressCollected_k.Args.hull_uv_offsets, core_memory.buffer(BufferType.HULL_UV_OFFSET))
            .buf_arg(EgressCollected_k.Args.uv_offsets, collected_group.buffer(BufferType.COLLECTED_UV_OFFSETS))
            .buf_arg(EgressCollected_k.Args.flags, collected_group.buffer(BufferType.COLLECTED_FLAG))
            .ptr_arg(EgressCollected_k.Args.counter, ptr_egress_size);
    }

    public void egress(int entity_count, int egress_count)
    {
        GPGPU.cl_zero_buffer(ptr_queue, ptr_egress_size, cl_int);
        collected_group.buffer(BufferType.COLLECTED_UV_OFFSETS).ensure_capacity(egress_count);
        collected_group.buffer(BufferType.COLLECTED_FLAG).ensure_capacity(egress_count);
        k_egress_collected.call(arg_long(entity_count));
    }

    public void unload(CollectedObjectBuffer.Raw raw, int count)
    {
        if (count > 0)
        {
            collected_group.buffer(BufferType.COLLECTED_UV_OFFSETS).transfer_out_int(raw.uv_offsets, cl_float, count);
            collected_group.buffer(BufferType.COLLECTED_FLAG).transfer_out_int(raw.flags, cl_int, count);
        }
    }

    public void destroy()
    {
        GPGPU.cl_release_buffer(ptr_egress_size);
        p_gpu_crud.destroy();
    }

    public static class Raw
    {
        public int[] uv_offsets = new int[0];
        public int[] flags = new int[0];

        public void ensure_space(int count)
        {
            uv_offsets = ensure_int(uv_offsets, count);
            flags      = ensure_int(flags, count);
        }

        private int[] ensure_int(int[] input, int required_capacity)
        {
            return input.length >= required_capacity
                ? input
                : new int[required_capacity];
        }
    }
}
