package com.controllerface.bvge.memory.sectors;

import com.controllerface.bvge.gpu.GPU;
import com.controllerface.bvge.gpu.GPUResource;
import com.controllerface.bvge.gpu.cl.buffers.BufferGroup;
import com.controllerface.bvge.gpu.cl.buffers.CL_Buffer;
import com.controllerface.bvge.gpu.cl.contexts.CL_CommandQueue;
import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;
import com.controllerface.bvge.gpu.cl.kernels.egress.EgressCollected_k;
import com.controllerface.bvge.gpu.cl.programs.GPUProgram;
import com.controllerface.bvge.gpu.cl.programs.crud.GPUCrud;
import com.controllerface.bvge.memory.GPUCoreMemory;
import com.controllerface.bvge.memory.types.CollectedBufferType;

import static com.controllerface.bvge.gpu.GPU.CL.arg_long;
import static com.controllerface.bvge.gpu.cl.buffers.CL_DataTypes.cl_int;

public class CollectedObjectBuffer implements GPUResource
{
    private final GPUProgram p_gpu_crud = new GPUCrud();
    private final GPUKernel k_egress_collected;
    private final BufferGroup<CollectedBufferType> collected_group;
    private final CL_CommandQueue cmd_queue;
    private final CL_Buffer ptr_egress_size;

    public CollectedObjectBuffer(CL_CommandQueue cmd_queue, GPUCoreMemory core_memory, String name)
    {
        this.p_gpu_crud.init();
        this.cmd_queue = cmd_queue;
        this.ptr_egress_size = GPU.CL.new_pinned_int(GPU.compute.context);

        collected_group = new BufferGroup<>(cmd_queue, CollectedBufferType.class, name, true);
        collected_group.init_buffer(CollectedBufferType.TYPES, 100L);

        k_egress_collected = new EgressCollected_k(cmd_queue, p_gpu_crud).init(core_memory, collected_group, ptr_egress_size);
    }

    public void egress(int entity_count, int egress_count)
    {
        GPU.CL.zero_buffer(cmd_queue, ptr_egress_size, cl_int.size());
        collected_group.buffer(CollectedBufferType.TYPES).ensure_capacity(egress_count);
        int entity_size  = GPU.compute.calculate_preferred_global_size(entity_count);
        k_egress_collected
            .set_arg(EgressCollected_k.Args.max_entity, entity_count)
            .call(arg_long(entity_size), GPU.compute.preferred_work_size);
    }

    public void unload(CollectedObjectBuffer.Raw raw, int count)
    {
        if (count > 0)
        {
            collected_group.buffer(CollectedBufferType.TYPES).transfer_out_int(raw.types, cl_int.size(), count);
        }
    }

    public void release()
    {
        p_gpu_crud.release();
        collected_group.release();
        ptr_egress_size.release();
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
