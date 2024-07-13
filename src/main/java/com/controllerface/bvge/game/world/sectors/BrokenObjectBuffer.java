package com.controllerface.bvge.game.world.sectors;

import com.controllerface.bvge.cl.CLSize;
import com.controllerface.bvge.cl.CLUtils;
import com.controllerface.bvge.cl.GPGPU;
import com.controllerface.bvge.cl.GPUCoreMemory;
import com.controllerface.bvge.cl.buffers.BufferGroup;
import com.controllerface.bvge.cl.buffers.BufferType;
import com.controllerface.bvge.cl.buffers.Destoryable;
import com.controllerface.bvge.cl.kernels.egress.EgressBroken_k;
import com.controllerface.bvge.cl.kernels.GPUKernel;
import com.controllerface.bvge.cl.kernels.Kernel;
import com.controllerface.bvge.cl.programs.GPUCrud;
import com.controllerface.bvge.cl.programs.GPUProgram;

import static com.controllerface.bvge.cl.CLSize.*;
import static com.controllerface.bvge.cl.buffers.CoreBufferType.*;

public class BrokenObjectBuffer implements Destoryable
{
    private final GPUProgram p_gpu_crud;
    private final GPUKernel k_egress_broken;
    private final BufferGroup<BrokenBuffer> broken_group;
    private final long ptr_queue;
    private final long ptr_egress_size;

    private enum BrokenBuffer implements BufferType
    {
        BROKEN_POSITIONS(cl_float2),
        BROKEN_ENTITY_TYPES(cl_int),
        BROKEN_MODEL_IDS(cl_int),

        ;

        private final int item_size;
        BrokenBuffer(int itemSize) { item_size = itemSize; }
        @Override public int size() { return item_size; }
    }

    public BrokenObjectBuffer(String name, long ptr_queue, GPUCoreMemory core_memory)
    {
        this.p_gpu_crud = new GPUCrud().init();
        this.ptr_queue  = ptr_queue;
        this.ptr_egress_size = GPGPU.cl_new_pinned_int();

        broken_group = new BufferGroup<>(BrokenBuffer.class, name, ptr_queue, true);
        broken_group.set_buffer(BrokenBuffer.BROKEN_POSITIONS,    100L);
        broken_group.set_buffer(BrokenBuffer.BROKEN_ENTITY_TYPES, 100L);
        broken_group.set_buffer(BrokenBuffer.BROKEN_MODEL_IDS,    100L);

        long k_ptr_egress_broken = this.p_gpu_crud.kernel_ptr(Kernel.egress_broken);
        k_egress_broken = new EgressBroken_k(this.ptr_queue, k_ptr_egress_broken)
            .buf_arg(EgressBroken_k.Args.entities, core_memory.get_buffer(ENTITY))
            .buf_arg(EgressBroken_k.Args.entity_flags, core_memory.get_buffer(ENTITY_FLAG))
            .buf_arg(EgressBroken_k.Args.entity_types, core_memory.get_buffer(ENTITY_TYPE))
            .buf_arg(EgressBroken_k.Args.entity_model_ids, core_memory.get_buffer(ENTITY_MODEL_ID))
            .buf_arg(EgressBroken_k.Args.positions, broken_group.get_buffer(BrokenBuffer.BROKEN_POSITIONS))
            .buf_arg(EgressBroken_k.Args.types, broken_group.get_buffer(BrokenBuffer.BROKEN_ENTITY_TYPES))
            .buf_arg(EgressBroken_k.Args.model_ids, broken_group.get_buffer(BrokenBuffer.BROKEN_MODEL_IDS))
            .ptr_arg(EgressBroken_k.Args.counter, ptr_egress_size);
    }

    public void egress(int entity_count, int egress_count)
    {
        GPGPU.cl_zero_buffer(ptr_queue, ptr_egress_size, cl_int);
        int entity_size  = GPGPU.calculate_preferred_global_size(entity_count);
        broken_group.get_buffer(BrokenBuffer.BROKEN_POSITIONS).ensure_capacity(egress_count);
        broken_group.get_buffer(BrokenBuffer.BROKEN_ENTITY_TYPES).ensure_capacity(egress_count);
        broken_group.get_buffer(BrokenBuffer.BROKEN_MODEL_IDS).ensure_capacity(egress_count);
        k_egress_broken
            .set_arg(EgressBroken_k.Args.max_entity, entity_count)
            .call(CLUtils.arg_long(entity_size), GPGPU.preferred_work_size);
    }

    public void unload(BrokenObjectBuffer.Raw raw, int count)
    {
        if (count > 0)
        {
            int count_vec2 = count * 2;
            broken_group.get_buffer(BrokenBuffer.BROKEN_POSITIONS).transfer_out_float(raw.positions, cl_float, count_vec2);
            broken_group.get_buffer(BrokenBuffer.BROKEN_ENTITY_TYPES).transfer_out_int(raw.entity_types, cl_int, count);
            broken_group.get_buffer(BrokenBuffer.BROKEN_MODEL_IDS).transfer_out_int(raw.model_ids, cl_int, count);
        }
    }

    public static class Raw
    {
        public float[] positions = new float[0];
        public int[] entity_types = new int[0];
        public int[] model_ids = new int[0];

        public void ensure_space(int count)
        {
            int entity_vec2 = count * 2;
            positions  = ensure_float(positions, entity_vec2);
            entity_types = ensure_int(entity_types, count);
            model_ids  = ensure_int(model_ids, count);
        }

        private float[] ensure_float(float[] input, int required_capacity)
        {
            return input.length >= required_capacity
                ? input
                : new float[required_capacity];
        }

        private int[] ensure_int(int[] input, int required_capacity)
        {
            return input.length >= required_capacity
                ? input
                : new int[required_capacity];
        }
    }

    public void destroy()
    {
        p_gpu_crud.destroy();
        broken_group.destroy();
        GPGPU.cl_release_buffer(ptr_egress_size);
    }
}
