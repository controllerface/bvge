package com.controllerface.bvge.ecs.systems.sectors;

import com.controllerface.bvge.cl.GPUCoreMemory;
import com.controllerface.bvge.cl.buffers.BasicBufferGroup;
import com.controllerface.bvge.cl.buffers.BufferGroup;
import com.controllerface.bvge.cl.programs.GPUCrud;
import com.controllerface.bvge.cl.programs.GPUProgram;

public class BrokenObjectBuffer
{
    private final GPUProgram p_gpu_crud = new GPUCrud();
    private final BufferGroup buffer;

    public BrokenObjectBuffer(long ptr_queue, GPUCoreMemory core_memory)
    {
        buffer = new BasicBufferGroup(ptr_queue);
        // todo: add buffers as needed
        // todo: add egress_broken kernel
    }

    public void egress_broken()
    {
        // todo call kernel here
    }
}
