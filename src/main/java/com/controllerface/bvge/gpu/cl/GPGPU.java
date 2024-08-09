package com.controllerface.bvge.gpu.cl;

import com.controllerface.bvge.ecs.ECS;
import com.controllerface.bvge.gpu.GPU;
import com.controllerface.bvge.memory.GPUCoreMemory;

public class GPGPU
{
    public static GPUCoreMemory core_memory;
    public static CL_ComputeController compute;

    public static void init(ECS ecs)
    {
        compute = GPU.CL.init_cl();
        core_memory = new GPUCoreMemory(compute, ecs);
    }

    public static void destroy()
    {
        core_memory.release();
        compute.release();
    }
}
