package com.controllerface.bvge.gpu.cl;

import com.controllerface.bvge.ecs.ECS;
import com.controllerface.bvge.gpu.GPU;
import com.controllerface.bvge.memory.GPUCoreMemory;
import org.lwjgl.BufferUtils;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/**
 * Core class used for executing General Purpose GPU (GPGPU) functions
 */
public class GPGPU
{
    //#region Constants

    /**
     * A convenience object, used when clearing out buffers to fill them with zeroes
     */
    private static final ByteBuffer ZERO_PATTERN_BUFFER = BufferUtils.createByteBuffer(4).order(ByteOrder.nativeOrder());
    private static final ByteBuffer NEGATIVE_ONE_PATTERN_BUFFER = BufferUtils.createByteBuffer(4).order(ByteOrder.nativeOrder());

    static
    {
        ZERO_PATTERN_BUFFER.putInt(0,0);
        NEGATIVE_ONE_PATTERN_BUFFER.putInt(0, -1);
    }

    //#endregion

    //#region Workgroup Variables

    public static GPUCoreMemory core_memory;
    public static CL_ComputeController compute;

    //#endregion


    //#region Misc. Public API


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

    //#endregion
}
