package com.controllerface.bvge.gpu.cl;

import com.controllerface.bvge.ecs.ECS;
import com.controllerface.bvge.editor.Editor;
import com.controllerface.bvge.gpu.GPU;
import com.controllerface.bvge.gpu.cl.buffers.CL_DataTypes;
import com.controllerface.bvge.memory.GPUCoreMemory;
import org.lwjgl.BufferUtils;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import static org.lwjgl.opencl.CL12.*;
import static org.lwjgl.opencl.CL20.*;

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



    //#region Utility Methods



    public static ByteBuffer cl_new_svm_int()
    {
        return clSVMAlloc(compute.context.ptr(), CL_MEM_READ_WRITE, CL_DataTypes.cl_int.size(), 0);
    }

    public static int cl_read_svm_int(long queue_ptr, ByteBuffer svm_buffer)
    {
        long s = Editor.ACTIVE ? System.nanoTime() : 0;
        int result = clEnqueueSVMMap(queue_ptr, true, CL_MAP_READ, svm_buffer, null, null);
        if (result != CL_SUCCESS)
        {
            System.out.println("Error on svm buffer map: " + result);
            throw new RuntimeException("Error on svm buffer map: " + result);
        }
        int v = svm_buffer.getInt(0);
        result = clEnqueueSVMUnmap(queue_ptr, svm_buffer, null, null);
        if (result != CL_SUCCESS)
        {
            System.out.println("Error on svm buffer unmap: " + result);
            throw new RuntimeException("Error on svm buffer unmap: " + result);
        }
        if (Editor.ACTIVE)
        {
            long e = System.nanoTime() - s;
            Editor.queue_event("cl_read_svm_int", String.valueOf(e));
        }
        return v;
    }

    //#endregion

    //#region Misc. Public API


    public static void init(ECS ecs)
    {
        compute = GPU.CL.init_cl();
        core_memory = new GPUCoreMemory(ecs);
    }

    public static void destroy()
    {
        core_memory.release();
        compute.release();
    }

    //#endregion
}
