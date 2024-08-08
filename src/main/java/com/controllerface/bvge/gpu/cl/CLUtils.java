package com.controllerface.bvge.gpu.cl;

import org.lwjgl.PointerBuffer;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.system.MemoryUtil;

import java.util.List;
import java.util.Objects;

import static org.lwjgl.opencl.CL12.*;
import static org.lwjgl.opencl.CL12GL.clEnqueueAcquireGLObjects;
import static org.lwjgl.opencl.CL12GL.clEnqueueReleaseGLObjects;

public class CLUtils
{
    public static final String BUFFER_PREFIX = "__global";
    public static final String BUFFER_SUFFIX = "*";


    public static long cl_k(long program_ptr, String kernel_name)
    {
        try (var stack = MemoryStack.stackPush())
        {
            var status = stack.mallocInt(1);
            long ptr = clCreateKernel(program_ptr, kernel_name, status);
            int result = status.get(0);
            if (result != CL_SUCCESS)
            {
                System.out.println("Error on kernel creation : " + result);
                throw new RuntimeException("Error on kernel creation : " + result);
            }
            return ptr;
        }
    }

    private static PointerBuffer int_to_buffer(MemoryStack mem_stack, long[] int_array)
    {
        return int_array == null
            ? null
            : mem_stack.callocPointer(1).put(0, int_array[0]);
    }

    public static void k_call(long command_queue_ptr,
                              long kernel_ptr,
                              long[] global_work_size,
                              long[] local_work_size,
                              long[] global_work_offset)
    {

        try (var mem_stack = MemoryStack.stackPush())
        {
            var global_offset_ptr = int_to_buffer(mem_stack, global_work_offset);
            var global_work_ptr = int_to_buffer(mem_stack, global_work_size);
            var local_work_ptr = int_to_buffer(mem_stack, local_work_size);

            int result = clEnqueueNDRangeKernel(command_queue_ptr,
                kernel_ptr,
                1,
                global_offset_ptr,
                global_work_ptr,
                local_work_ptr,
                null,
                null);

            if (result != CL_SUCCESS)
            {
                System.out.println("Error on kernel call : " + result);
                throw new RuntimeException("Error on kernel call : " + result);
            }
        }
    }

}
