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


    public static long cl_p(long context_ptr, long device_id_ptr, String ... src)
    {
        long program = clCreateProgramWithSource(context_ptr, src, null);
        int result = clBuildProgram(program, device_id_ptr,  "-cl-finite-math-only -cl-denorms-are-zero -cl-mad-enable -cl-fast-relaxed-math", null, 0);
        if (result != CL_SUCCESS)
        {
            System.err.println("Error on program creation : " + result);
            log_build_error(program, device_id_ptr);
            throw new RuntimeException("Error on program creation : " + result);
        }
        return program;
    }

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

    private static PointerBuffer mem_to_buffer(MemoryStack mem_stack, List<Long> mem)
    {
        Objects.requireNonNull(mem);
        var pointer_buffer = mem_stack.callocPointer(mem.size());
        for (int i = 0; i < mem.size(); i++)
        {
            pointer_buffer.put(i, mem.get(i));
        }
        return pointer_buffer;
    }

    public static void gl_acquire(long command_queue_ptr, List<Long> mem)
    {
        try (var mem_stack = MemoryStack.stackPush())
        {
            var buffer = mem_to_buffer(mem_stack, mem);
            int result = clEnqueueAcquireGLObjects(command_queue_ptr, buffer, null, null);
            if (result != CL_SUCCESS)
            {
                System.out.println("Error on GL memory acquire : " + result);
                throw new RuntimeException("Error on GL memory acquire : " + result);
            }
        }
    }

    public static void gl_release(long command_queue_ptr, List<Long> mem)
    {
        try (var mem_stack = MemoryStack.stackPush())
        {
            var buffer = mem_to_buffer(mem_stack, mem);
            int result = clEnqueueReleaseGLObjects(command_queue_ptr, buffer, null, null);
            if (result != CL_SUCCESS)
            {
                System.out.println("Error on GL memory release : " + result);
                throw new RuntimeException("Error on GL memory release : " + result);
            }
        }
    }

    /**
     * Returns the String value of the device info parameter with the given name
     *
     * @param device_ptr The device
     * @param param_code The parameter name
     * @return The value
     */
    public static String get_device_string(long device_ptr, int param_code)
    {
        var size_buffer = MemoryUtil.memAllocPointer(1);
        clGetDeviceInfo(device_ptr, param_code, (long[]) null, size_buffer);
        long size = size_buffer.get();
        var value_buffer = MemoryUtil.memAlloc((int)size);
        byte[] bytes = new byte[(int)size];
        clGetDeviceInfo(device_ptr, param_code, value_buffer, null);
        value_buffer.get(bytes);
        MemoryUtil.memFree(size_buffer);
        MemoryUtil.memFree(value_buffer);
        return new String(bytes, 0, bytes.length - 1);
    }

    /**
     * Returns the long value of the device info parameter with the given name
     *
     * @param device_ptr The device
     * @param param_code The parameter name
     * @return The value
     */
    public static long get_device_long(long device_ptr, int param_code)
    {
        var size_buffer = MemoryUtil.memAllocPointer(1);
        clGetDeviceInfo(device_ptr, param_code, (long[]) null, size_buffer);
        long size = size_buffer.get();
        var value_buffer = MemoryUtil.memAlloc((int)size);
        int r = clGetDeviceInfo(device_ptr, param_code, value_buffer, null);
        if (r != CL_SUCCESS)
        {
            System.out.println("debug error:" + r);
            return -1;
        }
        var result = size == 4
            ? value_buffer.getInt(0)
            : value_buffer.getLong(0);
        MemoryUtil.memFree(size_buffer);
        MemoryUtil.memFree(value_buffer);
        return result;
    }


    public static boolean get_device_boolean(long device_ptr, int param_code)
    {
        var size_buffer = MemoryUtil.memAllocPointer(1);
        clGetDeviceInfo(device_ptr, param_code, (long[]) null, size_buffer);
        long size = size_buffer.get();
        var value_buffer = MemoryUtil.memAlloc((int)size);
        clGetDeviceInfo(device_ptr, param_code, value_buffer, null);
        var result = value_buffer.get();

        MemoryUtil.memFree(size_buffer);
        MemoryUtil.memFree(value_buffer);
        return result == 1;
    }

    private static void log_build_error(long program, long device_id_ptr)
    {
        try (var mem_stack = MemoryStack.stackPush())
        {
            var size_buffer = mem_stack.callocPointer(1);
            clGetProgramBuildInfo(program, device_id_ptr, CL_PROGRAM_BUILD_LOG, (int[]) null, size_buffer);
            int size = (int)size_buffer.get(0);
            var message_buffer = mem_stack.calloc(size);
            clGetProgramBuildInfo(program, device_id_ptr, CL_PROGRAM_BUILD_LOG, message_buffer, null);
            byte[] bytes = new byte[size];
            for (int i =0; i < size; i ++)
            {
                bytes[i] = message_buffer.get();
            }
            var message = new String(bytes);
            System.err.println(message);
        }
    }
}
