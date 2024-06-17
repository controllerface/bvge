package com.controllerface.bvge.cl;

import com.controllerface.bvge.cl.kernels.CreatePoint_k;
import org.joml.Matrix4f;
import org.lwjgl.BufferUtils;
import org.lwjgl.PointerBuffer;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.system.MemoryUtil;

import java.io.IOException;
import java.nio.IntBuffer;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.stream.Collectors;

import static org.lwjgl.opencl.CL12.*;
import static org.lwjgl.opencl.CL12GL.clEnqueueAcquireGLObjects;
import static org.lwjgl.opencl.CL12GL.clEnqueueReleaseGLObjects;

public class CLUtils
{
    public static String read_src(String file)
    {
        try (var stream = GPGPU.class.getResourceAsStream("/cl/" + file))
        {
            byte [] bytes = Objects.requireNonNull(stream).readAllBytes();
            return new String(bytes, StandardCharsets.UTF_8);
        }
        catch (NullPointerException | IOException e)
        {
            throw new RuntimeException(e);
        }
    }

    public static <E extends Enum<E>> String generate_crud_kernel_header(int t_offset, String kernel_name, Class<E> enum_class, Map<Enum<?>, String> type_map)
    {
        var buffer = new StringBuilder();
        buffer.append("__kernel void ").append(kernel_name);
        var kernel_args = Arrays.stream(enum_class.getEnumConstants())
            .map(arg -> type_map.get(arg) + " " + arg.name())
            .collect(Collectors.joining(",\n\t","(",")\n"));
        buffer.append(kernel_args);
        buffer.append("{\n");
        E[] args = enum_class.getEnumConstants();
        for (int i = 0; i < t_offset; i++)
        {
            buffer.append("\t")
                .append(args[i].name())
                .append("[target] = ")
                .append(args[i+t_offset+1]).append(";\n");
        }
        buffer.append("}\n\n");
        return buffer.toString();
    }

    public static long cl_p(long context_ptr, long device_id_ptr, String ... src)
    {
        long program = clCreateProgramWithSource(context_ptr, src, null);
        int result = clBuildProgram(program, device_id_ptr,  "-cl-denorms-are-zero -cl-mad-enable", null, 0);
        if (result != CL_SUCCESS)
        {
            System.err.println("program build error: " + result);
            log_build_error(program, device_id_ptr);
        }
        return program;
    }

    public static long cl_k(long program_ptr, String kernel_name)
    {
        return clCreateKernel(program_ptr, kernel_name, (IntBuffer) null);
    }

    public static long[] arg_long(long arg)
    {
        return new long[]{ arg };
    }

    public static int[] arg_int4(int x, int y, int z, int w)
    {
        return new int[]{ x, y, z, w };
    }

    public static int[] arg_int2(int x, int y)
    {
        return new int[]{ x, y };
    }


    public static short[] arg_short2(short x, short y)
    {
        return new short[]{ x, y };
    }

    public static float[] arg_float2(float x, float y)
    {
        return new float[]{ x, y };
    }

    public static float[] arg_float4(float x, float y, float z, float w)
    {
        return new float[]{ x, y, z, w };
    }

    public static float[] arg_float16(float s0, float s1, float s2, float s3,
                                      float s4, float s5, float s6, float s7,
                                      float s8, float s9, float sA, float sB,
                                      float sC, float sD, float sE, float sF)
    {
        return new float[]
            {
                s0, s1, s2, s3,
                s4, s5, s6, s7,
                s8, s9, sA, sB,
                sC, sD, sE, sF
            };
    }

    public static float[] arg_float16_matrix(Matrix4f matrix)
    {
        return arg_float16(
            matrix.m00(), matrix.m01(), matrix.m02(), matrix.m03(),
            matrix.m10(), matrix.m11(), matrix.m12(), matrix.m13(),
            matrix.m20(), matrix.m21(), matrix.m22(), matrix.m23(),
            matrix.m30(), matrix.m31(), matrix.m32(), matrix.m33());
    }

    public static void k_call(long command_queue_ptr, long kernel_ptr, long[] global_work_size)
    {
        k_call(command_queue_ptr, kernel_ptr, global_work_size, null, null);
    }

    public static void k_call(long command_queue_ptr,
                              long kernel_ptr,
                              long[] global_work_size,
                              long[] local_work_size)
    {

        k_call(command_queue_ptr, kernel_ptr, global_work_size, local_work_size, null);
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

            int r = clEnqueueNDRangeKernel(command_queue_ptr,
                kernel_ptr,
                1,
                global_offset_ptr,
                global_work_ptr,
                local_work_ptr,
                null,
                null);

            if (r != CL_SUCCESS)
            {
                System.out.println("WTF!");
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
            int r = clEnqueueAcquireGLObjects(command_queue_ptr, buffer, null, null);
            if (r != CL_SUCCESS)
            {
                System.out.println("error: " + r);
            }
        }
    }

    public static void gl_release(long command_queue_ptr, List<Long> mem)
    {
        try (var mem_stack = MemoryStack.stackPush())
        {
            var buffer = mem_to_buffer(mem_stack, mem);
            int r = clEnqueueReleaseGLObjects(command_queue_ptr, buffer, null, null);
            if (r != CL_SUCCESS)
            {
                System.out.println("error: " + r);
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
        clGetDeviceInfo(device_ptr, param_code, value_buffer, null);
        var result = size == 4
            ? value_buffer.getInt(0)
            : value_buffer.getLong(0);
        MemoryUtil.memFree(size_buffer);
        MemoryUtil.memFree(value_buffer);
        return result;
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
