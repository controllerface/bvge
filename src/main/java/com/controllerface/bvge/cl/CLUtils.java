package com.controllerface.bvge.cl;

import org.jocl.Sizeof;
import org.joml.Matrix4f;
import org.lwjgl.BufferUtils;
import org.lwjgl.PointerBuffer;
import org.lwjgl.opencl.CL12;
import org.lwjgl.opencl.CL12GL;
import org.lwjgl.system.MemoryStack;

import java.io.IOException;
import java.nio.ByteOrder;
import java.nio.IntBuffer;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Objects;

public class CLUtils
{
    public static String read_src(String file)
    {
        try (var stream = GPU.class.getResourceAsStream("/cl/" + file))
        {
            byte [] bytes = stream.readAllBytes();
            return new String(bytes, StandardCharsets.UTF_8);
        }
        catch (IOException e)
        {
            throw new RuntimeException(e);
        }
    }

    public static long cl_p(long context_ptr, long device_id_ptr, String ... src)
    {
        long program = CL12.clCreateProgramWithSource(context_ptr, src, null);
        CL12.clBuildProgram(program, device_id_ptr, "", null, 0);
        return program;
    }

    public static long cl_k(long program_ptr, String kernel_name)
    {
        return CL12.clCreateKernel(program_ptr, kernel_name, (IntBuffer) null);
    }

    public static double[] arg_double(double arg)
    {
        return new double[]{arg};
    }

    public static long[] arg_long(long arg)
    {
        return new long[]{arg};
    }

    public static int[] arg_int(int arg)
    {
        return new int[]{arg};
    }

    public static int[] arg_int2(int x, int y)
    {
        return new int[]{x, y};
    }

    public static int[] arg_int4(int x, int y, int z, int w)
    {
        return new int[]{x, y, z, w};
    }

    public static float[] arg_float(float arg)
    {
        return new float[]{arg};
    }

    public static float[] arg_float2(float x, float y)
    {
        return new float[]{x, y};
    }

    public static float[] arg_float4(float x, float y, float z, float w)
    {
        return new float[]{x, y, z, w};
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

            int r = CL12.clEnqueueNDRangeKernel(command_queue_ptr,
                kernel_ptr,
                1,
                global_offset_ptr,
                global_work_ptr,
                local_work_ptr,
                null,null);

            if (r != CL12.CL_SUCCESS)
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
            int r = CL12GL.clEnqueueAcquireGLObjects(command_queue_ptr,
                mem_to_buffer(mem_stack, mem), null, null);
            if (r != 0)
            {
                System.out.println("error: " + r);
            }
        }
    }

    public static void gl_release(long command_queue_ptr, List<Long> mem)
    {
        try (var mem_stack = MemoryStack.stackPush())
        {
            int r = CL12GL.clEnqueueReleaseGLObjects(command_queue_ptr,
                mem_to_buffer(mem_stack, mem), null, null);
            if (r != 0)
            {
                System.out.println("error: " + r);
            }
        }
    }

    /**
     * Returns the value of the device info parameter with the given name
     *
     * @param device_id_ptr The device
     * @param paramName The parameter name
     * @return The value
     */
    public static String getString(long device_id_ptr, int paramName)
    {
        // Obtain the length of the string that will be queried
        var lp = BufferUtils.createPointerBuffer(1);
        CL12.clGetDeviceInfo(device_id_ptr, paramName, (long[]) null, lp);
        long sz = lp.get();

        // Create a buffer of the appropriate size and fill it with the info
        var buffer = BufferUtils.createByteBuffer((int)sz);
        byte[] bytes = new byte[(int)sz];
        CL12.clGetDeviceInfo(device_id_ptr, paramName, buffer, null);
        buffer.get(bytes);

        // Create a string with the buffer (excluding the trailing \0 byte)
        return new String(bytes, 0, bytes.length - 1);
    }

    /**
     * Returns the value of the device info parameter with the given name
     *
     * @param device_ptr The device
     * @param paramName The parameter name
     * @return The value
     */
    public static long getSize(long device_ptr, int paramName)
    {
        return getSizes(device_ptr, paramName, 1)[0];
    }

    /**
     * Returns the values of the device info parameter with the given name
     *
     * @param device_ptr The device
     * @param paramName The parameter name
     * @param numValues The number of values
     * @return The value
     */
    static long[] getSizes(long device_ptr, int paramName, int numValues)
    {
        // The size of the returned data has p2 depend on
        // the size of a size_t, which is handled here

        var buffer = BufferUtils.createByteBuffer(numValues * Sizeof.size_t)
            .order(ByteOrder.nativeOrder());

        CL12.clGetDeviceInfo(device_ptr,
            paramName, buffer, null);

        long[] values = new long[numValues];
        if (Sizeof.size_t == 4)
        {
            for (int i=0; i<numValues; i++)
            {
                values[i] = buffer.getInt(i * Sizeof.size_t);
            }
        }
        else
        {
            for (int i=0; i<numValues; i++)
            {
                values[i] = buffer.getLong(i * Sizeof.size_t);
            }
        }
        return values;
    }
}
