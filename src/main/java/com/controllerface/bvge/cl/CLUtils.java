package com.controllerface.bvge.cl;

import org.jocl.*;
import org.joml.Matrix4f;
import org.lwjgl.BufferUtils;
import org.lwjgl.PointerBuffer;
import org.lwjgl.opencl.CL12;
import org.lwjgl.opencl.CL12GL;
import org.lwjgl.system.MemoryStack;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.IntBuffer;
import java.nio.charset.StandardCharsets;
import java.util.Objects;

import static org.jocl.CL.*;

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

    public static cl_program cl_p(cl_context context, cl_device_id[] device_ids, String ... src)
    {
        var program = clCreateProgramWithSource(context, src.length, src, null, null);
        CL12.clBuildProgram(program.getNativePointer(), device_ids[0].getNativePointer(), "", null, 0);
        return program;
    }

    public static cl_kernel cl_k(cl_program program, String kernel_name)
    {
        return clCreateKernel(program, kernel_name, null);
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

    public static void k_call(cl_command_queue commandQueue, cl_kernel kernel, long[] global_work_size)
    {
        k_call(commandQueue, kernel, global_work_size, null, null);
    }

    public static void k_call(cl_command_queue commandQueue,
                              cl_kernel kernel,
                              long[] global_work_size,
                              long[] local_work_size)
    {

        k_call(commandQueue, kernel, global_work_size, local_work_size, null);
    }

    private static PointerBuffer int_to_buffer(MemoryStack mem_stack, long[] int_array)
    {
        return int_array == null
            ? null
            : mem_stack.callocPointer(1).put(0, int_array[0]);
    }

    public static void k_call(cl_command_queue commandQueue,
                              cl_kernel kernel,
                              long[] global_work_size,
                              long[] local_work_size,
                              long[] global_work_offset)
    {

        try (var mem_stack = MemoryStack.stackPush())
        {
            var global_offset_ptr = int_to_buffer(mem_stack, global_work_offset);
            var global_work_ptr = int_to_buffer(mem_stack, global_work_size);
            var local_work_ptr = int_to_buffer(mem_stack, local_work_size);

            int r = CL12.clEnqueueNDRangeKernel(commandQueue.getNativePointer(),
                kernel.getNativePointer(),
                1,
                global_offset_ptr,
                global_work_ptr,
                local_work_ptr,
                null,null);

            if (r != CL_SUCCESS)
            {
                System.out.println("WTF!");
            }
        }
    }

    private static PointerBuffer mem_to_buffer(MemoryStack mem_stack, cl_mem[] mem)
    {
        Objects.requireNonNull(mem);
        var pointer_buffer = mem_stack.callocPointer(mem.length);
        for (int i = 0; i < mem.length; i++)
        {
            pointer_buffer.put(i, mem[i].getNativePointer());
        }
        return pointer_buffer;
    }

    public static void gl_acquire(cl_command_queue commandQueue, cl_mem[] mem)
    {
        try (var mem_stack = MemoryStack.stackPush())
        {
            int r = CL12GL.clEnqueueAcquireGLObjects(commandQueue.getNativePointer(),
                mem_to_buffer(mem_stack, mem), null, null);
            if (r != 0)
            {
                System.out.println("error: " + r);
            }
        }
    }

    public static void gl_release(cl_command_queue commandQueue, cl_mem[] mem)
    {
        try (var mem_stack = MemoryStack.stackPush())
        {
            int r = CL12GL.clEnqueueReleaseGLObjects(commandQueue.getNativePointer(),
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
     * @param device The device
     * @param paramName The parameter name
     * @return The value
     */
    public static String getString(cl_device_id device, int paramName)
    {
        // Obtain the length of the string that will be queried
        long[] size = new long[1];
        clGetDeviceInfo(device, paramName, 0, null, size);

        // Create a buffer of the appropriate size and fill it with the info
        byte[] buffer = new byte[(int)size[0]];
        clGetDeviceInfo(device, paramName, buffer.length, Pointer.to(buffer), null);

        // Create a string p1 the buffer (excluding the trailing \0 byte)
        return new String(buffer, 0, buffer.length-1);
    }

    /**
     * Returns the value of the device info parameter with the given name
     *
     * @param device The device
     * @param paramName The parameter name
     * @return The value
     */
    public static long getSize(cl_device_id device, int paramName)
    {
        return getSizes(device, paramName, 1)[0];
    }

    /**
     * Returns the values of the device info parameter with the given name
     *
     * @param device The device
     * @param paramName The parameter name
     * @param numValues The number of values
     * @return The value
     */
    static long[] getSizes(cl_device_id device, int paramName, int numValues)
    {
        // The size of the returned data has p2 depend on
        // the size of a size_t, which is handled here
        ByteBuffer buffer = ByteBuffer.allocate(
            numValues * Sizeof.size_t).order(ByteOrder.nativeOrder());
        clGetDeviceInfo(device, paramName, (long) Sizeof.size_t * numValues,
            Pointer.to(buffer), null);
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
