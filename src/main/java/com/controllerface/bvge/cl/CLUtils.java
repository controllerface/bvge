package com.controllerface.bvge.cl;

import com.controllerface.bvge.cl.kernels.Kernel;
import com.controllerface.bvge.cl.kernels.crud.KernelArg;
import org.joml.Matrix4f;
import org.lwjgl.PointerBuffer;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.system.MemoryUtil;

import java.io.IOException;
import java.nio.IntBuffer;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;

import static org.lwjgl.opencl.CL12.*;
import static org.lwjgl.opencl.CL12GL.clEnqueueAcquireGLObjects;
import static org.lwjgl.opencl.CL12GL.clEnqueueReleaseGLObjects;

public class CLUtils
{
    public static String read_src(String file)
    {
        try (var stream = CLUtils.class.getResourceAsStream("/cl/" + file))
        {
            byte [] bytes = Objects.requireNonNull(stream).readAllBytes();
            return new String(bytes, StandardCharsets.UTF_8);
        }
        catch (NullPointerException | IOException e)
        {
            throw new RuntimeException(e);
        }
    }

    /**
     * Generates source code for an Open CL kernel that can be called to create or update an object
     * using the specified arguments as input. Run any of the tests in the following package to see
     * example kernel output: {@link com.controllerface.bvge.cl.kernels.crud}
     *
     * @param kernel {@linkplain Kernel} enum class identifying the name of the kernel to generate
     * @param args_enum {@linkplain KernelArg} enum class defining the ordered set of argument to the kernel
     * @return a String containing the generated kernel source
     * @param <E> Type argument restricting the args_enum implementation
     */
    public static <E extends Enum<E> & KernelArg> String crud_create_k_src(Kernel kernel, Class<E> args_enum)
    {
        E[] arguments = args_enum.getEnumConstants();

        /*
        By convention, all Create* kernels define an argument named `target` which is used to determine the
        target position within the GPU buffer where the object being created will be stored. if missing,
        it is considered a critical failure.
         */
        int target_index = Arrays.stream(args_enum.getEnumConstants())
            .filter(arg -> arg.name().equals("target"))
            .map(Enum::ordinal)
            .findAny()
            .orElseThrow();

        var src = new StringBuilder("__kernel void ").append(kernel.name());

        var parameters = Arrays.stream(args_enum.getEnumConstants())
            .map(arg -> String.join(" ", arg.cl_type(), arg.name()))
            .collect(Collectors.joining(",\n\t","(",")\n"));

        src.append(parameters);
        src.append("{\n");
        for (int name_index = 0; name_index < target_index; name_index++)
        {
            int value_index = name_index + target_index + 1;
            var name        = arguments[name_index].name();
            var value       = arguments[value_index];

            src.append("\t")
                .append(name)
                .append("[target] = ")
                .append(value)
                .append(";\n");
        }
        src.append("}\n\n");
        return src.toString();
    }

    /**
     * Generates source code for an Open CL kernel that can be called to compact the buffers used
     * for logical objects. Run any of the tests in the following package to see example kernel
     * output: {@link com.controllerface.bvge.cl.kernels.compact}
     *
     * @param kernel {@linkplain Kernel} enum class identifying the name of the kernel to generate
     * @param args_enum {@linkplain KernelArg} enum class defining the ordered set of arguments to the kernel
     * @return a String containing the generated kernel source
     * @param <E> Type argument restricting the args_enum implementation
     */
    public static <E extends Enum<E> & KernelArg> String compact_k_src(Kernel kernel, Class<E> args_enum)
    {
        E[] arguments = args_enum.getEnumConstants();

        var src = new StringBuilder("__kernel void ").append(kernel.name());

        var parameters = Arrays.stream(args_enum.getEnumConstants())
            .map(arg -> String.join(" ", arg.cl_type(), arg.name()))
            .collect(Collectors.joining(",\n\t","(",")\n"));

        src.append(parameters);
        src.append("{\n");
        src.append("\tint current = get_global_id(0);\n");
        src.append("\tint shift = ").append(arguments[0].name()).append("[current];\n");

        for (int arg_index = 1; arg_index < arguments.length; arg_index++)
        {
            var arg  = arguments[arg_index];
            var type = arg.cl_type()
                .replace(KernelArg.Type.BUFFER_PREFIX, "")
                .replace(KernelArg.Type.BUFFER_SUFFIX, "")
                .trim();
            var _name = "_" + arg.name();
            src.append("\t")
                .append(type).append(" ")
                .append(_name).append(" = ")
                .append(arg.name()).append("[current]")
                .append(";\n");
        }
        src.append("\tbarrier(CLK_GLOBAL_MEM_FENCE);\n");
        src.append("\tif (shift > 0)\n");
        src.append("\t{\n");
        src.append("\t\tint new_index = current - shift;\n");

        for (int arg_index = 1; arg_index < arguments.length; arg_index++)
        {
            var arg   = arguments[arg_index];
            var _name = "_" + arg.name();
            src.append("\t\t")
                .append(arg.name()).append("[new_index]")
                .append(" = ")
                .append(_name)
                .append(";\n");
        }

        src.append("\t}\n");
        src.append("}\n\n");
        return src.toString();
    }

    public static long cl_p(long context_ptr, long device_id_ptr, String ... src)
    {
        long program = clCreateProgramWithSource(context_ptr, src, null);
        int result = clBuildProgram(program, device_id_ptr,  "-cl-denorms-are-zero -cl-mad-enable -cl-fast-relaxed-math", null, 0);
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


    public static int[] arg_int(int arg)
    {
        return new int[]{ arg };
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
