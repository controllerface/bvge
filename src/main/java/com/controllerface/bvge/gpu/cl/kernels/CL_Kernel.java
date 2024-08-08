package com.controllerface.bvge.gpu.cl.kernels;

import com.controllerface.bvge.gpu.GPUResource;
import org.lwjgl.system.MemoryStack;

import java.nio.ByteBuffer;

import static org.lwjgl.opencl.CL10.*;
import static org.lwjgl.opencl.CL20.clSetKernelArgSVMPointer;

public record CL_Kernel(long ptr) implements GPUResource
{
    public void ptr_arg(int arg_position, long pointer)
    {
        try (var mem_stack = MemoryStack.stackPush())
        {
            var pointerBuffer = mem_stack.callocPointer(1).put(0, pointer);
            int result = clSetKernelArg(ptr, arg_position, pointerBuffer);
            if (result != CL_SUCCESS) throw new RuntimeException("Error: clSetKernelArg(ptr): " + result);
        }
    }

    public void ptr_arg(int arg_position, ByteBuffer buffer)
    {
        int result = clSetKernelArgSVMPointer(ptr, arg_position, buffer);
        if (result != CL_SUCCESS) throw new RuntimeException("Error: clSetKernelArgSVMPointer(): " + result);
    }

    public void loc_arg(int arg_position, long size)
    {
        int result = clSetKernelArg(ptr, arg_position, size);
        if (result != CL_SUCCESS) throw new RuntimeException("Error: clSetKernelArg(__local): " + result);
    }

    public void set_arg(int arg_position, double[] value)
    {
        try (var mem_stack = MemoryStack.stackPush())
        {
            var doubleBuffer = mem_stack.doubles(value);
            int result = clSetKernelArg(ptr, arg_position, doubleBuffer);
            if (result != CL_SUCCESS) throw new RuntimeException("Error: clSetKernelArg(double[]): " + result);
        }
    }

    public void set_arg(int arg_position, double value)
    {
        try (var mem_stack = MemoryStack.stackPush())
        {
            var doubleBuffer = mem_stack.doubles(value);
            int result = clSetKernelArg(ptr, arg_position, doubleBuffer);
            if (result != CL_SUCCESS) throw new RuntimeException("Error: clSetKernelArg(double): " + result);
        }
    }

    public void set_arg(int arg_position, float[] value)
    {
        try (var mem_stack = MemoryStack.stackPush())
        {
            var floatBuffer = mem_stack.floats(value);
            int result = clSetKernelArg(ptr, arg_position, floatBuffer);
            if (result != CL_SUCCESS) throw new RuntimeException("Error: clSetKernelArg(float[]): " + result);
        }
    }

    public void set_arg(int arg_position, float value)
    {
        try (var mem_stack = MemoryStack.stackPush())
        {
            var floatBuffer = mem_stack.floats(value);
            int result = clSetKernelArg(ptr, arg_position, floatBuffer);
            if (result != CL_SUCCESS) throw new RuntimeException("Error: clSetKernelArg(float): " + result);
        }
    }

    public void set_arg(int arg_position, short[] value)
    {
        try (var mem_stack = MemoryStack.stackPush())
        {
            var short_buffer = mem_stack.shorts(value);
            int result = clSetKernelArg(ptr, arg_position, short_buffer);
            if (result != CL_SUCCESS) throw new RuntimeException("Error: clSetKernelArg(short[]): " + result);
        }
    }

    public void set_arg(int arg_position, short value)
    {
        try (var mem_stack = MemoryStack.stackPush())
        {
            var short_buffer = mem_stack.shorts(value);
            int result = clSetKernelArg(ptr, arg_position, short_buffer);
            if (result != CL_SUCCESS) throw new RuntimeException("Error: clSetKernelArg(short): " + result);
        }
    }

    public void set_arg(int arg_position, int[] value)
    {
        try (var mem_stack = MemoryStack.stackPush())
        {
            var int_buffer = mem_stack.ints(value);
            int result = clSetKernelArg(ptr, arg_position, int_buffer);
            if (result != CL_SUCCESS) throw new RuntimeException("Error: clSetKernelArg(int[]): " + result);
        }
    }

    public void set_arg(int arg_position, int value)
    {
        try (var mem_stack = MemoryStack.stackPush())
        {
            var int_buffer = mem_stack.ints(value);
            int result = clSetKernelArg(ptr, arg_position, int_buffer);
            if (result != CL_SUCCESS) throw new RuntimeException("Error: clSetKernelArg(int): " + result);
        }
    }

    @Override
    public void release()
    {
        int result = clReleaseKernel(ptr);
        if (result != CL_SUCCESS) throw new RuntimeException("Error: clReleaseKernel()" + result);
    }
}
