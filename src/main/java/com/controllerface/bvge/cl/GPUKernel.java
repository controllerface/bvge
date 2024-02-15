package com.controllerface.bvge.cl;

import org.jocl.Sizeof;
import org.lwjgl.opencl.CL12;
import org.lwjgl.system.MemoryStack;

import java.util.ArrayList;
import java.util.List;

import static com.controllerface.bvge.cl.CLUtils.k_call;

/**
 * A class for defining and organizing GPU kernel functions. This class is used to wrap an Open Cl kernel
 * making it easier to reason about in the host Java code. Helpers are provided for setting arguments
 * and calling the kernel itself. Kernels are typically provided by {@link GPUProgram} objects, which are
 * compiled first before their constituent kernels are linked and loaded via implementations of this class.
 */
public abstract class GPUKernel<E extends Enum<E> & GPUKernel.GPUKernelArg>
{
    final long command_queue_ptr;
    final long kernel_ptr;
    final List<Long> shared_memory_ptrs = new ArrayList<>();
    final long[] arg_sizes;

    /**
     * Kernel subclasses must define an enum that implements this simple interface in order to rigidly
     * define kernel argument positions and data sizes.
     */
    public interface GPUKernelArg
    {
        long size();
    }

    public GPUKernel(long command_queue_ptr, long kernel_ptr, E[] args)
    {
        this.command_queue_ptr = command_queue_ptr;
        this.kernel_ptr = kernel_ptr;
        this.arg_sizes = new long[args.length];
        for (var arg : args)
        {
            def_arg(arg.ordinal(), arg.size());
        }
    }

    /**
     * Allows calling code to designate a memory buffer as being shared between Open CL and Open Gl contexts.
     * This allows the kernel call to automatically acquire and release the shared objects, before and after
     * the kernel call, respectively. The shared memory list si also cleared after each call, so memory
     * objects must be shared individually before every call.
     *
     * @param mem_ptr pointer to the memory buffer to mark as shared with this kernel
     */
    public void share_mem(long mem_ptr)
    {
        shared_memory_ptrs.add(mem_ptr);
    }

    /**
     * Convenience method for setting kernel arguments that are pointers to memory. The primary use
     * case for this method is providing points to memory buffers that do not change at runtime. The
     * intent is that these pointers are set ars kernel arguments once at startup. This helps increase
     * efficiency as setting arguments on kernels involves a small driver overhead.
     * @param val the Enum value of the argument position to set
     * @param gpu_memory GPUMemory object containing the memory buffer which will be set as the argument
     * @return this kernel instance, allowing for chaining of argument setting calls
     */
    public GPUKernel<?> mem_arg(Enum<?> val, GPUMemory gpu_memory)
    {
        def_arg(val.ordinal(), Sizeof.cl_mem);
        ptr_arg(val.ordinal(), gpu_memory.memory().getNativePointer());
        return this;
    }

    /**
     * Sets a new argument to a null pointer. This is used for defining buffers that are local to the
     * GPU kernel execution. Buffers of this nature are sized but have no data transferred, the empty
     * buffer is created by the GPU and is destroyed after the call returns.
     * @param pos argument position
     * @param size size of the empty buffer to be created for the argument
     */
    public void loc_arg(int pos, long size)
    {
        def_arg(pos, size);
        CL12.clSetKernelArg(this.kernel_ptr, pos, size);
    }

    /**
     * Defines an argument position as having a specific size, without assigning it any data yet.
     * This is used to predefine argument sizes, typically before a kernel is used, so that the
     * values can be updated at runtime. This makes calling code more concise as the size value
     * does not need to be passed in.
     *
     * @param pos argument position
     * @param size size of the memory buffer being passed for the argument
     */
    public void def_arg(int pos, long size)
    {
        arg_sizes[pos] = size;
    }

    /**
     * Update an argument to contain a value from a provided buffer, replacing the value currently
     * set for the argument position. The size of the buffer is retained from previous calls to
     * new_arg and/or def_arg, so if the value being passed is of a different size, new_arg should be
     * called instead.
     *
     * @param pos argument position
     * @param pointer pointer to the memory buffer being passed
     */
    public void ptr_arg(int pos, long pointer)
    {
        try (var mem_stack = MemoryStack.stackPush())
        {
            var pointerBuffer = mem_stack.callocPointer(1).put(0, pointer);
            CL12.clSetKernelArg(this.kernel_ptr, pos, pointerBuffer);
        }
    }


    public void set_arg(int pos, double[] value)
    {
        try (var mem_stack = MemoryStack.stackPush())
        {
            var doubleBuffer = mem_stack.doubles(value);
            CL12.clSetKernelArg(this.kernel_ptr, pos, doubleBuffer);
        }
    }

    public void set_arg(int pos, double value)
    {
        try (var mem_stack = MemoryStack.stackPush())
        {
            var doubleBuffer = mem_stack.doubles(value);
            CL12.clSetKernelArg(this.kernel_ptr, pos, doubleBuffer);
        }
    }


    public void set_arg(int pos, float[] value)
    {
        try (var mem_stack = MemoryStack.stackPush())
        {
            var floatBuffer = mem_stack.floats(value);
            CL12.clSetKernelArg(this.kernel_ptr, pos, floatBuffer);
        }
    }

    public void set_arg(int pos, float value)
    {
        try (var mem_stack = MemoryStack.stackPush())
        {
            var floatBuffer = mem_stack.floats(value);
            CL12.clSetKernelArg(this.kernel_ptr, pos, floatBuffer);
        }
    }

    public void set_arg(int pos, int[] value)
    {
        try (var mem_stack = MemoryStack.stackPush())
        {
            var intBuffer = mem_stack.ints(value);
            CL12.clSetKernelArg(this.kernel_ptr, pos, intBuffer);
        }
    }

    public void set_arg(int pos, int value)
    {
        try (var mem_stack = MemoryStack.stackPush())
        {
            var intBuffer = mem_stack.ints(value);
            CL12.clSetKernelArg(this.kernel_ptr, pos, intBuffer);
        }
    }




    /**
     * Call this kernel, executing it on the GPU. This variant lets the GPu decide the best size for the local work
     * group. This is useful for cases where the work does not depend on properly sized local groups.
     *
     * @param global_work_size total number of threads that will execute in the call
     */
    public void call(long[] global_work_size)
    {
        call(global_work_size, null);
    }

    public void call(long[] global_work_size, long[] local_work_size)
    {
        call(global_work_size, local_work_size, null);
    }

    /**
     * Call this kernel, executing it on the GPU. Global and local work sizes are provided that tell the GPU
     * how best to divide up the work being done.
     *
     * @param global_work_size total number of threads that will execute in the call
     * @param local_work_size number of threads that will execute in a single work group
     */
    public void call(long[] global_work_size, long[] local_work_size, long[] global_work_offset)
    {
        if (!shared_memory_ptrs.isEmpty())
        {
            CLUtils.gl_acquire(command_queue_ptr, shared_memory_ptrs);
        }

        k_call(command_queue_ptr, kernel_ptr, global_work_size, local_work_size, global_work_offset);

        if (!shared_memory_ptrs.isEmpty())
        {
            CLUtils.gl_release(command_queue_ptr, shared_memory_ptrs);
        }

        shared_memory_ptrs.clear();
    }
}
