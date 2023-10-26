package com.controllerface.bvge.cl;

import org.jocl.*;

import java.util.ArrayList;
import java.util.List;

import static com.controllerface.bvge.cl.CLUtils.k_call;
import static org.jocl.CL.clSetKernelArg;

/**
 * A class for defining and organizing GPU kernel functions. This class is used to wrap an Open Cl kernel
 * making it easier to reason about in the host Java code. Helpers are provided for setting arguments
 * and calling the kernel itself. Kernels are typically provided by {@link GPUProgram} objects, which are
 * compiled first before their constituent kernels are linked and loaded via implementations of this class.
 */
public class GPUKernel
{
    final cl_command_queue command_queue;
    final cl_kernel kernel;
    final List<cl_mem> shared_memory = new ArrayList<>();
    final long[] arg_sizes;

    public GPUKernel(cl_command_queue command_queue, cl_kernel kernel, int arg_count)
    {
        this.command_queue = command_queue;
        this.kernel = kernel;
        this.arg_sizes = new long[arg_count];
    }

    /**
     * Allows calling code to designate a memory buffer as being shared between Open CL and Open Gl contexts.
     * This allows the kernel call to automatically acquire and release the shared objects, before and after
     * the kernel call, respectively. The shared memory list si also cleared after each call, so memory
     * objects must be shared individually before every call.
     *
     * @param mem the memory buffer to mark as shared with this kernel
     */
    public void share_mem(cl_mem mem)
    {
        shared_memory.add(mem);
    }

    /**
     * Sets a new argument in this kernel. Can be used for arguments that are set once and then
     * do not change for the life of the program, though calling code is not prevented from
     * calling this function to update arguments. Other options are provided solely for convenience
     * to calling code.
     *
     * @param pos argument position
     * @param size size of the memory buffer being passed for the argument
     * @param pointer pointer to the memory buffer being passed
     */
    public void new_arg(int pos, long size, Pointer pointer)
    {
        def_arg(pos, size);
        clSetKernelArg(this.kernel, pos, size, pointer);
    }

    /**
     * Sets a new argument to a null pointer. This is used for defining buffers that are local to the
     * GPU kernel execution. Buffers of this nature are sized but have no data transferred, the empty
     * buffer is created by the GPU and is destroyed after the call returns.
     * @param pos argument position
     * @param size size of the empty buffer to be created for the argument
     */
    public void new_arg(int pos, long size)
    {
        new_arg(pos, size, null);
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
    public void set_arg(int pos, Pointer pointer)
    {
        clSetKernelArg(this.kernel, pos, arg_sizes[pos], pointer);
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

    /**
     * Call this kernel, executing it on the GPU. Global and local work sizes are provided that tell the GPU
     * how best to divide up the work being done.
     *
     * @param global_work_size total number of threads that will execute in the call
     * @param local_work_size number of threads that will execute in a single work group
     */
    public void call(long[] global_work_size, long[] local_work_size)
    {
        var shared_ = shared_memory.toArray(new cl_mem[]{});

        if (shared_.length > 0)
        {
            CLUtils.gl_acquire(command_queue, shared_);
        }

        k_call(command_queue, kernel, global_work_size, local_work_size);

        if (shared_.length > 0)
        {
            CLUtils.gl_release(command_queue, shared_);
        }

        shared_memory.clear();
    }
}
