package com.controllerface.bvge.cl;

import org.jocl.*;

import java.util.ArrayList;
import java.util.List;

import static com.controllerface.bvge.cl.CLUtils.k_call;
import static org.jocl.CL.clSetKernelArg;

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

    public void call(long[] global_work_size)
    {
        call(global_work_size, null);
    }

    public void call(long[] global_work_size, long[] local_work_size)
    {
        if (!shared_memory.isEmpty())
        {
            shared_memory.forEach(m->CLUtils.gl_acquire(command_queue, m));
        }
        k_call(command_queue, kernel, global_work_size, local_work_size);
        if (!shared_memory.isEmpty())
        {
            shared_memory.forEach(m->CLUtils.gl_release(command_queue, m));
        }
        shared_memory.clear();
    }
}
