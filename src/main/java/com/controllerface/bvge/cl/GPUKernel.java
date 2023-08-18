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

    public GPUKernel(cl_command_queue command_queue, cl_kernel kernel)
    {
        this.command_queue = command_queue;
        this.kernel = kernel;
    }

    public void share(cl_mem mem)
    {
        shared_memory.add(mem);
    }

    public void set_arg(int idx, int size, Pointer pointer)
    {
        clSetKernelArg(this.kernel, idx, size, pointer);
    }

    public void call(long[] global_work_size)
    {
        if (!shared_memory.isEmpty())
        {
            shared_memory.forEach(m->CLUtils.gl_acquire(command_queue, m));
        }
        k_call(command_queue, kernel, global_work_size);
        if (!shared_memory.isEmpty())
        {
            shared_memory.forEach(m->CLUtils.gl_release(command_queue, m));
        }
        shared_memory.clear();
    }
}
