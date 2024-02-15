package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;
import com.controllerface.bvge.cl.GPUProgram;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;

public class CompactHulls_k extends GPUKernel<CompactHulls_k.Args>
{
    private static final GPU.Program program = GPU.Program.scan_deletes;
    private static final GPU.Kernel kernel = GPU.Kernel.compact_hulls;

    public enum Args implements GPUKernelArg
    {
        hull_shift(Sizeof.cl_mem),
        hulls(Sizeof.cl_mem),
        hull_mesh_ids(Sizeof.cl_mem),
        hull_rotations(Sizeof.cl_mem),
        hull_flags(Sizeof.cl_mem),
        element_tables(Sizeof.cl_mem),
        bounds(Sizeof.cl_mem),
        bounds_index_data(Sizeof.cl_mem),
        bounds_bank_data(Sizeof.cl_mem);

        public final long size;
        Args(long size) { this.size = size; }
        @Override public long size() { return size; }
    }

    public CompactHulls_k(long command_queue_ptr)
    {
        super(command_queue_ptr, program.kernel_ptr(kernel), Args.values());
    }
}
