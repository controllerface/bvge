package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;

public class CompactHulls_k extends GPUKernel
{
    private static final GPU.Program program = GPU.Program.scan_deletes;
    private static final GPU.Kernel kernel = GPU.Kernel.compact_hulls;

    public enum Args
    {
        hull_shift,
        hulls,
        hull_mesh_ids,
        hull_rotations,
        hull_flags,
        element_tables,
        bounds,
        bounds_index_data,
        bounds_bank_data;
    }

    public CompactHulls_k(long command_queue_ptr)
    {
        super(command_queue_ptr, program.kernel_ptr(kernel));
    }
}
