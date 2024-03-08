package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;

public class TransferDetailData_k extends GPUKernel
{
    private static final GPU.Program program = GPU.Program.mesh_query;
    private static final GPU.Kernel kernel = GPU.Kernel.transfer_detail_data;

    public enum Args
    {
        mesh_details,
        mesh_transfer,
        offset;
    }

    public TransferDetailData_k(long command_queue_ptr)
    {
        super(command_queue_ptr, program.gpu.kernel_ptr(kernel));
    }
}
