package com.controllerface.bvge.cl.kernels;

public class TransferDetailData_k extends GPUKernel
{
    public enum Args
    {
        mesh_details,
        mesh_transfer,
        offset;
    }

    public TransferDetailData_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
