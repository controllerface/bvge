package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;
import com.controllerface.bvge.cl.GPUProgram;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;

public class TransferDetailData_k extends GPUKernel<TransferDetailData_k.Args>
{
    public enum Args implements GPUKernelArg
    {
        mesh_details(Sizeof.cl_mem),
        mesh_transfer(Sizeof.cl_mem),
        offset(Sizeof.cl_int);

        public final long size;
        Args(long size) { this.size = size; }
        @Override public long size() { return size; }
    }

    public TransferDetailData_k(long command_queue_ptr)
    {
        super(command_queue_ptr, GPU.Program.mesh_query.gpu.kernels().get(GPU.Kernel.transfer_detail_data), Args.values());
    }
}
