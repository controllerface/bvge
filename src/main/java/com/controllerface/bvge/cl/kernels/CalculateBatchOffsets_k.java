package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;
import com.controllerface.bvge.cl.GPUProgram;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;

public class CalculateBatchOffsets_k extends GPUKernel<CalculateBatchOffsets_k.Args>
{
    public enum Args implements GPUKernelArg
    {
        mesh_offsets(Sizeof.cl_mem),
        mesh_details(Sizeof.cl_mem),
        count(Sizeof.cl_int);

        public final long size;
        Args(long size) { this.size = size; }
        @Override public long size() { return size; }
    }

    public CalculateBatchOffsets_k(long command_queue_ptr)
    {
        super(command_queue_ptr, GPU.Program.mesh_query.gpu.kernels().get(GPU.Kernel.calculate_batch_offsets), Args.values());
    }
}
