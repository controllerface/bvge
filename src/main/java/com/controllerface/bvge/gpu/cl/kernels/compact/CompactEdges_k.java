package com.controllerface.bvge.gpu.cl.kernels.compact;

import com.controllerface.bvge.gpu.GPU;
import com.controllerface.bvge.gpu.cl.buffers.CL_DataTypes;
import com.controllerface.bvge.gpu.cl.buffers.ResizableBuffer;
import com.controllerface.bvge.gpu.cl.contexts.CL_CommandQueue;
import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;
import com.controllerface.bvge.gpu.cl.kernels.KernelArg;
import com.controllerface.bvge.gpu.cl.kernels.KernelType;
import com.controllerface.bvge.gpu.cl.programs.GPUProgram;
import com.controllerface.bvge.memory.groups.CoreBufferGroup;

import static com.controllerface.bvge.memory.types.CoreBufferType.*;

public class CompactEdges_k extends GPUKernel
{
    public static final String kernel_source = GPU.CL.compact_k_src(KernelType.compact_edges, Args.class);

    public enum Args implements KernelArg
    {
        edge_shift          (CL_DataTypes.cl_int.buffer_name()),
        edges               (EDGE.data_type().buffer_name()),
        edge_lengths        (EDGE_LENGTH.data_type().buffer_name()),
        edge_flags          (EDGE_FLAG.data_type().buffer_name()),
        edge_pins           (EDGE_PIN.data_type().buffer_name()),

        ;

        private final String cl_type;
        Args(String clType) { cl_type = clType; }
        public String cl_type() { return cl_type; }
    }

    public CompactEdges_k(CL_CommandQueue command_queue_ptr, GPUProgram program)
    {
        super(command_queue_ptr, program.get_kernel(KernelType.compact_edges));
    }

    public GPUKernel init(CoreBufferGroup sector_buffers, ResizableBuffer b_edge_shift)
    {
        return this.buf_arg(Args.edge_shift, b_edge_shift)
            .buf_arg(Args.edges, sector_buffers.buffer(EDGE))
            .buf_arg(Args.edge_lengths, sector_buffers.buffer(EDGE_LENGTH))
            .buf_arg(Args.edge_flags, sector_buffers.buffer(EDGE_FLAG))
            .buf_arg(Args.edge_pins, sector_buffers.buffer(EDGE_PIN));
    }
}
