package com.controllerface.bvge.gpu.cl.kernels.egress;

import com.controllerface.bvge.gpu.cl.buffers.ResizableBuffer;
import com.controllerface.bvge.gpu.cl.contexts.CL_CommandQueue;
import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;
import com.controllerface.bvge.gpu.cl.kernels.KernelType;
import com.controllerface.bvge.gpu.cl.programs.GPUProgram;
import com.controllerface.bvge.memory.GPUCoreMemory;
import com.controllerface.bvge.memory.groups.UnorderedCoreBufferGroup;

import static com.controllerface.bvge.memory.types.CoreBufferType.*;

public class EgressEdges_k extends GPUKernel
{
    public enum Args
    {
        edges_in,
        edge_lengths_in,
        edge_flags_in,
        edge_pins_in,
        edges_out,
        edge_lengths_out,
        edge_flags_out,
        edge_pins_out,
        new_edges,
        max_edge,
    }

    public EgressEdges_k(CL_CommandQueue command_queue_ptr, GPUProgram program)
    {
        super(command_queue_ptr, program.get_kernel(KernelType.egress_edges));
    }

    public GPUKernel init(GPUCoreMemory core_memory,
                          UnorderedCoreBufferGroup sector_buffers,
                          ResizableBuffer b_edge_shift)
    {
        return this.buf_arg(Args.edges_in, core_memory.get_buffer(EDGE))
            .buf_arg(Args.edge_lengths_in, core_memory.get_buffer(EDGE_LENGTH))
            .buf_arg(Args.edge_flags_in, core_memory.get_buffer(EDGE_FLAG))
            .buf_arg(Args.edge_pins_in, core_memory.get_buffer(EDGE_PIN))
            .buf_arg(Args.edges_out, sector_buffers.buffer(EDGE))
            .buf_arg(Args.edge_lengths_out, sector_buffers.buffer(EDGE_LENGTH))
            .buf_arg(Args.edge_flags_out, sector_buffers.buffer(EDGE_FLAG))
            .buf_arg(Args.edge_pins_out, sector_buffers.buffer(EDGE_PIN))
            .buf_arg(Args.new_edges, b_edge_shift);
    }
}
