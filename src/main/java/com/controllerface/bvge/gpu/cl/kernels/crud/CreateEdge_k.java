package com.controllerface.bvge.gpu.cl.kernels.crud;

import com.controllerface.bvge.gpu.GPU;
import com.controllerface.bvge.gpu.cl.contexts.CL_CommandQueue;
import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;
import com.controllerface.bvge.gpu.cl.kernels.KernelArg;
import com.controllerface.bvge.gpu.cl.kernels.KernelType;
import com.controllerface.bvge.gpu.cl.programs.GPUProgram;
import com.controllerface.bvge.memory.groups.CoreBufferGroup;

import static com.controllerface.bvge.gpu.cl.buffers.CL_DataTypes.*;
import static com.controllerface.bvge.memory.types.CoreBufferType.*;

public class CreateEdge_k extends GPUKernel
{
    public static final String kernel_source = GPU.CL.crud_create_k_src(KernelType.create_edge, Args.class);

    public enum Args implements KernelArg
    {
        edges(cl_int2.buffer_name()),
        edge_lengths(cl_float.buffer_name()),
        edge_flags(cl_int.buffer_name()),
        edge_pins(cl_int.buffer_name()),
        target(cl_int.name()),
        new_edge(cl_int2.name()),
        new_edge_length(cl_float.name()),
        new_edge_flag(cl_int.name()),
        new_edge_pin(cl_int.name()),

        ;

        private final String cl_type;
        Args(String clType) { cl_type = clType; }
        public String cl_type() { return cl_type; }
    }

    public CreateEdge_k(CL_CommandQueue command_queue_ptr, GPUProgram program)
    {
        super(command_queue_ptr, program.get_kernel(KernelType.create_edge));
    }

    public GPUKernel init(CoreBufferGroup core_buffers)
    {
        return this.buf_arg(Args.edges, core_buffers.buffer(EDGE))
            .buf_arg(Args.edge_lengths, core_buffers.buffer(EDGE_LENGTH))
            .buf_arg(Args.edge_flags, core_buffers.buffer(EDGE_FLAG))
            .buf_arg(Args.edge_pins, core_buffers.buffer(EDGE_PIN));
    }
}
