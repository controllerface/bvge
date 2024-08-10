package com.controllerface.bvge.gpu.cl.kernels.physics;

import com.controllerface.bvge.gpu.GPU;
import com.controllerface.bvge.gpu.cl.contexts.CL_CommandQueue;
import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;
import com.controllerface.bvge.gpu.cl.kernels.KernelType;
import com.controllerface.bvge.gpu.cl.programs.GPUProgram;

import static com.controllerface.bvge.memory.types.CoreBufferType.*;

public class ResolveConstraints_k extends GPUKernel
{
    public enum Args
    {
        hulls,
        hull_flags,
        entities,
        hull_edge_tables,
        bounds_bank_data,
        points,
        edges,
        edge_lengths,
        edge_flags,
        edge_pins,
        process_all,
        max_hull,
    }

    public ResolveConstraints_k(CL_CommandQueue command_queue_ptr, GPUProgram program)
    {
        super(command_queue_ptr, program.get_kernel(KernelType.resolve_constraints));
    }

    public GPUKernel init()
    {
        return this.buf_arg(Args.hulls, GPU.memory.get_buffer(HULL))
            .buf_arg(Args.hull_flags, GPU.memory.get_buffer(HULL_FLAG))
            .buf_arg(Args.entities, GPU.memory.get_buffer(ENTITY))
            .buf_arg(Args.hull_edge_tables, GPU.memory.get_buffer(HULL_EDGE_TABLE))
            .buf_arg(Args.bounds_bank_data, GPU.memory.get_buffer(HULL_AABB_KEY_TABLE))
            .buf_arg(Args.points, GPU.memory.get_buffer(POINT))
            .buf_arg(Args.edges, GPU.memory.get_buffer(EDGE))
            .buf_arg(Args.edge_lengths, GPU.memory.get_buffer(EDGE_LENGTH))
            .buf_arg(Args.edge_flags, GPU.memory.get_buffer(EDGE_FLAG))
            .buf_arg(Args.edge_pins, GPU.memory.get_buffer(EDGE_PIN));
    }
}
