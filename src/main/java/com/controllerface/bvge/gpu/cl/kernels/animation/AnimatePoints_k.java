package com.controllerface.bvge.gpu.cl.kernels.animation;

import com.controllerface.bvge.gpu.GPU;
import com.controllerface.bvge.gpu.cl.contexts.CL_CommandQueue;
import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;
import com.controllerface.bvge.gpu.cl.kernels.KernelType;
import com.controllerface.bvge.gpu.cl.programs.GPUProgram;

import static com.controllerface.bvge.memory.types.CoreBufferType.*;
import static com.controllerface.bvge.memory.types.ReferenceBufferType.VERTEX_REFERENCE;
import static com.controllerface.bvge.memory.types.ReferenceBufferType.VERTEX_WEIGHT;

public class AnimatePoints_k extends GPUKernel
{
    public enum Args
    {
        points,
        hull_scales,
        hull_entity_ids,
        hull_flags,
        point_vertex_references,
        point_hull_indices,
        bone_tables,
        vertex_weights,
        entities,
        vertex_references,
        bones,
        max_point,
    }

    public AnimatePoints_k(CL_CommandQueue command_queue_ptr, GPUProgram program)
    {
        super(command_queue_ptr, program.get_kernel(KernelType.animate_points));
    }

    public GPUKernel init()
    {
        return this.buf_arg(Args.points, GPU.memory.get_buffer(POINT))
            .buf_arg(Args.hull_scales, GPU.memory.get_buffer(HULL_SCALE))
            .buf_arg(Args.hull_entity_ids, GPU.memory.get_buffer(HULL_ENTITY_ID))
            .buf_arg(Args.hull_flags, GPU.memory.get_buffer(HULL_FLAG))
            .buf_arg(Args.point_vertex_references, GPU.memory.get_buffer(POINT_VERTEX_REFERENCE))
            .buf_arg(Args.point_hull_indices, GPU.memory.get_buffer(POINT_HULL_INDEX))
            .buf_arg(Args.bone_tables, GPU.memory.get_buffer(POINT_BONE_TABLE))
            .buf_arg(Args.vertex_weights, GPU.memory.get_buffer(VERTEX_WEIGHT))
            .buf_arg(Args.entities, GPU.memory.get_buffer(ENTITY))
            .buf_arg(Args.vertex_references, GPU.memory.get_buffer(VERTEX_REFERENCE))
            .buf_arg(Args.bones, GPU.memory.get_buffer(HULL_BONE));
    }
}
