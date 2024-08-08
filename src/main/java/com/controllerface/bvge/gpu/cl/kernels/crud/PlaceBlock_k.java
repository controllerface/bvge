package com.controllerface.bvge.gpu.cl.kernels.crud;

import com.controllerface.bvge.gpu.cl.contexts.CL_CommandQueue;
import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;
import com.controllerface.bvge.gpu.cl.kernels.KernelType;
import com.controllerface.bvge.gpu.cl.programs.GPUProgram;

public class PlaceBlock_k extends GPUKernel
{
    public enum Args
    {
        entities,
        entity_hull_tables,
        hulls,
        hull_point_tables,
        hull_rotations,
        points,
        src,
        dest,
    }

    public PlaceBlock_k(CL_CommandQueue command_queue_ptr, GPUProgram program)
    {
        super(command_queue_ptr, program.get_kernel(KernelType.place_block));
    }
}
