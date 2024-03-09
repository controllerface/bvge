package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPUKernel;

public class CreateMeshFace_k extends GPUKernel
{
    public enum Args
    {
        mesh_faces,
        target,
        new_mesh_face;
    }

    public CreateMeshFace_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
