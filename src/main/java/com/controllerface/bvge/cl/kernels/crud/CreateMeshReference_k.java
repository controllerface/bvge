package com.controllerface.bvge.cl.kernels.crud;

import com.controllerface.bvge.cl.CLUtils;
import com.controllerface.bvge.cl.kernels.GPUKernel;
import com.controllerface.bvge.cl.kernels.Kernel;

public class CreateMeshReference_k extends GPUKernel
{
    public static final String kernel_source = CLUtils.crud_k_src(Kernel.create_mesh_reference, Args.class);

    public enum Args implements KernelArg
    {
        mesh_vertex_tables    (Type.int2_buffer),
        mesh_face_tables      (Type.int2_buffer),
        target                (Type.int_arg),
        new_mesh_vertex_table (Type.int2_arg),
        new_mesh_face_table   (Type.int2_arg),

        ;

        private final String cl_type;
        Args(String clType) { cl_type = clType; }
        public String cl_type() { return cl_type; }
    }

    public CreateMeshReference_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
