package com.controllerface.bvge.gpu.cl.kernels.crud;

import com.controllerface.bvge.gpu.GPU;
import com.controllerface.bvge.gpu.cl.CLUtils;
import com.controllerface.bvge.gpu.cl.CL_DataTypes;
import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;
import com.controllerface.bvge.gpu.cl.kernels.KernelType;
import com.controllerface.bvge.gpu.cl.kernels.KernelArg;

public class CreateMeshReference_k extends GPUKernel
{
    public static final String kernel_source = GPU.CL.crud_create_k_src(KernelType.create_mesh_reference, Args.class);

    public enum Args implements KernelArg
    {
        mesh_vertex_tables    (CL_DataTypes.cl_int2.buffer_name()),
        mesh_face_tables      (CL_DataTypes.cl_int2.buffer_name()),
        target                (CL_DataTypes.cl_int.name()),
        new_mesh_vertex_table (CL_DataTypes.cl_int2.name()),
        new_mesh_face_table   (CL_DataTypes.cl_int2.name()),

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
