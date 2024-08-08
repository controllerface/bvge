package com.controllerface.bvge.gpu.cl.kernels.crud;

import com.controllerface.bvge.gpu.GPU;
import com.controllerface.bvge.gpu.cl.buffers.CL_DataTypes;
import com.controllerface.bvge.gpu.cl.contexts.CL_CommandQueue;
import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;
import com.controllerface.bvge.gpu.cl.kernels.KernelType;
import com.controllerface.bvge.gpu.cl.kernels.KernelArg;
import com.controllerface.bvge.gpu.cl.programs.GPUProgram;

public class CreateVertexRef_k extends GPUKernel
{
    public static final String kernel_source = GPU.CL.crud_create_k_src(KernelType.create_vertex_reference, Args.class);

    public enum Args implements KernelArg
    {
        vertex_references    (CL_DataTypes.cl_float2.buffer_name()),
        vertex_weights       (CL_DataTypes.cl_float4.buffer_name()),
        uv_tables            (CL_DataTypes.cl_int2.buffer_name()),
        target               (CL_DataTypes.cl_int.name()),
        new_vertex_reference (CL_DataTypes.cl_float2.name()),
        new_vertex_weights   (CL_DataTypes.cl_float4.name()),
        new_uv_table         (CL_DataTypes.cl_int2.name()),

        ;

        private final String cl_type;
        Args(String clType) { cl_type = clType; }
        public String cl_type() { return cl_type; }
    }

    public CreateVertexRef_k(CL_CommandQueue command_queue_ptr, GPUProgram program)
    {
        super(command_queue_ptr, program.get_kernel(KernelType.create_vertex_reference));
    }
}
