package com.controllerface.bvge.cl.kernels.crud;

import com.controllerface.bvge.cl.CLUtils;
import com.controllerface.bvge.cl.kernels.GPUKernel;
import com.controllerface.bvge.cl.kernels.Kernel;

import static com.controllerface.bvge.cl.CLData.*;

public class CreateVertexRef_k extends GPUKernel
{
    public static final String kernel_source = CLUtils.crud_create_k_src(Kernel.create_vertex_reference, Args.class);

    public enum Args implements KernelArg
    {
        vertex_references    (cl_float2.buffer_name()),
        vertex_weights       (cl_float4.buffer_name()),
        uv_tables            (cl_int2.buffer_name()),
        target               (cl_int.name()),
        new_vertex_reference (cl_float2.name()),
        new_vertex_weights   (cl_float4.name()),
        new_uv_table         (cl_int2.name()),

        ;

        private final String cl_type;
        Args(String clType) { cl_type = clType; }
        public String cl_type() { return cl_type; }
    }

    public CreateVertexRef_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
