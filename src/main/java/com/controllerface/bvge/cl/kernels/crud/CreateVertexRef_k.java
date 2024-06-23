package com.controllerface.bvge.cl.kernels.crud;

import com.controllerface.bvge.cl.CLUtils;
import com.controllerface.bvge.cl.kernels.GPUKernel;
import com.controllerface.bvge.cl.kernels.Kernel;

public class CreateVertexRef_k extends GPUKernel
{
    public static final String kernel_source = CLUtils.crud_create_k_src(Kernel.create_vertex_reference, Args.class);

    public enum Args implements KernelArg
    {
        vertex_references    (Type.buffer_float2),
        vertex_weights       (Type.buffer_float4),
        uv_tables            (Type.buffer_int2),
        target               (Type.arg_int),
        new_vertex_reference (Type.arg_float2),
        new_vertex_weights   (Type.arg_float4),
        new_uv_table         (Type.arg_int2),

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
