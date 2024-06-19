package com.controllerface.bvge.cl.kernels.crud;

import com.controllerface.bvge.cl.CLUtils;
import com.controllerface.bvge.cl.kernels.GPUKernel;
import com.controllerface.bvge.cl.kernels.Kernel;

public class CreateVertexRef_k extends GPUKernel
{
    public static final String kernel_source = CLUtils.crud_k_src(Kernel.create_vertex_reference, Args.class);

    public enum Args implements KernelArg
    {
        vertex_references    (Type.float2_buffer),
        vertex_weights       (Type.float4_buffer),
        uv_tables            (Type.int2_buffer),
        target               (Type.int_arg),
        new_vertex_reference (Type.float2_arg),
        new_vertex_weights   (Type.float4_arg),
        new_uv_table         (Type.int2_arg),

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