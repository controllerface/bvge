package com.controllerface.bvge.cl.kernels.crud;

import com.controllerface.bvge.cl.CLUtils;
import com.controllerface.bvge.cl.kernels.GPUKernel;
import com.controllerface.bvge.cl.kernels.Kernel;

public class CreateEdge_k extends GPUKernel
{
    public static final String kernel_source = CLUtils.crud_create_k_src(Kernel.create_edge, Args.class);

    public enum Args implements KernelArg
    {
        edges           (Type.buffer_int2),
        edge_lengths    (Type.buffer_float),
        edge_flags      (Type.buffer_int),
        target          (Type.arg_int),
        new_edge        (Type.arg_int2),
        new_edge_length (Type.arg_float),
        new_edge_flag   (Type.arg_int),

        ;

        private final String cl_type;
        Args(String clType) { cl_type = clType; }
        public String cl_type() { return cl_type; }
    }

    public CreateEdge_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
