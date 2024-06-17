package com.controllerface.bvge.cl.kernels.crud;

import com.controllerface.bvge.cl.CLUtils;
import com.controllerface.bvge.cl.kernels.GPUKernel;
import com.controllerface.bvge.cl.kernels.Kernel;

public class CreateEdge_k extends GPUKernel
{
    public static final String kernel_source = CLUtils.crud_k_src(Kernel.create_edge, Args.class);

    public enum Args implements KernelArg
    {
        edges           (Type.int2_buffer),
        edge_lengths    (Type.float_buffer),
        edge_flags      (Type.int_buffer),
        target          (Type.int_arg),
        new_edge        (Type.int2_arg),
        new_edge_length (Type.float_arg),
        new_edge_flag   (Type.int_arg),

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
