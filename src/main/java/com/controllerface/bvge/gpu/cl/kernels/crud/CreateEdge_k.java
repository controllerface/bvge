package com.controllerface.bvge.gpu.cl.kernels.crud;

import com.controllerface.bvge.gpu.GPU;
import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;
import com.controllerface.bvge.gpu.cl.kernels.KernelType;
import com.controllerface.bvge.gpu.cl.kernels.KernelArg;

import static com.controllerface.bvge.gpu.cl.buffers.CL_DataTypes.*;

public class CreateEdge_k extends GPUKernel
{
    public static final String kernel_source = GPU.CL.crud_create_k_src(KernelType.create_edge, Args.class);

    public enum Args implements KernelArg
    {
        edges           (cl_int2.buffer_name()),
        edge_lengths    (cl_float.buffer_name()),
        edge_flags      (cl_int.buffer_name()),
        edge_pins       (cl_int.buffer_name()),
        target          (cl_int.name()),
        new_edge        (cl_int2.name()),
        new_edge_length (cl_float.name()),
        new_edge_flag   (cl_int.name()),
        new_edge_pin    (cl_int.name()),

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
