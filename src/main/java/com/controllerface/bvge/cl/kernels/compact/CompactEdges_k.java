package com.controllerface.bvge.cl.kernels.compact;

import com.controllerface.bvge.cl.CLUtils;
import com.controllerface.bvge.cl.kernels.GPUKernel;
import com.controllerface.bvge.cl.kernels.Kernel;
import com.controllerface.bvge.cl.kernels.KernelArg;

import static com.controllerface.bvge.cl.CLData.*;
import static com.controllerface.bvge.cl.buffers.CoreBufferType.*;

public class CompactEdges_k extends GPUKernel
{
    public static final String kernel_source = CLUtils.compact_k_src(Kernel.compact_edges, Args.class);

    public enum Args implements KernelArg
    {
        edge_shift          (cl_int.buffer_name()),
        edges               (EDGE.data_type().buffer_name()),
        edge_lengths        (EDGE_LENGTH.data_type().buffer_name()),
        edge_flags          (EDGE_FLAG.data_type().buffer_name()),
        edge_aabb           (EDGE_AABB.data_type().buffer_name()),
        edge_aabb_index     (EDGE_AABB_INDEX.data_type().buffer_name()),
        edge_aabb_key_table (EDGE_AABB_KEY_TABLE.data_type().buffer_name()),

        ;

        private final String cl_type;
        Args(String clType) { cl_type = clType; }
        public String cl_type() { return cl_type; }
    }

    public CompactEdges_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
