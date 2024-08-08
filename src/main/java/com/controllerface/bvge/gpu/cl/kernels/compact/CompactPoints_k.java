package com.controllerface.bvge.gpu.cl.kernels.compact;

import com.controllerface.bvge.gpu.GPU;
import com.controllerface.bvge.gpu.cl.CLUtils;
import com.controllerface.bvge.gpu.cl.CL_DataTypes;
import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;
import com.controllerface.bvge.gpu.cl.kernels.KernelType;
import com.controllerface.bvge.gpu.cl.kernels.KernelArg;

import static com.controllerface.bvge.memory.types.CoreBufferType.*;

public class CompactPoints_k extends GPUKernel
{
    public static final String kernel_source = GPU.CL.compact_k_src(KernelType.compact_points, Args.class);

    public enum Args implements KernelArg
    {
        point_shift             (CL_DataTypes.cl_int.buffer_name()),
        points                  (POINT.data_type().buffer_name()),
        anti_gravity            (POINT_ANTI_GRAV.data_type().buffer_name()),
        anti_time               (POINT_ANTI_TIME.data_type().buffer_name()),
        point_vertex_references (POINT_VERTEX_REFERENCE.data_type().buffer_name()),
        point_hull_indices      (POINT_HULL_INDEX.data_type().buffer_name()),
        point_flags             (POINT_FLAG.data_type().buffer_name()),
        point_hit_counts        (POINT_HIT_COUNT.data_type().buffer_name()),
        bone_tables             (POINT_BONE_TABLE.data_type().buffer_name()),

        ;

        private final String cl_type;
        Args(String clType) { cl_type = clType; }
        public String cl_type() { return cl_type; }
    }

    public CompactPoints_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
