package com.controllerface.bvge.cl.kernels.crud;

import com.controllerface.bvge.cl.CLUtils;
import com.controllerface.bvge.cl.kernels.GPUKernel;
import com.controllerface.bvge.cl.kernels.Kernel;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

public class CreateBoneChannel_k extends GPUKernel
{
    private static final String kernel_name = Kernel.create_bone_channel.name();
    private static String kernel_source = "";

    public enum Args
    {
        animation_timing_indices("__global int *"),
        bone_pos_channel_tables("__global int2 *"),
        bone_rot_channel_tables("__global int2 *"),
        bone_scl_channel_tables("__global int2 *"),
        target("int"),
        new_animation_timing_index("int"),
        new_bone_pos_channel_table("int2"),
        new_bone_rot_channel_table("int2"),
        new_bone_scl_channel_table("int2"),

        ;

        private static final Map<Enum<?>, String> type_map = new HashMap<>();
        private final String cl_type;

        static
        {
            Arrays.stream(values())
                .forEach(arg -> type_map.put(arg, arg.cl_type));
        }

        Args(String clType)
        {
            cl_type = clType;
        }
    }

    public CreateBoneChannel_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }

    public static String cl_kernel()
    {
        if (kernel_source.isEmpty())
        {
            kernel_source = CLUtils.crud_kernel(Args.target.ordinal(), kernel_name, Args.class, Args.type_map);
        }
        return kernel_source;
    }
}
