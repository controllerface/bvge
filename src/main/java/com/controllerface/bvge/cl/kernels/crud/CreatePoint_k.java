package com.controllerface.bvge.cl.kernels.crud;

import com.controllerface.bvge.cl.CLUtils;
import com.controllerface.bvge.cl.kernels.GPUKernel;
import com.controllerface.bvge.cl.kernels.Kernel;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

public class CreatePoint_k extends GPUKernel
{
    private static final String kernel_name = Kernel.create_point.name();
    private static String kernel_source = "";

    public enum Args
    {
        points("__global float4 *"),
        point_vertex_references("__global int *"),
        point_hull_indices("__global int *"),
        point_hit_counts("__global short *"),
        point_flags("__global int *"),
        point_bone_tables("__global int4 *"),
        target("int"),
        new_point("float4"),
        new_point_vertex_reference("int"),
        new_point_hull_index("int"),
        new_point_hit_count("short"),
        new_point_flags("int"),
        new_point_bone_table("int4"),

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

    public CreatePoint_k(long command_queue_ptr, long kernel_ptr)
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
