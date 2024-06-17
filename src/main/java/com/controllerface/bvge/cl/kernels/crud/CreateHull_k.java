package com.controllerface.bvge.cl.kernels.crud;

import com.controllerface.bvge.cl.CLUtils;
import com.controllerface.bvge.cl.kernels.GPUKernel;
import com.controllerface.bvge.cl.kernels.Kernel;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

public class CreateHull_k extends GPUKernel
{
    private static final String kernel_name = Kernel.create_hull.name();
    private static String kernel_source = "";

    public enum Args
    {
        hulls("__global float4 *"),
        hull_scales("__global float2 *"),
        hull_rotations("__global float2 *"),
        hull_frictions("__global float *"),
        hull_restitutions("__global float *"),
        hull_point_tables("__global int2 *"),
        hull_edge_tables("__global int2 *"),
        hull_bone_tables("__global int2 *"),
        hull_entity_ids("__global int *"),
        hull_flags("__global int *"),
        hull_mesh_ids("__global int *"),
        hull_uv_offsets("__global int *"),
        hull_integrity("__global int *"),
        target("int"),
        new_hull("float4"),
        new_hull_scale("float2"),
        new_rotation("float2"),
        new_friction("float"),
        new_restitution("float"),
        new_point_table("int2"),
        new_edge_table("int2"),
        new_bone_table("int2"),
        new_entity_id("int"),
        new_flags("int"),
        new_hull_mesh_id("int"),
        new_hull_uv_offset("int"),
        new_hull_integrity("int"),

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

    public CreateHull_k(long command_queue_ptr, long kernel_ptr)
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
