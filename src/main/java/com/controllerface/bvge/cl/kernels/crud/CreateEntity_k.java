package com.controllerface.bvge.cl.kernels.crud;

import com.controllerface.bvge.cl.CLUtils;
import com.controllerface.bvge.cl.kernels.GPUKernel;
import com.controllerface.bvge.cl.kernels.Kernel;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

public class CreateEntity_k extends GPUKernel
{
    private static final String kernel_name = Kernel.create_entity.name();
    private static String kernel_source = "";

    public enum Args
    {
        entities("__global float4 *"),
        entity_animation_elapsed("__global float2 *"),
        entity_motion_states("__global short2 *"),
        entity_animation_indices("__global int2 *"),
        entity_hull_tables("__global int2 *"),
        entity_bone_tables("__global int2 *"),
        entity_masses("__global float *"),
        entity_root_hulls("__global int *"),
        entity_model_indices("__global int *"),
        entity_model_transforms("__global int *"),
        entity_types("__global int *"),
        entity_flags("__global int *"),
        target("int"),
        new_entity("float4"),
        new_entity_animation_time("float2"),
        new_entity_animation_state("short2"),
        new_entity_animation_index("int2"),
        new_entity_hull_table("int2"),
        new_entity_bone_table("int2"),
        new_entity_mass("float"),
        new_entity_root_hull("int"),
        new_entity_model_id("int"),
        new_entity_model_transform("int"),
        new_entity_type("int"),
        new_entity_flags("int"),

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

    public CreateEntity_k(long command_queue_ptr, long kernel_ptr)
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
