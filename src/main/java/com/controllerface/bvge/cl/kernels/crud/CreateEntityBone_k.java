package com.controllerface.bvge.cl.kernels.crud;

import com.controllerface.bvge.cl.CLUtils;
import com.controllerface.bvge.cl.kernels.GPUKernel;
import com.controllerface.bvge.cl.kernels.Kernel;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

public class CreateEntityBone_k extends GPUKernel
{
    private static final String kernel_name = Kernel.create_entity_bone.name();
    private static String kernel_source = "";

    public enum Args
    {
        entity_bones("__global float16 *"),
        entity_bone_reference_ids("__global int *"),
        entity_bone_parent_ids("__global int *"),
        target("int"),
        new_armature_bone("float16"),
        new_armature_bone_reference("int"),
        new_armature_bone_parent_id("int"),

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

    public CreateEntityBone_k(long command_queue_ptr, long kernel_ptr)
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
