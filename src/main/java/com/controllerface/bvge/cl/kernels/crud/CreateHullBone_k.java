package com.controllerface.bvge.cl.kernels.crud;

import com.controllerface.bvge.cl.CLUtils;
import com.controllerface.bvge.cl.kernels.GPUKernel;
import com.controllerface.bvge.cl.kernels.Kernel;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

public class CreateHullBone_k extends GPUKernel
{
    private static final String kernel_name = Kernel.create_hull_bone.name();
    private static String kernel_source = "";

    public enum Args
    {
        hull_bones("__global float16 *"),
        hull_bind_pose_indicies("__global int *"),
        hull_inv_bind_pose_indicies("__global int *"),
        target("int"),
        new_hull_bone("float16"),
        new_hull_bind_pose_id("int"),
        new_hull_inv_bind_pose_id("int"),

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

    public CreateHullBone_k(long command_queue_ptr, long kernel_ptr)
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
