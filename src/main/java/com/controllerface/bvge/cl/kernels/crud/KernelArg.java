package com.controllerface.bvge.cl.kernels.crud;

public interface KernelArg
{
    class Type
    {
        public static final String BUFFER_PREFIX = "__global";
        public static final String BUFFER_SUFFIX = "*";

    }

    String cl_type();
}
