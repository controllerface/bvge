package com.controllerface.bvge.cl.kernels.crud;

public interface KernelArg
{
    class Type
    {
        public static String short_arg   = "short";
        public static String short2_arg  = "short2";
        public static String int_arg     = "int";
        public static String int2_arg    = "int2";
        public static String int4_arg    = "int4";
        public static String float_arg   = "float";
        public static String float2_arg  = "float2";
        public static String float4_arg  = "float4";
        public static String float16_arg = "float16";

        public static String short_buffer   = "__global short *";
        public static String short2_buffer  = "__global short2 *";
        public static String int_buffer     = "__global int *";
        public static String int2_buffer    = "__global int2 *";
        public static String int4_buffer    = "__global int4 *";
        public static String float_buffer   = "__global float *";
        public static String float2_buffer  = "__global float2 *";
        public static String float4_buffer  = "__global float4 *";
        public static String float16_buffer = "__global float16 *";
    }

    String cl_type();
}
