package com.controllerface.bvge.cl.kernels.crud;

public interface KernelArg
{
    class Type
    {
        private static final String BUFFER_PREFIX = "__global";
        private static final String BUFFER_SUFFIX = "*";
        private static final String SPACE         = " ";

        public static String arg_short   = "short";
        public static String arg_short2  = "short2";
        public static String arg_int     = "int";
        public static String arg_int2    = "int2";
        public static String arg_int4    = "int4";
        public static String arg_float   = "float";
        public static String arg_float2  = "float2";
        public static String arg_float4  = "float4";
        public static String arg_float16 = "float16";

        public static String buffer_short   = String.join(SPACE, BUFFER_PREFIX, arg_short,   BUFFER_SUFFIX);
        public static String buffer_short2  = String.join(SPACE, BUFFER_PREFIX, arg_short2,  BUFFER_SUFFIX);
        public static String buffer_int     = String.join(SPACE, BUFFER_PREFIX, arg_int,     BUFFER_SUFFIX);
        public static String buffer_int2    = String.join(SPACE, BUFFER_PREFIX, arg_int2,    BUFFER_SUFFIX);
        public static String buffer_int4    = String.join(SPACE, BUFFER_PREFIX, arg_int4,    BUFFER_SUFFIX);
        public static String buffer_float   = String.join(SPACE, BUFFER_PREFIX, arg_float,   BUFFER_SUFFIX);
        public static String buffer_float2  = String.join(SPACE, BUFFER_PREFIX, arg_float2,  BUFFER_SUFFIX);
        public static String buffer_float4  = String.join(SPACE, BUFFER_PREFIX, arg_float4,  BUFFER_SUFFIX);
        public static String buffer_float16 = String.join(SPACE, BUFFER_PREFIX, arg_float16, BUFFER_SUFFIX);
    }

    String cl_type();
}
