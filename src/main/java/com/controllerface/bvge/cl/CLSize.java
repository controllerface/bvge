package com.controllerface.bvge.cl;



public class CLSize
{
    public static final int cl_char = 1;
    public static final int cl_uchar = 1;
    public static final int cl_short = 2;
    public static final int cl_ushort = 2;
    public static final int cl_int = 4;
    public static final int cl_uint = 4;
    public static final int cl_long = 8;
    public static final int cl_ulong = 8;
    public static final int cl_half = 2;
    public static final int cl_float = 4;
    public static final int cl_double = 8;
    public static final int cl_char2 = 2;
    public static final int cl_char4 = 4;
    public static final int cl_char8 = 8;
    public static final int cl_char16 = 16;
    public static final int cl_char3 = 4;
    public static final int cl_uchar2 = 2;
    public static final int cl_uchar4 = 4;
    public static final int cl_uchar8 = 8;
    public static final int cl_uchar16 = 16;
    public static final int cl_uchar3 = 4;
    public static final int cl_short2 = 4;
    public static final int cl_short4 = 8;
    public static final int cl_short8 = 16;
    public static final int cl_short16 = 32;
    public static final int cl_short3 = 8;
    public static final int cl_ushort2 = 4;
    public static final int cl_ushort4 = 8;
    public static final int cl_ushort8 = 16;
    public static final int cl_ushort16 = 32;
    public static final int cl_ushort3 = 8;
    public static final int cl_int2 = 8;
    public static final int cl_int4 = 16;
    public static final int cl_int8 = 32;
    public static final int cl_int16 = 64;
    public static final int cl_int3 = 16;
    public static final int cl_uint2 = 8;
    public static final int cl_uint4 = 16;
    public static final int cl_uint8 = 32;
    public static final int cl_uint16 = 64;
    public static final int cl_uint3 = 16;
    public static final int cl_long2 = 16;
    public static final int cl_long4 = 32;
    public static final int cl_long8 = 64;
    public static final int cl_long16 = 128;
    public static final int cl_long3 = 32;
    public static final int cl_ulong2 = 16;
    public static final int cl_ulong4 = 32;
    public static final int cl_ulong8 = 64;
    public static final int cl_ulong16 = 128;
    public static final int cl_ulong3 = 32;
    public static final int cl_float2 = 8;
    public static final int cl_float4 = 16;
    public static final int cl_float8 = 32;
    public static final int cl_float16 = 64;
    public static final int cl_float3 = 16;
    public static final int cl_double2 = 16;
    public static final int cl_double4 = 32;
    public static final int cl_double8 = 64;
    public static final int cl_double16 = 128;
    public static final int cl_double3 = 32;
    public static final int size_t;

    static
    {
        var bits = System.getProperty("sun.arch.data.model");
        int pointer_size = 0;
        if ("64".equals(bits))
        {
            pointer_size = 8;
        }
        else
        {
            pointer_size = 4;
        }
        size_t = pointer_size;
    }
}
