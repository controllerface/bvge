package com.controllerface.bvge.cl;

/**
 * A utility class that has constants defined indicating the size (in bytes) of the most
 * common data types supported in Open CL kernels. These sizes are useful when working
 * directly with memory buffers. The system pointer size is also defined, based on the
 * architecture on which this class is loaded. While useful for some initialization tasks,
 * notably determining certain GPU settings and limits, it should probably not be relied
 * upon for other uses.
 */
public class CLData
{
    public record CLType(int size, String name)
    {
        public String buffer_name()
        {
            return String.join(" ", CLUtils.BUFFER_PREFIX, name(),  CLUtils.BUFFER_SUFFIX);
        }
    }

    // char
    public static final CLType cl_char   = new CLType(1,  "char");
    public static final CLType cl_char2  = new CLType(2,  "char2");
    public static final CLType cl_char3  = new CLType(4,  "char3");
    public static final CLType cl_char4  = new CLType(4,  "char4");
    public static final CLType cl_char8  = new CLType(8,  "char8");
    public static final CLType cl_char16 = new CLType(16, "char16");

    // unsigned char
    public static final CLType cl_uchar   = new CLType(1,  "uchar");
    public static final CLType cl_uchar2  = new CLType(2,  "uchar2");
    public static final CLType cl_uchar3  = new CLType(4,  "uchar3");
    public static final CLType cl_uchar4  = new CLType(4,  "uchar4");
    public static final CLType cl_uchar8  = new CLType(8,  "uchar8");
    public static final CLType cl_uchar16 = new CLType(16, "uchar16");

    // short
    public static final CLType cl_short   = new CLType(2,   "short");
    public static final CLType cl_short2  = new CLType(4,   "short2");
    public static final CLType cl_short3  = new CLType(8,   "short3");
    public static final CLType cl_short4  = new CLType(8,   "short4");
    public static final CLType cl_short8  = new CLType(16,  "short8");
    public static final CLType cl_short16 = new CLType(32,  "short16");

    // unsigned short
    public static final CLType cl_ushort   = new CLType(2,   "ushort");
    public static final CLType cl_ushort2  = new CLType(4,   "ushort2");
    public static final CLType cl_ushort3  = new CLType(8,   "ushort3");
    public static final CLType cl_ushort4  = new CLType(8,   "ushort4");
    public static final CLType cl_ushort8  = new CLType(16,  "ushort8");
    public static final CLType cl_ushort16 = new CLType(32,  "ushort16");

    // integer
    public static final CLType cl_int    = new CLType(4,   "int");
    public static final CLType cl_int2   = new CLType(8,   "int2");
    public static final CLType cl_int3   = new CLType(16,   "int3");
    public static final CLType cl_int4   = new CLType(16,   "int4");
    public static final CLType cl_int8   = new CLType(32,   "int8");
    public static final CLType cl_int16  = new CLType(64,   "int16");

    // unsigned integer
    public static final CLType cl_uint    = new CLType(4,   "uint");
    public static final CLType cl_uint2   = new CLType(8,   "uint2");
    public static final CLType cl_uint3   = new CLType(16,   "uint3");
    public static final CLType cl_uint4   = new CLType(16,   "uint4");
    public static final CLType cl_uint8   = new CLType(32,   "uint8");
    public static final CLType cl_uint16  = new CLType(64,   "uint16");

    // long
    public static final int cl_long = 8;
    public static final int cl_long2 = 16;
    public static final int cl_long3 = 32;
    public static final int cl_long4 = 32;
    public static final int cl_long8 = 64;
    public static final int cl_long16 = 128;

    // unsigned long
    public static final int cl_ulong = 8;
    public static final int cl_ulong2 = 16;
    public static final int cl_ulong3 = 32;
    public static final int cl_ulong4 = 32;
    public static final int cl_ulong8 = 64;
    public static final int cl_ulong16 = 128;

    // single-precision floating point
    public static final CLType cl_float   = new CLType(4,  "float");
    public static final CLType cl_float2  = new CLType(8,  "float2");
    public static final CLType cl_float3  = new CLType(16, "float3");
    public static final CLType cl_float4  = new CLType(16, "float4");
    public static final CLType cl_float8  = new CLType(32, "float8");
    public static final CLType cl_float16 = new CLType(64, "float16");


    // double-precision floating point
    public static final int cl_double = 8;
    public static final int cl_double2 = 16;
    public static final int cl_double3 = 32;
    public static final int cl_double4 = 32;
    public static final int cl_double8 = 64;
    public static final int cl_double16 = 128;

    // system pointer size
    public static final int size_t;

    static
    {
        var arch = System.getProperty("sun.arch.data.model");
        size_t = "64".equals(arch)
            ? 8
            : 4;
    }
}
