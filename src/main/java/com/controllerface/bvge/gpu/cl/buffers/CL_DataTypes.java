package com.controllerface.bvge.gpu.cl.buffers;

import com.controllerface.bvge.gpu.cl.CLUtils;

/**
 * A utility class that has constants defined indicating the size (in bytes) of the most
 * common data types supported in Open CL kernels. These sizes are useful when working
 * directly with memory buffers. The system pointer size is also defined, based on the
 * architecture on which this class is loaded. While useful for some initialization tasks,
 * notably determining certain GPU settings and limits, it should probably not be relied
 * upon for other uses.
 */
public class CL_DataTypes
{
    public record CL_Type(int size, String name)
    {
        public String buffer_name()
        {
            return String.join(" ", CLUtils.BUFFER_PREFIX, name(),  CLUtils.BUFFER_SUFFIX);
        }
    }

    public static final CL_Type cl_char      = new CL_Type(1,  "char");
    public static final CL_Type cl_char2     = new CL_Type(2,  "char2");
    public static final CL_Type cl_char3     = new CL_Type(4,  "char3");
    public static final CL_Type cl_char4     = new CL_Type(4,  "char4");
    public static final CL_Type cl_char8     = new CL_Type(8,  "char8");
    public static final CL_Type cl_char16    = new CL_Type(16, "char16");

    public static final CL_Type cl_uchar     = new CL_Type(1,  "uchar");
    public static final CL_Type cl_uchar2    = new CL_Type(2,  "uchar2");
    public static final CL_Type cl_uchar3    = new CL_Type(4,  "uchar3");
    public static final CL_Type cl_uchar4    = new CL_Type(4,  "uchar4");
    public static final CL_Type cl_uchar8    = new CL_Type(8,  "uchar8");
    public static final CL_Type cl_uchar16   = new CL_Type(16, "uchar16");

    public static final CL_Type cl_short     = new CL_Type(2,   "short");
    public static final CL_Type cl_short2    = new CL_Type(4,   "short2");
    public static final CL_Type cl_short3    = new CL_Type(8,   "short3");
    public static final CL_Type cl_short4    = new CL_Type(8,   "short4");
    public static final CL_Type cl_short8    = new CL_Type(16,  "short8");
    public static final CL_Type cl_short16   = new CL_Type(32,  "short16");

    public static final CL_Type cl_ushort    = new CL_Type(2,   "ushort");
    public static final CL_Type cl_ushort2   = new CL_Type(4,   "ushort2");
    public static final CL_Type cl_ushort3   = new CL_Type(8,   "ushort3");
    public static final CL_Type cl_ushort4   = new CL_Type(8,   "ushort4");
    public static final CL_Type cl_ushort8   = new CL_Type(16,  "ushort8");
    public static final CL_Type cl_ushort16  = new CL_Type(32,  "ushort16");

    public static final CL_Type cl_int       = new CL_Type(4,   "int");
    public static final CL_Type cl_int2      = new CL_Type(8,   "int2");
    public static final CL_Type cl_int3      = new CL_Type(16,  "int3");
    public static final CL_Type cl_int4      = new CL_Type(16,  "int4");
    public static final CL_Type cl_int8      = new CL_Type(32,  "int8");
    public static final CL_Type cl_int16     = new CL_Type(64,  "int16");

    public static final CL_Type cl_uint      = new CL_Type(4,   "uint");
    public static final CL_Type cl_uint2     = new CL_Type(8,   "uint2");
    public static final CL_Type cl_uint3     = new CL_Type(16,  "uint3");
    public static final CL_Type cl_uint4     = new CL_Type(16,  "uint4");
    public static final CL_Type cl_uint8     = new CL_Type(32,  "uint8");
    public static final CL_Type cl_uint16    = new CL_Type(64,  "uint16");

    public static final CL_Type cl_long      = new CL_Type(8,   "long");
    public static final CL_Type cl_long2     = new CL_Type(16,  "long2");
    public static final CL_Type cl_long3     = new CL_Type(32,  "long3");
    public static final CL_Type cl_long4     = new CL_Type(32,  "long4");
    public static final CL_Type cl_long8     = new CL_Type(64,  "long8");
    public static final CL_Type cl_long16    = new CL_Type(128, "long16");

    public static final CL_Type cl_ulong     = new CL_Type(8,   "ulong");
    public static final CL_Type cl_ulong2    = new CL_Type(16,  "ulong2");
    public static final CL_Type cl_ulong3    = new CL_Type(32,  "ulong3");
    public static final CL_Type cl_ulong4    = new CL_Type(32,  "ulong4");
    public static final CL_Type cl_ulong8    = new CL_Type(64,  "ulong8");
    public static final CL_Type cl_ulong16   = new CL_Type(128, "ulong16");

    public static final CL_Type cl_float     = new CL_Type(4,   "float");
    public static final CL_Type cl_float2    = new CL_Type(8,   "float2");
    public static final CL_Type cl_float3    = new CL_Type(16,  "float3");
    public static final CL_Type cl_float4    = new CL_Type(16,  "float4");
    public static final CL_Type cl_float8    = new CL_Type(32,  "float8");
    public static final CL_Type cl_float16   = new CL_Type(64,  "float16");

    public static final CL_Type cl_double    = new CL_Type(8,   "double");
    public static final CL_Type cl_double2   = new CL_Type(16,  "double2");
    public static final CL_Type cl_double3   = new CL_Type(32,  "double3");
    public static final CL_Type cl_double4   = new CL_Type(32,  "double4");
    public static final CL_Type cl_double8   = new CL_Type(64,  "double8");
    public static final CL_Type cl_double16  = new CL_Type(128, "double16");
}
