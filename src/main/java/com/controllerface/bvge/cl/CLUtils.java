package com.controllerface.bvge.cl;

import org.jocl.*;
import org.joml.Matrix4f;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;

import static org.jocl.CL.*;

public class CLUtils
{
    public static String read_src(String file)
    {
        try (var stream = GPU.class.getResourceAsStream("/cl/" + file))
        {
            byte [] bytes = stream.readAllBytes();
            return new String(bytes, StandardCharsets.UTF_8);
        }
        catch (IOException e)
        {
            throw new RuntimeException(e);
        }
    }

    public static cl_program cl_p(cl_context context, cl_device_id[] device_ids, String ... src)
    {
        var program = clCreateProgramWithSource(context, src.length, src, null, null);
        clBuildProgram(program, 1, device_ids, null, null, null);
        return program;
    }

    public static cl_kernel cl_k(cl_program program, String kernel_name)
    {
        return clCreateKernel(program, kernel_name, null);
    }

    public static double[] arg_double(double arg)
    {
        return new double[]{arg};
    }

    public static long[] arg_long(long arg)
    {
        return new long[]{arg};
    }

    public static int[] arg_int(int arg)
    {
        return new int[]{arg};
    }

    public static int[] arg_int2(int x, int y)
    {
        return new int[]{x, y};
    }

    public static int[] arg_int4(int x, int y, int z, int w)
    {
        return new int[]{x, y, z, w};
    }

    public static float[] arg_float(float arg)
    {
        return new float[]{arg};
    }

    public static float[] arg_float2(float x, float y)
    {
        return new float[]{x, y};
    }

    public static float[] arg_float4(float x, float y, float z, float w)
    {
        return new float[]{x, y, z, w};
    }

    public static float[] arg_float16(float s0, float s1, float s2, float s3,
                                      float s4, float s5, float s6, float s7,
                                      float s8, float s9, float sA, float sB,
                                      float sC, float sD, float sE, float sF)
    {
        return new float[]
            {
                s0, s1, s2, s3,
                s4, s5, s6, s7,
                s8, s9, sA, sB,
                sC, sD, sE, sF
            };
    }

    public static float[] arg_float16_matrix(Matrix4f matrix)
    {
        return arg_float16(
            matrix.m00(), matrix.m01(), matrix.m02(), matrix.m03(),
            matrix.m10(), matrix.m11(), matrix.m12(), matrix.m13(),
            matrix.m20(), matrix.m21(), matrix.m22(), matrix.m23(),
            matrix.m30(), matrix.m31(), matrix.m32(), matrix.m33());
    }

    public static void k_call(cl_command_queue commandQueue, cl_kernel kernel, long[] global_work_size)
    {
        k_call(commandQueue, kernel, global_work_size, null, null);
    }

    public static void k_call(cl_command_queue commandQueue,
                              cl_kernel kernel,
                              long[] global_work_size,
                              long[] local_work_size)
    {

        k_call(commandQueue, kernel, global_work_size, local_work_size, null);
    }

    public static void k_call(cl_command_queue commandQueue,
                              cl_kernel kernel,
                              long[] global_work_size,
                              long[] local_work_size,
                              long[] global_work_offset)
    {
        int kernel_result = clEnqueueNDRangeKernel(commandQueue,
            kernel,
            1,
            global_work_offset,
            global_work_size,
            local_work_size,
            0,
            null,
            null);

        if (kernel_result != CL_SUCCESS)
        {
            System.out.println("WTF!");
        }
    }

    public static void gl_acquire(cl_command_queue commandQueue, cl_mem[] mem)
    {
        clEnqueueAcquireGLObjects(commandQueue, mem.length, mem, 0, null, null);
    }

    public static void gl_release(cl_command_queue commandQueue, cl_mem[] mem)
    {
        clEnqueueReleaseGLObjects(commandQueue, mem.length, mem, 0, null, null);
    }

    /**
     * Dump device data for debugging.
     *
     * @param devices
     */
    public static void debugDeviceDetails(cl_device_id[] devices)
    {
        for (cl_device_id device : devices)
        {
            // CL_DEVICE_NAME
            String deviceName = getString(device, CL_DEVICE_NAME);
            System.out.println("--- Info for device "+deviceName+": ---");
            System.out.printf("CL_DEVICE_NAME: \t\t\t%s\n", deviceName);

            // CL_DEVICE_VENDOR
            String deviceVendor = getString(device, CL_DEVICE_VENDOR);
            System.out.printf("CL_DEVICE_VENDOR: \t\t\t%s\n", deviceVendor);

            // CL_DRIVER_VERSION
            String driverVersion = getString(device, CL_DRIVER_VERSION);
            System.out.printf("CL_DRIVER_VERSION: \t\t\t%s\n", driverVersion);

            // CL_DEVICE_TYPE
            long deviceType = getLong(device, CL_DEVICE_TYPE);
            if( (deviceType & CL_DEVICE_TYPE_CPU) != 0)
                System.out.printf("CL_DEVICE_TYPE:\t\t\t\t%s\n", "CL_DEVICE_TYPE_CPU");
            if( (deviceType & CL_DEVICE_TYPE_GPU) != 0)
                System.out.printf("CL_DEVICE_TYPE:\t\t\t\t%s\n", "CL_DEVICE_TYPE_GPU");
            if( (deviceType & CL_DEVICE_TYPE_ACCELERATOR) != 0)
                System.out.printf("CL_DEVICE_TYPE:\t\t\t\t%s\n", "CL_DEVICE_TYPE_ACCELERATOR");
            if( (deviceType & CL_DEVICE_TYPE_DEFAULT) != 0)
                System.out.printf("CL_DEVICE_TYPE:\t\t\t\t%s\n", "CL_DEVICE_TYPE_DEFAULT");

            // CL_DEVICE_MAX_COMPUTE_UNITS
            int maxComputeUnits = getInt(device, CL_DEVICE_MAX_COMPUTE_UNITS);
            System.out.printf("CL_DEVICE_MAX_COMPUTE_UNITS:\t\t%d\n", maxComputeUnits);

            // CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS
            long maxWorkItemDimensions = getLong(device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS);
            System.out.printf("CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS:\t%d\n", maxWorkItemDimensions);

            // CL_DEVICE_MAX_WORK_ITEM_SIZES
            long[] maxWorkItemSizes = getSizes(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, 3);
            System.out.printf("CL_DEVICE_MAX_WORK_ITEM_SIZES:\t\t%d / %d / %d \n",
                maxWorkItemSizes[0], maxWorkItemSizes[1], maxWorkItemSizes[2]);

            // CL_DEVICE_MAX_WORK_GROUP_SIZE
            long maxWorkGroupSize = getSize(device, CL_DEVICE_MAX_WORK_GROUP_SIZE);
            System.out.printf("CL_DEVICE_MAX_WORK_GROUP_SIZE:\t\t%d\n", maxWorkGroupSize);

            // CL_DEVICE_MAX_CLOCK_FREQUENCY
            long maxClockFrequency = getLong(device, CL_DEVICE_MAX_CLOCK_FREQUENCY);
            System.out.printf("CL_DEVICE_MAX_CLOCK_FREQUENCY:\t\t%d MHz\n", maxClockFrequency);

            // CL_DEVICE_ADDRESS_BITS
            int addressBits = getInt(device, CL_DEVICE_ADDRESS_BITS);
            System.out.printf("CL_DEVICE_ADDRESS_BITS:\t\t\t%d\n", addressBits);

            // CL_DEVICE_MAX_MEM_ALLOC_SIZE
            long maxMemAllocSize = getLong(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE);
            System.out.printf("CL_DEVICE_MAX_MEM_ALLOC_SIZE:\t\t%d MByte\n", (int)(maxMemAllocSize / (1024 * 1024)));

            // CL_DEVICE_GLOBAL_MEM_SIZE
            long globalMemSize = getLong(device, CL_DEVICE_GLOBAL_MEM_SIZE);
            System.out.printf("CL_DEVICE_GLOBAL_MEM_SIZE:\t\t%d MByte\n", (int)(globalMemSize / (1024 * 1024)));

            // CL_DEVICE_ERROR_CORRECTION_SUPPORT
            int errorCorrectionSupport = getInt(device, CL_DEVICE_ERROR_CORRECTION_SUPPORT);
            System.out.printf("CL_DEVICE_ERROR_CORRECTION_SUPPORT:\t%s\n", errorCorrectionSupport != 0 ? "yes" : "no");

            // CL_DEVICE_LOCAL_MEM_TYPE
            int localMemType = getInt(device, CL_DEVICE_LOCAL_MEM_TYPE);
            System.out.printf("CL_DEVICE_LOCAL_MEM_TYPE:\t\t%s\n", localMemType == 1 ? "local" : "global");

            // CL_DEVICE_LOCAL_MEM_SIZE
            long localMemSize = getLong(device, CL_DEVICE_LOCAL_MEM_SIZE);
            System.out.printf("CL_DEVICE_LOCAL_MEM_SIZE:\t\t%d KByte\n", (int)(localMemSize / 1024));

            // CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE
            long maxConstantBufferSize = getLong(device, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE);
            System.out.printf("CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE:\t%d KByte\n", (int)(maxConstantBufferSize / 1024));

            // CL_DEVICE_QUEUE_PROPERTIES
            long queueProperties = getLong(device, CL_DEVICE_QUEUE_PROPERTIES);
            if(( queueProperties & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE ) != 0)
                System.out.printf("CL_DEVICE_QUEUE_PROPERTIES:\t\t%s\n", "CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE");
            if(( queueProperties & CL_QUEUE_PROFILING_ENABLE ) != 0)
                System.out.printf("CL_DEVICE_QUEUE_PROPERTIES:\t\t%s\n", "CL_QUEUE_PROFILING_ENABLE");

            // CL_DEVICE_IMAGE_SUPPORT
            int imageSupport = getInt(device, CL_DEVICE_IMAGE_SUPPORT);
            System.out.printf("CL_DEVICE_IMAGE_SUPPORT:\t\t%d\n", imageSupport);

            // CL_DEVICE_MAX_READ_IMAGE_ARGS
            int maxReadImageArgs = getInt(device, CL_DEVICE_MAX_READ_IMAGE_ARGS);
            System.out.printf("CL_DEVICE_MAX_READ_IMAGE_ARGS:\t\t%d\n", maxReadImageArgs);

            // CL_DEVICE_MAX_WRITE_IMAGE_ARGS
            int maxWriteImageArgs = getInt(device, CL_DEVICE_MAX_WRITE_IMAGE_ARGS);
            System.out.printf("CL_DEVICE_MAX_WRITE_IMAGE_ARGS:\t\t%d\n", maxWriteImageArgs);

            // CL_DEVICE_SINGLE_FP_CONFIG
            long singleFpConfig = getLong(device, CL_DEVICE_SINGLE_FP_CONFIG);
            System.out.printf("CL_DEVICE_SINGLE_FP_CONFIG:\t\t%s\n",
                stringFor_cl_device_fp_config(singleFpConfig));

            // CL_DEVICE_IMAGE2D_MAX_WIDTH
            long image2dMaxWidth = getSize(device, CL_DEVICE_IMAGE2D_MAX_WIDTH);
            System.out.printf("CL_DEVICE_2D_MAX_WIDTH\t\t\t%d\n", image2dMaxWidth);

            // CL_DEVICE_IMAGE2D_MAX_HEIGHT
            long image2dMaxHeight = getSize(device, CL_DEVICE_IMAGE2D_MAX_HEIGHT);
            System.out.printf("CL_DEVICE_2D_MAX_HEIGHT\t\t\t%d\n", image2dMaxHeight);

            // CL_DEVICE_IMAGE3D_MAX_WIDTH
            long image3dMaxWidth = getSize(device, CL_DEVICE_IMAGE3D_MAX_WIDTH);
            System.out.printf("CL_DEVICE_3D_MAX_WIDTH\t\t\t%d\n", image3dMaxWidth);

            // CL_DEVICE_IMAGE3D_MAX_HEIGHT
            long image3dMaxHeight = getSize(device, CL_DEVICE_IMAGE3D_MAX_HEIGHT);
            System.out.printf("CL_DEVICE_3D_MAX_HEIGHT\t\t\t%d\n", image3dMaxHeight);

            // CL_DEVICE_IMAGE3D_MAX_DEPTH
            long image3dMaxDepth = getSize(device, CL_DEVICE_IMAGE3D_MAX_DEPTH);
            System.out.printf("CL_DEVICE_3D_MAX_DEPTH\t\t\t%d\n", image3dMaxDepth);

            // CL_DEVICE_PREFERRED_VECTOR_WIDTH_<type>
            System.out.printf("CL_DEVICE_PREFERRED_VECTOR_WIDTH_<t>\t");
            int preferredVectorWidthChar = getInt(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR);
            int preferredVectorWidthShort = getInt(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT);
            int preferredVectorWidthInt = getInt(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT);
            int preferredVectorWidthLong = getInt(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG);
            int preferredVectorWidthFloat = getInt(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT);
            int preferredVectorWidthDouble = getInt(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE);
            System.out.printf("CHAR %d, SHORT %d, INT %d, LONG %d, FLOAT %d, DOUBLE %d\n\n\n",
                preferredVectorWidthChar, preferredVectorWidthShort,
                preferredVectorWidthInt, preferredVectorWidthLong,
                preferredVectorWidthFloat, preferredVectorWidthDouble);
        }
    }

    /**
     * Returns the value of the device info parameter with the given name
     *
     * @param device The device
     * @param paramName The parameter name
     * @return The value
     */
    private static int getInt(cl_device_id device, int paramName)
    {
        return getInts(device, paramName, 1)[0];
    }

    /**
     * Returns the values of the device info parameter with the given name
     *
     * @param device The device
     * @param paramName The parameter name
     * @param numValues The number of values
     * @return The value
     */
    private static int[] getInts(cl_device_id device, int paramName, int numValues)
    {
        int[] values = new int[numValues];
        clGetDeviceInfo(device, paramName, (long) Sizeof.cl_int * numValues, Pointer.to(values), null);
        return values;
    }

    /**
     * Returns the value of the device info parameter with the given name
     *
     * @param device The device
     * @param paramName The parameter name
     * @return The value
     */
    private static long getLong(cl_device_id device, int paramName)
    {
        return getLongs(device, paramName, 1)[0];
    }

    /**
     * Returns the values of the device info parameter with the given name
     *
     * @param device The device
     * @param paramName The parameter name
     * @param numValues The number of values
     * @return The value
     */
    private static long[] getLongs(cl_device_id device, int paramName, int numValues)
    {
        long[] values = new long[numValues];
        clGetDeviceInfo(device, paramName, (long) Sizeof.cl_long * numValues, Pointer.to(values), null);
        return values;
    }

    /**
     * Returns the value of the device info parameter with the given name
     *
     * @param device The device
     * @param paramName The parameter name
     * @return The value
     */
    public static String getString(cl_device_id device, int paramName)
    {
        // Obtain the length of the string that will be queried
        long[] size = new long[1];
        clGetDeviceInfo(device, paramName, 0, null, size);

        // Create a buffer of the appropriate size and fill it with the info
        byte[] buffer = new byte[(int)size[0]];
        clGetDeviceInfo(device, paramName, buffer.length, Pointer.to(buffer), null);

        // Create a string p1 the buffer (excluding the trailing \0 byte)
        return new String(buffer, 0, buffer.length-1);
    }

    /**
     * Returns the value of the device info parameter with the given name
     *
     * @param device The device
     * @param paramName The parameter name
     * @return The value
     */
    public static long getSize(cl_device_id device, int paramName)
    {
        return getSizes(device, paramName, 1)[0];
    }

    /**
     * Returns the values of the device info parameter with the given name
     *
     * @param device The device
     * @param paramName The parameter name
     * @param numValues The number of values
     * @return The value
     */
    static long[] getSizes(cl_device_id device, int paramName, int numValues)
    {
        // The size of the returned data has p2 depend on
        // the size of a size_t, which is handled here
        ByteBuffer buffer = ByteBuffer.allocate(
            numValues * Sizeof.size_t).order(ByteOrder.nativeOrder());
        clGetDeviceInfo(device, paramName, (long) Sizeof.size_t * numValues,
            Pointer.to(buffer), null);
        long[] values = new long[numValues];
        if (Sizeof.size_t == 4)
        {
            for (int i=0; i<numValues; i++)
            {
                values[i] = buffer.getInt(i * Sizeof.size_t);
            }
        }
        else
        {
            for (int i=0; i<numValues; i++)
            {
                values[i] = buffer.getLong(i * Sizeof.size_t);
            }
        }
        return values;
    }
}
