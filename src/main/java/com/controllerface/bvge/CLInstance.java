package com.controllerface.bvge;

import org.jocl.*;

import java.nio.FloatBuffer;

import static org.jocl.CL.*;
import static org.jocl.CL.clReleaseContext;

public class CLInstance
{

    static cl_command_queue commandQueue;
    static cl_context context;

    static cl_kernel kernel;
    static cl_program program;

    static cl_kernel kernel2;
    static cl_program program2;


    private static String vectorDotproduct =
        "__kernel void "+
            "vectorDotProduct(__global const float2 *a,"+
            "             __global const float2 *b,"+
            "             __global float *c)"+
            "{"+
            "    int gid = get_global_id(0);"+
            "    c[gid] = (float)dot(a[gid], b[gid]);"+
            "}";

    private static String vectorDistance =
        "__kernel void "+
            "vectorDistance(__global const float2 *a,"+
            "             __global const float2 *b,"+
            "             __global float *c)"+
            "{"+
            "    int gid = get_global_id(0);"+
            "    c[gid] = (float)distance(a[gid], b[gid]);"+
            "}";

    private static cl_device_id[] commonInit()
    {
        // The platform, device type and device number
        // that will be used
        final int platformIndex = 0;
        final long deviceType = CL_DEVICE_TYPE_ALL;
        final int deviceIndex = 0;

        // Enable exceptions and subsequently omit error checks in this sample
        CL.setExceptionsEnabled(true);

        // Obtain the number of platforms
        int numPlatformsArray[] = new int[1];
        clGetPlatformIDs(0, null, numPlatformsArray);
        int numPlatforms = numPlatformsArray[0];

        // Obtain a platform ID
        cl_platform_id platforms[] = new cl_platform_id[numPlatforms];
        clGetPlatformIDs(platforms.length, platforms, null);
        cl_platform_id platform = platforms[platformIndex];

        // Initialize the context properties
        cl_context_properties contextProperties = new cl_context_properties();
        contextProperties.addProperty(CL_CONTEXT_PLATFORM, platform);

        // Obtain the number of devices for the platform
        int numDevicesArray[] = new int[1];
        clGetDeviceIDs(platform, deviceType, 0, null, numDevicesArray);
        int numDevices = numDevicesArray[0];

        // Obtain a device ID
        cl_device_id devices[] = new cl_device_id[numDevices];
        clGetDeviceIDs(platform, deviceType, numDevices, devices, null);
        cl_device_id device = devices[deviceIndex];

        var x = new cl_device_id[]{device};
        // Create a context for the selected device
        context = clCreateContext(
            contextProperties, 1, x,
            null, null, null);

        // Create a command-queue for the selected device
        cl_queue_properties properties = new cl_queue_properties();
        commandQueue = clCreateCommandQueueWithProperties(
            context, device, properties, null);

        return x;

    }

    public static void init()
    {
        var x = commonInit();

        program = clCreateProgramWithSource(context, 1, new String[]{vectorDistance}, null, null);
        clBuildProgram(program, 1, x, null, null, null);
        kernel = clCreateKernel(program, "vectorDistance", null);

        program2 = clCreateProgramWithSource(context, 1, new String[]{vectorDotproduct}, null, null);
        clBuildProgram(program2, 1, x, null, null, null);
        kernel2 = clCreateKernel(program2, "vectorDotProduct", null);
    }

    public static void destroy()
    {
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseKernel(kernel2);
        clReleaseProgram(program2);
        clReleaseCommandQueue(commandQueue);
        clReleaseContext(context);
    }

    private static void vectorScalarFunction(FloatBuffer srcArrayA,
                                             FloatBuffer srcArrayB,
                                             FloatBuffer dstArray,
                                             cl_kernel kernel)
    {
        int n = srcArrayA.limit();
        assert n % 2 == 0 : "Invalid length";
        // Set the work-item dimensions
        long global_work_size[] = new long[]{n};

        Pointer srcA = Pointer.to(srcArrayA);
        Pointer srcB = Pointer.to(srcArrayB);
        Pointer dst = Pointer.to(dstArray);

        // Allocate the memory objects for the input- and output data
        cl_mem srcMemA = clCreateBuffer(context,
            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            Sizeof.cl_float * n, srcA, null);

        cl_mem srcMemB = clCreateBuffer(context,
            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            Sizeof.cl_float * n, srcB, null);

        cl_mem dstMem = clCreateBuffer(context,
            CL_MEM_READ_WRITE,
            Sizeof.cl_float * (n / 2), null, null);

        // Set the arguments for the kernel
        int a = 0;
        clSetKernelArg(kernel, a++, Sizeof.cl_mem, Pointer.to(srcMemA));
        clSetKernelArg(kernel, a++, Sizeof.cl_mem, Pointer.to(srcMemB));
        clSetKernelArg(kernel, a++, Sizeof.cl_mem, Pointer.to(dstMem));

        // Execute the kernel
        clEnqueueNDRangeKernel(commandQueue, kernel, 1, null,
            global_work_size, null, 0, null, null);

        // Read the output data
        clEnqueueReadBuffer(commandQueue, dstMem, CL_TRUE, 0,
            (n / 2) * Sizeof.cl_float, dst, 0, null, null);

        clReleaseMemObject(srcMemA);
        clReleaseMemObject(srcMemB);
        clReleaseMemObject(dstMem);
    }

    public static void vectorDistance(float[] srcArrayA, float[] srcArrayB, float[] dstArray)
    {
        vectorScalarFunction(FloatBuffer.wrap(srcArrayA),
            FloatBuffer.wrap(srcArrayB),
            FloatBuffer.wrap(dstArray),
            kernel);
    }

    public static void vectorDotProduct(float[] srcArrayA, float[] srcArrayB, float[] dstArray)
    {
        vectorScalarFunction(FloatBuffer.wrap(srcArrayA),
            FloatBuffer.wrap(srcArrayB),
            FloatBuffer.wrap(dstArray),
            kernel2);
    }

    public static void vectorDotProduct(FloatBuffer srcArrayA, FloatBuffer srcArrayB, FloatBuffer dstArray)
    {
        vectorScalarFunction(srcArrayA, srcArrayB, dstArray, kernel2);
    }
}
