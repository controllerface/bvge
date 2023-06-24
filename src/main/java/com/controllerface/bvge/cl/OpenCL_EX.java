package com.controllerface.bvge.cl;

import org.jocl.*;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.charset.StandardCharsets;

import static org.jocl.CL.*;

public class OpenCL_EX
{
    static cl_command_queue commandQueue;
    static cl_context context;

    static cl_kernel k_verletIntegrate;
    static cl_program p_verletIntegrate;
    private static String src_verletIntegrate = readSrc("integrate.cl");

    private static String readSrc(String file)
    {
        var stream = OpenCL_EX.class.getResourceAsStream("/kernels/" + file);
        try
        {
            byte [] bytes = stream.readAllBytes();
            return new String(bytes, StandardCharsets.UTF_8);
        }
        catch (IOException e)
        {
            throw new RuntimeException(e);
        }
    }

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

        var device_ids = new cl_device_id[]{device};
//        OpenCL.printDeviceDetails(device_ids);
        // Create a context for the selected device
        context = clCreateContext(
            contextProperties, 1, device_ids,
            null, null, null);

        // Create a command-queue for the selected device
        cl_queue_properties properties = new cl_queue_properties();
        commandQueue = clCreateCommandQueueWithProperties(
            context, device, properties, null);

        return device_ids;

    }

    public static void init()
    {
        var x = commonInit();
        p_verletIntegrate = clCreateProgramWithSource(context, 1, new String[]{src_verletIntegrate}, null, null);
        clBuildProgram(p_verletIntegrate, 1, x, null, null, null);
        k_verletIntegrate = clCreateKernel(p_verletIntegrate, "integrate", null);
    }

    public static void destroy()
    {
        clReleaseKernel(k_verletIntegrate);
        clReleaseProgram(p_verletIntegrate);
        clReleaseCommandQueue(commandQueue);
        clReleaseContext(context);
    }

    public static void integrate(float[] bodies, float[] points)
    {
        var bBuf = FloatBuffer.wrap(bodies);
        var pBuf = FloatBuffer.wrap(points);

        int bSize = bodies.length;
        int pSize = points.length;
        //assert n % 2 == 0 : "Invalid length";
        // Set the work-item dimensions
        long global_work_size[] = new long[]{bSize / 16};
        float tick_rate = (1f / 60f);
        float[] dt = { tick_rate * tick_rate };

        Pointer srcB = Pointer.to(bBuf);
        Pointer srcP = Pointer.to(pBuf);
        Pointer srcDt = Pointer.to(FloatBuffer.wrap(dt));


        long bBufsize = Sizeof.cl_float * bSize;
        long pBufsize = Sizeof.cl_float * pSize;

        // Allocate the memory objects for the input- and output data
        // Note that the src B/P and dest B/P buffers will effectively be the same as the data is transferred
        // directly p2 thr destination p1 the result fo the kernel call. This avoids
        // needing p2 use an intermediate buffer and System.arrayCopy() calls.
        cl_mem srcMemB = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, bBufsize, srcB, null);
        cl_mem srcMemP = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, pBufsize, srcP, null);
        cl_mem dstMemB = clCreateBuffer(context, CL_MEM_READ_WRITE, bBufsize, null, null);
        cl_mem dstMemP = clCreateBuffer(context, CL_MEM_READ_WRITE, bBufsize, null, null);
        cl_mem dtMem = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, Sizeof.cl_float, srcDt, null);

        // Set the arguments for the kernel
        int a = 0;

        clSetKernelArg(k_verletIntegrate, a++, Sizeof.cl_mem, Pointer.to(srcMemB));
        clSetKernelArg(k_verletIntegrate, a++, Sizeof.cl_mem, Pointer.to(srcMemP));
        clSetKernelArg(k_verletIntegrate, a++, Sizeof.cl_mem, Pointer.to(dstMemB));
        clSetKernelArg(k_verletIntegrate, a++, Sizeof.cl_mem, Pointer.to(dstMemP));
        clSetKernelArg(k_verletIntegrate, a++, Sizeof.cl_mem, Pointer.to(dtMem));

        // Execute the kernel
        clEnqueueNDRangeKernel(commandQueue, k_verletIntegrate, 1, null,
            global_work_size, null, 0, null, null);

        // Read the output data
        clEnqueueReadBuffer(commandQueue, dstMemB, CL_TRUE, 0,
            bBufsize, srcB, 0, null, null);

        clEnqueueReadBuffer(commandQueue, dstMemP, CL_TRUE, 0,
            pBufsize, srcP, 0, null, null);

        clReleaseMemObject(srcMemB);
        clReleaseMemObject(srcMemP);
        clReleaseMemObject(dtMem);
    }
}
