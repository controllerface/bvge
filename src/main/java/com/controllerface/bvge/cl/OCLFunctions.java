package com.controllerface.bvge.cl;

import com.controllerface.bvge.Main;
import org.jocl.*;

import java.io.IOException;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.charset.StandardCharsets;

import static org.jocl.CL.*;

public class OCLFunctions
{
    static cl_command_queue commandQueue;
    static cl_context context;

    static cl_kernel k_verletIntegrate;
    static cl_program p_verletIntegrate;
    private static final String src_verletIntegrate = readSrc("integrate.cl");

    static cl_kernel k_collide;
    static cl_program p_collide;
    private static final String src_collide = readSrc("collide.cl");

    private static String readSrc(String file)
    {
        var stream = OCLFunctions.class.getResourceAsStream("/kernels/" + file);
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
        var device_id = commonInit();

        p_verletIntegrate = clCreateProgramWithSource(context, 1, new String[]{src_verletIntegrate}, null, null);
        clBuildProgram(p_verletIntegrate, 1, device_id, null, null, null);
        k_verletIntegrate = clCreateKernel(p_verletIntegrate, "integrate", null);

        p_collide = clCreateProgramWithSource(context, 1, new String[]{src_collide}, null, null);
        clBuildProgram(p_collide, 1, device_id, null, null, null);
        k_collide = clCreateKernel(p_collide, "collide", null);
    }

    public static void destroy()
    {
        clReleaseKernel(k_verletIntegrate);
        clReleaseProgram(p_verletIntegrate);
        clReleaseCommandQueue(commandQueue);
        clReleaseContext(context);
    }

    public static void integrate(float tick_rate, float x_spacing, float y_spacing)
    {
        int bodiesSize = Main.Memory.bodyLength();
        int pointsSize = Main.Memory.pointLength();
        int boundsSize = Main.Memory.boundsLength();

        var bodyBuffer = FloatBuffer.wrap(Main.Memory.body_buffer, 0, bodiesSize);
        var pointBuffer = FloatBuffer.wrap(Main.Memory.point_buffer, 0, pointsSize);
        var boundsBuffer = FloatBuffer.wrap(Main.Memory.bounds_buffer, 0, boundsSize);

        // Set the work-item dimensions
        long global_work_size[] = new long[]{Main.Memory.bodyCount()};
        float[] args = { tick_rate * tick_rate, x_spacing, y_spacing };

        Pointer srcBodies = Pointer.to(bodyBuffer);
        Pointer srcPoints = Pointer.to(pointBuffer);
        Pointer srcBounds = Pointer.to(boundsBuffer);
        Pointer srcDt = Pointer.to(FloatBuffer.wrap(args));

        long bodyBufsize = (long)Sizeof.cl_float * bodiesSize;
        long pointBufsize = (long)Sizeof.cl_float * pointsSize;
        long boundsBufsize = (long)Sizeof.cl_float * boundsSize;

        // Allocate the memory objects for the input- and output data
        // Note that the src B/P and dest B/P buffers will effectively be the same as the data is transferred
        // directly p2 thr destination p1 the result fo the kernel call. This avoids
        // needing p2 use an intermediate buffer and System.arrayCopy() calls.
        cl_mem srcMemBodies = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, bodyBufsize, srcBodies, null);
        cl_mem srcMemPoints = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, pointBufsize, srcPoints, null);
        cl_mem srcMemBounds = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, boundsBufsize, srcBounds, null);

        cl_mem dstMemBodies = clCreateBuffer(context, CL_MEM_READ_WRITE, bodyBufsize, null, null);
        cl_mem dstMemPoints = clCreateBuffer(context, CL_MEM_READ_WRITE, pointBufsize, null, null);
        cl_mem dstMemBounds = clCreateBuffer(context, CL_MEM_READ_WRITE, boundsBufsize, null, null);

        cl_mem dtMem = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, Sizeof.cl_float * args.length, srcDt, null);

        // Set the arguments for the kernel
        int a = 0;
        clSetKernelArg(k_verletIntegrate, a++, Sizeof.cl_mem, Pointer.to(srcMemBodies));
        clSetKernelArg(k_verletIntegrate, a++, Sizeof.cl_mem, Pointer.to(srcMemPoints));
        clSetKernelArg(k_verletIntegrate, a++, Sizeof.cl_mem, Pointer.to(srcMemBounds));
        clSetKernelArg(k_verletIntegrate, a++, Sizeof.cl_mem, Pointer.to(dstMemBodies));
        clSetKernelArg(k_verletIntegrate, a++, Sizeof.cl_mem, Pointer.to(dstMemPoints));
        clSetKernelArg(k_verletIntegrate, a++, Sizeof.cl_mem, Pointer.to(dstMemBounds));
        clSetKernelArg(k_verletIntegrate, a++, Sizeof.cl_mem, Pointer.to(dtMem));

        // Execute the kernel
        clEnqueueNDRangeKernel(commandQueue, k_verletIntegrate, 1, null,
            global_work_size, null, 0, null, null);

        // Read the output data
        clEnqueueReadBuffer(commandQueue, dstMemBodies, CL_TRUE, 0,
            bodyBufsize, srcBodies, 0, null, null);

        clEnqueueReadBuffer(commandQueue, dstMemPoints, CL_TRUE, 0,
            pointBufsize, srcPoints, 0, null, null);

        clEnqueueReadBuffer(commandQueue, dstMemBounds, CL_TRUE, 0,
            boundsBufsize, srcBounds, 0, null, null);

        clReleaseMemObject(srcMemBodies);
        clReleaseMemObject(srcMemPoints);
        clReleaseMemObject(srcMemBounds);
        clReleaseMemObject(dstMemBodies);
        clReleaseMemObject(dstMemPoints);
        clReleaseMemObject(dstMemBounds);
        clReleaseMemObject(dtMem);
    }

    public static void collide(IntBuffer candidates, FloatBuffer reactions)
    {
        int candidatesSize = candidates.limit();
        int bodiesSize = Main.Memory.bodyLength();
        int pointsSize = Main.Memory.pointLength();
        int reactionsSize = reactions.limit();

        var bodyBuffer = FloatBuffer.wrap(Main.Memory.body_buffer, 0, bodiesSize);
        var pointBuffer = FloatBuffer.wrap(Main.Memory.point_buffer, 0, pointsSize);

        // Set the work-item dimensions
        long global_work_size[] = new long[]{candidates.limit() / Main.Memory.Width.COLLISION};

        Pointer srcCandidates = Pointer.to(candidates);
        Pointer srcBodies = Pointer.to(bodyBuffer);
        Pointer srcPoints = Pointer.to(pointBuffer);
        Pointer dstReactions = Pointer.to(reactions);

        long candidateBufSize = Sizeof.cl_int * candidatesSize;
        long bodyBufsize = Sizeof.cl_float * bodiesSize;
        long pointBufsize = Sizeof.cl_float * pointsSize;
        long reactionBufsize = Sizeof.cl_float * reactionsSize;

        cl_mem srcMemCandidates = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, candidateBufSize, srcCandidates, null);
        cl_mem srcMemBodies = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, bodyBufsize, srcBodies, null);
        cl_mem srcMemPoints = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, pointBufsize, srcPoints, null);
        cl_mem dstMemReactions = clCreateBuffer(context, CL_MEM_READ_WRITE, reactionBufsize, null, null);


        // Set the arguments for the kernel
        int a = 0;
        clSetKernelArg(k_collide, a++, Sizeof.cl_mem, Pointer.to(srcMemCandidates));
        clSetKernelArg(k_collide, a++, Sizeof.cl_mem, Pointer.to(srcMemBodies));
        clSetKernelArg(k_collide, a++, Sizeof.cl_mem, Pointer.to(srcMemPoints));
        clSetKernelArg(k_collide, a++, Sizeof.cl_mem, Pointer.to(dstMemReactions));

        // Execute the kernel
        clEnqueueNDRangeKernel(commandQueue, k_collide, 1, null,
                global_work_size, null, 0, null, null);

        // Read the output data
        clEnqueueReadBuffer(commandQueue, dstMemReactions, CL_TRUE, 0,
                reactionBufsize, dstReactions, 0, null, null);

        clReleaseMemObject(srcMemCandidates);
        clReleaseMemObject(srcMemBodies);
        clReleaseMemObject(srcMemPoints);
        clReleaseMemObject(dstMemReactions);
    }
}
