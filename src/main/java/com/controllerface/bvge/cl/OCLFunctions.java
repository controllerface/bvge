package com.controllerface.bvge.cl;

import com.controllerface.bvge.Main;
import com.controllerface.bvge.ecs.systems.physics.SpatialPartition;
import org.jocl.*;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
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

    static cl_kernel exclusiveScanKernel;
    static cl_program p_test;
    private static final String src_test = readSrc("test_2.cl");

    static cl_kernel addOffsetKernel;
    static cl_program p_test2;
    private static final String src_test2 = readSrc("test_add_offset.cl");

    public static String readSrc(String file)
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

        p_test = clCreateProgramWithSource(context, 1, new String[]{src_test}, null, null);
        clBuildProgram(p_test, 1, device_id, null, null, null);
        exclusiveScanKernel = clCreateKernel(p_test, "scan", null);

        p_test2 = clCreateProgramWithSource(context, 1, new String[]{src_test2}, null, null);
        clBuildProgram(p_test2, 1, device_id, null, null, null);
        addOffsetKernel = clCreateKernel(p_test2, "addOffset", null);

    }

    public static void destroy()
    {
        clReleaseKernel(k_verletIntegrate);
        clReleaseProgram(p_verletIntegrate);
        clReleaseCommandQueue(commandQueue);
        clReleaseContext(context);
    }



    public static final int wx = 256;
    public static final int m = wx * 2;

    public static void scan(float[] inputData, float[] outputData, int size)
    {
        cl_mem inputBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            Sizeof.cl_float * size, Pointer.to(inputData), null);
        cl_mem outputBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
            Sizeof.cl_float * size, null, null);

        // Set the kernel arguments
        clSetKernelArg(exclusiveScanKernel, 0, Sizeof.cl_mem, Pointer.to(inputBuffer));
        clSetKernelArg(exclusiveScanKernel, 1, Sizeof.cl_mem, Pointer.to(outputBuffer));
        clSetKernelArg(exclusiveScanKernel, 2, Sizeof.cl_int, Pointer.to(new int[]{size}));

        // Enqueue the kernel for execution
        clEnqueueNDRangeKernel(commandQueue, exclusiveScanKernel, 1, null, new long[]{size}, null, 0, null, null);

        // Read the output buffer to retrieve the result
        clEnqueueReadBuffer(commandQueue, outputBuffer, CL_TRUE, 0, Sizeof.cl_float * size,
            Pointer.to(outputData), 0, null, null);

        // Print the results
        System.out.println("Input: " + java.util.Arrays.toString(inputData));
        System.out.println("GPU Output: " + java.util.Arrays.toString(outputData));

    }


//    public static void scan(int[] data, int[] flag, int n)
//    {
//        int k = (int) Math.ceil((float)n/(float)m);
//
//        var inputBuffer = IntBuffer.wrap(data);
//        var flagBuffer = IntBuffer.wrap(flag);
//        Pointer srcInput = Pointer.to(inputBuffer);
//        Pointer srcFlag = Pointer.to(flagBuffer);
//        long inputBufsize = (long)Sizeof.cl_int * k * m;
//
//        cl_mem d_data = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, inputBufsize, srcInput, null);
//        //clw.dev_malloc(sizeof(int)*k*m);
//        cl_mem d_part = clw.dev_malloc(sizeof(int)*k*m);
//        cl_mem d_flag = clw.dev_malloc(sizeof(int)*k*m);
//
//        m0 += clw.memcpy_to_dev(d_data, sizeof(int)*n, data);
//        m1 += clw.memcpy_to_dev(d_part, sizeof(int)*n, flag);
//        m2 += clw.memcpy_to_dev(d_flag, sizeof(int)*n, flag);
//
//        recursive_scan(d_data, d_part, d_flag, n);
//
//        m3 += clw.memcpy_from_dev(d_data, sizeof(int)*n, data);
//
//        clw.dev_free(d_data);
//        clw.dev_free(d_part);
//        clw.dev_free(d_flag);
//    }


//    public static void recursive_scan(cl_mem d_data, cl_mem d_part, cl_mem d_flag, int n)
//    {
//        int k = (int) Math.ceil((float)n/(float)m);
//        //size of each subarray stored in local memory
//        var bufsize = Sizeof.cl_int + m;//sizeof(int)*m;
//        if (k == 1) {
//            clw.kernel_arg(scan_pad_to_pow2,
//                d_data,  d_part,  d_flag,
//                bufsize, bufsize, bufsize,
//                n);
//            k0 += clw.run_kernel_with_timing(scan_pad_to_pow2, /*dim=*/1, &wx, &wx);
//
//        } else {
//            size_t gx = k * wx;
//            cl_mem d_data2 = clw.dev_malloc(sizeof(int)*k);
//            cl_mem d_part2 = clw.dev_malloc(sizeof(int)*k);
//            cl_mem d_flag2 = clw.dev_malloc(sizeof(int)*k);
//            clw.kernel_arg(upsweep_subarrays,
//                d_data,  d_part,  d_flag,
//                d_data2, d_part2, d_flag2,
//                bufsize, bufsize, bufsize,
//                n);
//            k1 += clw.run_kernel_with_timing(upsweep_subarrays, /*dim=*/1, &gx, &wx);
//
//            recursive_scan(d_data2, d_part2, d_flag2, k);
//
//            clw.kernel_arg(downsweep_subarrays,
//                d_data,  d_part,  d_flag,
//                d_data2, d_part2, d_flag2,
//                bufsize, bufsize, bufsize,
//                n);
//            k2 += clw.run_kernel_with_timing(downsweep_subarrays, /*dim=*/1, &gx, &wx);
//
//            clw.dev_free(d_data2);
//            clw.dev_free(d_part2);
//            clw.dev_free(d_flag2);
//        }
//    }


    public static void integrate(float tick_rate, SpatialPartition spatialPartition)
    {
        int bodiesSize = Main.Memory.bodyLength();
        int pointsSize = Main.Memory.pointLength();
        int boundsSize = Main.Memory.boundsLength();

        var bodyBuffer = FloatBuffer.wrap(Main.Memory.body_buffer, 0, bodiesSize);
        var pointBuffer = FloatBuffer.wrap(Main.Memory.point_buffer, 0, pointsSize);
        var boundsBuffer = FloatBuffer.wrap(Main.Memory.bounds_buffer, 0, boundsSize);



//        float x_spacing, float y_spacing,
//        float x_origin, float y_origin,
//        float width, float height,
//        int x_subdivisions, int y_subdivisions


        // Set the work-item dimensions
        long global_work_size[] = new long[]{Main.Memory.bodyCount()};
        float[] args =
        {
            tick_rate * tick_rate,
            spatialPartition.getX_spacing(),
            spatialPartition.getY_spacing(),
            spatialPartition.getX_origin(),
            spatialPartition.getY_origin(),
            spatialPartition.getWidth(),
            spatialPartition.getHeight(),
            (float)spatialPartition.getX_subdivisions(),
            (float)spatialPartition.getY_subdivisions()
        };

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
