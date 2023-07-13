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

    static cl_kernel k_scan_key_bank;
    static cl_program p_scan_key_bank;
    private static final String src_scan_key_bank = readSrc("scan_key_bank.cl");

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

        p_scan_key_bank = clCreateProgramWithSource(context, 1, new String[]{src_scan_key_bank}, null, null);
        clBuildProgram(p_scan_key_bank, 1, device_id, null, null, null);
        k_scan_key_bank = clCreateKernel(p_scan_key_bank, "scan_key_bank", null);
    }

    public static void destroy()
    {
        clReleaseKernel(k_verletIntegrate);
        clReleaseKernel(k_collide);
        clReleaseKernel(k_scan_key_bank);
        clReleaseProgram(p_verletIntegrate);
        clReleaseProgram(p_collide);
        clReleaseProgram(p_scan_key_bank);
        clReleaseCommandQueue(commandQueue);
        clReleaseContext(context);
    }

    private static int wx = 256;
    private static int m = wx * 2;

    public static void scan_key_bank()
    {
        int boundsSize = Main.Memory.boundsLength();
        var input = FloatBuffer.wrap(Main.Memory.bounds_buffer, 0, boundsSize);

        int n = Main.Memory.boundsCount();
        int k = (int) Math.ceil((float)n / (float)m);
        cl_mem d_data;
        long sz = ((long)Sizeof.cl_float16 * (long)k * (long)m);

        long flags = CL.CL_MEM_READ_WRITE | CL.CL_MEM_COPY_HOST_PTR;
        d_data = CL.clCreateBuffer(context, flags, sz, Pointer.to(input), null);
        //out_data = CL.clCreateBuffer(context, flags, sz2, Pointer.to(output), null);

        long data_buf_size = (long)Sizeof.cl_float16 * n;
        Pointer dst_data = Pointer.to(input);
        scan(d_data, /*out_data,*/ n, k);
        //System.out.println("t2e: " + (System.currentTimeMillis() - start));
       // start = System.currentTimeMillis();
        // transfer results into local memory.
        // Note: this is only needed once, after all operations are done
        clEnqueueReadBuffer(commandQueue, d_data, CL_TRUE, 0,
            data_buf_size, dst_data, 0, null, null);
        clReleaseMemObject(d_data);
    }


    private static void scan(cl_mem d_data, /*cl_mem out_data,*/ int n, int k)
    {
        if (k == 1)
        {
            scan_single_block(d_data, /*out_data,*/ n);
        }
        else
        {
            //scan_multi_block(d_data, n, k);
        }
    }

    private static void scan_single_block(cl_mem d_data, /*cl_mem out_data,*/ int n)
    {
        // set up buffers
        int localBufferSize = Sizeof.cl_int * m;
        Pointer src_data = Pointer.to(d_data);
        //Pointer dst_data = Pointer.to(out_data);

        // pass in arguments
        clSetKernelArg(k_scan_key_bank, 0, Sizeof.cl_mem, src_data);
        clSetKernelArg(k_scan_key_bank, 1, localBufferSize,null);
        //clSetKernelArg(k_scan_key_bank, 2, Sizeof.cl_mem, dst_data);
        clSetKernelArg(k_scan_key_bank, 2, Sizeof.cl_int, Pointer.to(new int[]{n}));

        // call kernel
        clEnqueueNDRangeKernel(commandQueue, k_scan_key_bank, 1, null,
            new long[]{wx}, new long[]{wx}, 0, null, null);
    }

//    private static void scan_multi_block(cl_mem d_data, int n, int k)
//    {
//        // set up buffers
//        int localBufferSize = Sizeof.cl_int * m;
//        int gx = k * m;
//        long part_buf_size = ((long)Sizeof.cl_int * ((long)k * 2));
//        int[] partial_sums = new int[k * 2];
//        cl_mem p_data = CL.clCreateBuffer(context, CL.CL_MEM_READ_WRITE | CL.CL_MEM_COPY_HOST_PTR, part_buf_size, Pointer.to(partial_sums), null);
//        Pointer src_data = Pointer.to(d_data);
//        Pointer src_part = Pointer.to(p_data);
//
//        // pass in arguments
//        clSetKernelArg(k_scan_multi_block, 0, Sizeof.cl_mem, src_data);
//        clSetKernelArg(k_scan_multi_block, 1, localBufferSize,null);
//        clSetKernelArg(k_scan_multi_block, 2, Sizeof.cl_mem, src_part);
//        clSetKernelArg(k_scan_multi_block, 3, Sizeof.cl_int, Pointer.to(new int[]{n}));
//
//        // call kernel
//        clEnqueueNDRangeKernel(commandQueue, k_scan_multi_block, 1, null,
//            new long[]{gx}, new long[]{wx}, 0, null, null);
//
//        // do scan on block sums
//        int n2 = partial_sums.length;
//        int k2 = (int) Math.ceil((float)n2 / (float)m);
//        scan(p_data, n2, k2);
//
//        // pass in arguments
//        clSetKernelArg(k_complete_multi_block, 0, Sizeof.cl_mem, src_data);
//        clSetKernelArg(k_complete_multi_block, 1, localBufferSize,null);
//        clSetKernelArg(k_complete_multi_block, 2, Sizeof.cl_mem, src_part);
//        clSetKernelArg(k_complete_multi_block, 3, Sizeof.cl_int, Pointer.to(new int[]{n}));
//
//        // call kernel
//        clEnqueueNDRangeKernel(commandQueue, k_complete_multi_block, 1, null,
//            new long[]{gx}, new long[]{wx}, 0, null, null);
//
//        clReleaseMemObject(p_data);
//    }






    public static void integrate(float tick_rate, SpatialPartition spatialPartition)
    {
        int bodiesSize = Main.Memory.bodyLength();
        int pointsSize = Main.Memory.pointLength();
        int boundsSize = Main.Memory.boundsLength();

        var bodyBuffer = FloatBuffer.wrap(Main.Memory.body_buffer, 0, bodiesSize);
        var pointBuffer = FloatBuffer.wrap(Main.Memory.point_buffer, 0, pointsSize);
        var boundsBuffer = FloatBuffer.wrap(Main.Memory.bounds_buffer, 0, boundsSize);

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
        cl_mem srcMemBodies = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, bodyBufsize, srcBodies, null);
        cl_mem srcMemPoints = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, pointBufsize, srcPoints, null);
        cl_mem srcMemBounds = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, boundsBufsize, srcBounds, null);

        cl_mem dtMem = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, Sizeof.cl_float * args.length, srcDt, null);

        // Set the arguments for the kernel
        int a = 0;
        clSetKernelArg(k_verletIntegrate, a++, Sizeof.cl_mem, Pointer.to(srcMemBodies));
        clSetKernelArg(k_verletIntegrate, a++, Sizeof.cl_mem, Pointer.to(srcMemPoints));
        clSetKernelArg(k_verletIntegrate, a++, Sizeof.cl_mem, Pointer.to(srcMemBounds));
        clSetKernelArg(k_verletIntegrate, a++, Sizeof.cl_mem, Pointer.to(dtMem));

        // Execute the kernel
        clEnqueueNDRangeKernel(commandQueue, k_verletIntegrate, 1, null,
            global_work_size, null, 0, null, null);

        // Read the output data
        clEnqueueReadBuffer(commandQueue, srcMemBodies, CL_TRUE, 0,
            bodyBufsize, srcBodies, 0, null, null);

        clEnqueueReadBuffer(commandQueue, srcMemPoints, CL_TRUE, 0,
            pointBufsize, srcPoints, 0, null, null);

        clEnqueueReadBuffer(commandQueue, srcMemBounds, CL_TRUE, 0,
            boundsBufsize, srcBounds, 0, null, null);

        clReleaseMemObject(srcMemBodies);
        clReleaseMemObject(srcMemPoints);
        clReleaseMemObject(srcMemBounds);
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
