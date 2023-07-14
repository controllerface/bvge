package com.controllerface.bvge.cl;

import com.controllerface.bvge.Main;
import com.controllerface.bvge.ecs.systems.physics.SpatialPartition;
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

    static cl_kernel k_scan_key_bank;
    static cl_kernel k_scan_key_bank_block;
    static cl_kernel k_finish_key_bank_block;
    static cl_kernel k_scan_single_block;
    static cl_kernel k_scan_multi_block;
    static cl_kernel k_complete_multi_block;
    static cl_program p_scan_key_bank;
    private static final String src_scan_key_bank = readSrc("scan_key_bank.cl");

    static cl_kernel k_generate_keys;
    static cl_program p_generate_keys;
    private static final String src_generate_keys = readSrc("generate_keys.cl");

    static cl_kernel k_build_key_map;
    static cl_program p_build_key_map;
    private static final String src_build_key_map = readSrc("build_key_map.cl");

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
        k_scan_key_bank_block = clCreateKernel(p_scan_key_bank, "scan_key_bank_block", null);
        k_finish_key_bank_block = clCreateKernel(p_scan_key_bank, "finish_key_bank_block", null);
        k_scan_single_block = clCreateKernel(p_scan_key_bank, "scan_single_block", null);;
        k_scan_multi_block = clCreateKernel(p_scan_key_bank, "scan_multi_block", null);;
        k_complete_multi_block = clCreateKernel(p_scan_key_bank, "complete_multi_block", null);


        p_generate_keys = clCreateProgramWithSource(context, 1, new String[]{src_generate_keys}, null, null);
        clBuildProgram(p_generate_keys, 1, device_id, null, null, null);
        k_generate_keys = clCreateKernel(p_generate_keys, "generate_keys", null);

        p_build_key_map = clCreateProgramWithSource(context, 1, new String[]{src_build_key_map}, null, null);
        clBuildProgram(p_build_key_map, 1, device_id, null, null, null);
        k_build_key_map = clCreateKernel(p_build_key_map, "build_key_map", null);
    }

    public static void destroy()
    {
        clReleaseKernel(k_verletIntegrate);
        clReleaseKernel(k_collide);
        clReleaseKernel(k_scan_key_bank);
        clReleaseKernel(k_scan_key_bank_block);
        clReleaseKernel(k_finish_key_bank_block);
        clReleaseKernel(k_generate_keys);
        clReleaseKernel(k_complete_multi_block);
        clReleaseKernel(k_build_key_map);
        clReleaseProgram(p_verletIntegrate);
        clReleaseProgram(p_collide);
        clReleaseProgram(p_scan_key_bank);
        clReleaseProgram(p_generate_keys);
        clReleaseProgram(p_build_key_map);
        clReleaseCommandQueue(commandQueue);
        clReleaseContext(context);
    }

    public static void generate_key_map(SpatialPartition spatialPartition)
    {
        int[] key_map = spatialPartition.getKey_map();
        int[] key_offsets = spatialPartition.getKey_offsets();
        int[] key_counts = new int[key_offsets.length]; // todo: maybe generate on GPU?
        int x_subdivisions = spatialPartition.getX_subdivisions();
        int boundsSize = Main.Memory.boundsLength();
        var input = FloatBuffer.wrap(Main.Memory.bounds_buffer, 0, boundsSize);
        var minput = IntBuffer.wrap(key_map);
        var oinput = IntBuffer.wrap(key_offsets);
        var cinput = IntBuffer.wrap(key_counts);



        int n = Main.Memory.boundsCount();
        long data_buf_size = (long)Sizeof.cl_float16 * n;
        long map_buf_size = (long)Sizeof.cl_int * key_map.length;
        long offsets_buf_size = (long)Sizeof.cl_int * key_offsets.length;
        long counts_buf_size = (long)Sizeof.cl_int * key_counts.length;
        long flags = CL.CL_MEM_READ_WRITE | CL.CL_MEM_COPY_HOST_PTR;
        cl_mem bounds_data = CL.clCreateBuffer(context, flags, data_buf_size, Pointer.to(input), null);
        cl_mem map_data = CL.clCreateBuffer(context, flags, map_buf_size, Pointer.to(minput), null);
        cl_mem offset_data = CL.clCreateBuffer(context, flags, offsets_buf_size, Pointer.to(oinput), null);
        cl_mem counts_data = CL.clCreateBuffer(context, flags, counts_buf_size, Pointer.to(cinput), null);
        Pointer src_data = Pointer.to(bounds_data);
        Pointer dst_map = Pointer.to(minput);
        Pointer src_map = Pointer.to(map_data);
        //Pointer dst_offset = Pointer.to(key_offsets);
        Pointer src_offset = Pointer.to(offset_data);
        //Pointer dst_counts = Pointer.to(key_counts);
        Pointer src_counts = Pointer.to(counts_data);

        clSetKernelArg(k_build_key_map, 0, Sizeof.cl_mem, src_data);
        clSetKernelArg(k_build_key_map, 1, Sizeof.cl_mem, src_map);
        clSetKernelArg(k_build_key_map, 2, Sizeof.cl_mem, src_offset);
        clSetKernelArg(k_build_key_map, 3, Sizeof.cl_mem, src_counts);
        clSetKernelArg(k_build_key_map, 4, Sizeof.cl_int, Pointer.to(new int[]{x_subdivisions}));
        clSetKernelArg(k_build_key_map, 5, Sizeof.cl_int, Pointer.to(new int[]{key_counts.length}));


        clEnqueueNDRangeKernel(commandQueue, k_build_key_map, 1, null,
            new long[]{n}, null, 0, null, null);

        clEnqueueReadBuffer(commandQueue, map_data, CL_TRUE, 0,
            map_buf_size, dst_map, 0, null, null);
    }

    public static void generate_key_bank(SpatialPartition spatialPartition)
    {
        int[] key_bank = spatialPartition.getKey_bank();
        int[] key_counts = spatialPartition.getKey_counts();
        int x_subdivisions = spatialPartition.getX_subdivisions();
        int boundsSize = Main.Memory.boundsLength();
        var input = FloatBuffer.wrap(Main.Memory.bounds_buffer, 0, boundsSize);
        var binput = IntBuffer.wrap(key_bank);
        var cinput = IntBuffer.wrap(key_counts);

        int n = Main.Memory.boundsCount();
        long data_buf_size = (long)Sizeof.cl_float16 * n;
        long bank_buf_size = (long)Sizeof.cl_int * key_bank.length;
        long counts_buf_size = (long)Sizeof.cl_int * key_counts.length;
        long flags = CL.CL_MEM_READ_WRITE | CL.CL_MEM_COPY_HOST_PTR;
        cl_mem bounds_data = CL.clCreateBuffer(context, flags, data_buf_size, Pointer.to(input), null);
        cl_mem bank_data = CL.clCreateBuffer(context, flags, bank_buf_size, Pointer.to(binput), null);
        cl_mem counts_data = CL.clCreateBuffer(context, flags, counts_buf_size, Pointer.to(cinput), null);
        Pointer src_data = Pointer.to(bounds_data);
        Pointer dst_bank = Pointer.to(key_bank);
        Pointer src_bank = Pointer.to(bank_data);
        Pointer dst_counts = Pointer.to(key_counts);
        Pointer src_counts = Pointer.to(counts_data);

        // pass in arguments
        clSetKernelArg(k_generate_keys, 0, Sizeof.cl_mem, src_data);
        clSetKernelArg(k_generate_keys, 1, Sizeof.cl_mem, src_bank);
        clSetKernelArg(k_generate_keys, 2, Sizeof.cl_mem, src_counts);
        clSetKernelArg(k_generate_keys, 3, Sizeof.cl_int, Pointer.to(new int[]{x_subdivisions}));
        clSetKernelArg(k_generate_keys, 4, Sizeof.cl_int, Pointer.to(new int[]{key_bank.length}));
        clSetKernelArg(k_generate_keys, 5, Sizeof.cl_int, Pointer.to(new int[]{key_counts.length}));

        // call kernel
        clEnqueueNDRangeKernel(commandQueue, k_generate_keys, 1, null,
            new long[]{n}, null, 0, null, null);

        clEnqueueReadBuffer(commandQueue, bank_data, CL_TRUE, 0,
            bank_buf_size, dst_bank, 0, null, null);

        clEnqueueReadBuffer(commandQueue, counts_data, CL_TRUE, 0,
            counts_buf_size, dst_counts, 0, null, null);

        clReleaseMemObject(bounds_data);
        clReleaseMemObject(bank_data);
        clReleaseMemObject(counts_data);
    }

    private static int wx = 256; // todo: query hardware for this limit
    private static int m = wx * 2;

    public static void scan_key_offsets(SpatialPartition spatialPartition)
    {
        int[] key_counts = spatialPartition.getKey_counts();
        int[] key_offsets = spatialPartition.getKey_offsets();
        int n = key_counts.length;
        int k = (int) Math.ceil((float)n / (float)m);
        cl_mem d_data;
        long data_buf_size = (long)Sizeof.cl_int * n;
        long flags = CL.CL_MEM_READ_WRITE | CL.CL_MEM_COPY_HOST_PTR;
        d_data = CL.clCreateBuffer(context, flags, data_buf_size, Pointer.to(key_counts), null);
        Pointer dst_data = Pointer.to(key_offsets);
        scan_int(d_data, n, k);
        clEnqueueReadBuffer(commandQueue, d_data, CL_TRUE, 0,
            data_buf_size, dst_data, 0, null, null);
        clReleaseMemObject(d_data);
    }

    public static void calculate_key_bank_offsets()
    {
        int boundsSize = Main.Memory.boundsLength();
        var input = FloatBuffer.wrap(Main.Memory.bounds_buffer, 0, boundsSize);

        int n = Main.Memory.boundsCount();
        int k = (int) Math.ceil((float)n / (float)m);
        cl_mem d_data;
        long data_buf_size = (long)Sizeof.cl_float16 * n;
        long flags = CL.CL_MEM_READ_WRITE | CL.CL_MEM_COPY_HOST_PTR;
        d_data = CL.clCreateBuffer(context, flags, data_buf_size, Pointer.to(input), null);

        Pointer dst_data = Pointer.to(input);
        scan_key_bounds(d_data, n, k);
        // transfer results into local memory.
        // Note: this is only needed once, after all operations are done
        clEnqueueReadBuffer(commandQueue, d_data, CL_TRUE, 0,
            data_buf_size, dst_data, 0, null, null);
        clReleaseMemObject(d_data);
    }


    private static void scan_int(cl_mem d_data, int n, int k)
    {
        if (k == 1)
        {
            scan_single_block_int(d_data, n);
        }
        else
        {
            scan_multi_block_int(d_data, n, k);
        }
    }

    private static void scan_single_block_int(cl_mem d_data, int n)
    {
        // set up buffers
        int localBufferSize = Sizeof.cl_int * m;
        Pointer src_data = Pointer.to(d_data);

        // pass in arguments
        clSetKernelArg(k_scan_single_block, 0, Sizeof.cl_mem, src_data);
        clSetKernelArg(k_scan_single_block, 1, localBufferSize,null);
        clSetKernelArg(k_scan_single_block, 2, Sizeof.cl_int, Pointer.to(new int[]{n}));

        // call kernel
        clEnqueueNDRangeKernel(commandQueue, k_scan_single_block, 1, null,
            new long[]{wx}, new long[]{wx}, 0, null, null);
    }

    private static void scan_multi_block_int(cl_mem d_data, int n, int k)
    {
        // set up buffers
        int localBufferSize = Sizeof.cl_int * m;
        int gx = k * m;
        long part_buf_size = ((long)Sizeof.cl_int * ((long)k * 2));
        int[] partial_sums = new int[k * 2];
        cl_mem p_data = CL.clCreateBuffer(context, CL.CL_MEM_READ_WRITE | CL.CL_MEM_COPY_HOST_PTR, part_buf_size, Pointer.to(partial_sums), null);
        Pointer src_data = Pointer.to(d_data);
        Pointer src_part = Pointer.to(p_data);

        // pass in arguments
        clSetKernelArg(k_scan_multi_block, 0, Sizeof.cl_mem, src_data);
        clSetKernelArg(k_scan_multi_block, 1, localBufferSize,null);
        clSetKernelArg(k_scan_multi_block, 2, Sizeof.cl_mem, src_part);
        clSetKernelArg(k_scan_multi_block, 3, Sizeof.cl_int, Pointer.to(new int[]{n}));

        // call kernel
        clEnqueueNDRangeKernel(commandQueue, k_scan_multi_block, 1, null,
            new long[]{gx}, new long[]{wx}, 0, null, null);

        // do scan on block sums
        int n2 = partial_sums.length;
        int k2 = (int) Math.ceil((float)n2 / (float)m);
        scan_int(p_data, n2, k2);

        // pass in arguments
        clSetKernelArg(k_complete_multi_block, 0, Sizeof.cl_mem, src_data);
        clSetKernelArg(k_complete_multi_block, 1, localBufferSize,null);
        clSetKernelArg(k_complete_multi_block, 2, Sizeof.cl_mem, src_part);
        clSetKernelArg(k_complete_multi_block, 3, Sizeof.cl_int, Pointer.to(new int[]{n}));

        // call kernel
        clEnqueueNDRangeKernel(commandQueue, k_complete_multi_block, 1, null,
            new long[]{gx}, new long[]{wx}, 0, null, null);

        clReleaseMemObject(p_data);
    }


    private static void scan_key_bounds(cl_mem d_data, int n, int k)
    {
        if (k == 1)
        {
            scan_single_block_key(d_data, n);
        }
        else
        {
            scan_multi_block_key(d_data, n, k);
        }
    }

    private static void scan_single_block_key(cl_mem d_data, int n)
    {
        // set up buffers
        int localBufferSize = Sizeof.cl_int * m;
        Pointer src_data = Pointer.to(d_data);

        // pass in arguments
        clSetKernelArg(k_scan_key_bank, 0, Sizeof.cl_mem, src_data);
        clSetKernelArg(k_scan_key_bank, 1, localBufferSize,null);
        clSetKernelArg(k_scan_key_bank, 2, Sizeof.cl_int, Pointer.to(new int[]{n}));

        // call kernel
        clEnqueueNDRangeKernel(commandQueue, k_scan_key_bank, 1, null,
            new long[]{wx}, new long[]{wx}, 0, null, null);
    }

    private static void scan_multi_block_key(cl_mem d_data, int n, int k)
    {
        // set up buffers
        int localBufferSize = Sizeof.cl_int * m;
        int gx = k * m;
        long part_buf_size = ((long)Sizeof.cl_int * ((long)k * 2));
        int[] partial_sums = new int[k * 2];
        cl_mem p_data = CL.clCreateBuffer(context, CL.CL_MEM_READ_WRITE | CL.CL_MEM_COPY_HOST_PTR, part_buf_size, Pointer.to(partial_sums), null);
        Pointer src_data = Pointer.to(d_data);
        Pointer src_part = Pointer.to(p_data);

        // pass in arguments
        clSetKernelArg(k_scan_key_bank_block, 0, Sizeof.cl_mem, src_data);
        clSetKernelArg(k_scan_key_bank_block, 1, localBufferSize,null);
        clSetKernelArg(k_scan_key_bank_block, 2, Sizeof.cl_mem, src_part);
        clSetKernelArg(k_scan_key_bank_block, 3, Sizeof.cl_int, Pointer.to(new int[]{n}));

        // call kernel
        clEnqueueNDRangeKernel(commandQueue, k_scan_key_bank_block, 1, null,
            new long[]{gx}, new long[]{wx}, 0, null, null);

        // do scan on block sums
        int n2 = partial_sums.length;
        int k2 = (int) Math.ceil((float)n2 / (float)m);


        scan_int(p_data, n2, k2);

        // pass in arguments
        clSetKernelArg(k_finish_key_bank_block, 0, Sizeof.cl_mem, src_data);
        clSetKernelArg(k_finish_key_bank_block, 1, localBufferSize,null);
        clSetKernelArg(k_finish_key_bank_block, 2, Sizeof.cl_mem, src_part);
        clSetKernelArg(k_finish_key_bank_block, 3, Sizeof.cl_int, Pointer.to(new int[]{n}));

        // call kernel
        clEnqueueNDRangeKernel(commandQueue, k_finish_key_bank_block, 1, null,
            new long[]{gx}, new long[]{wx}, 0, null, null);

        clReleaseMemObject(p_data);
    }






    public static void integrate(float tick_rate, float gravity_x, float gravity_y, float friction, SpatialPartition spatialPartition)
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
            (float)spatialPartition.getY_subdivisions(),
            gravity_x,
            gravity_y,
            friction
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
