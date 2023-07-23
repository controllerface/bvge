package com.controllerface.bvge.cl;

import com.controllerface.bvge.Main;
import com.controllerface.bvge.ecs.systems.physics.MemoryBuffer;
import com.controllerface.bvge.ecs.systems.physics.PhysicsBuffer;
import com.controllerface.bvge.ecs.systems.physics.SpatialPartition;
import org.jocl.*;

import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.util.ArrayList;
import java.util.List;

import static com.controllerface.bvge.cl.OpenCLUtils.read_src;
import static org.jocl.CL.*;

public class OpenCL
{
    static cl_command_queue commandQueue;
    static cl_context context;
    static cl_device_id[] device_ids;

    static String prag_int32_base_atomics   = read_src("pragma/int32_base_atomics.cl");;

    /**
     * Helper functions
     */
    static String func_is_in_bounds         = read_src("functions/is_in_bounds.cl");
    static String func_get_extents          = read_src("functions/get_extents.cl");
    static String func_get_key_for_point    = read_src("functions/get_key_for_point.cl");
    static String func_calculate_key_index  = read_src("functions/calculate_key_index.cl");
    static String func_exclusive_scan       = read_src("functions/exclusive_scan.cl");
    static String func_do_bounds_intersect  = read_src("functions/do_bounds_intersect.cl");
    static String func_project_polygon      = read_src("functions/project_polygon.cl");
    static String func_polygon_distance     = read_src("functions/polygon_distance.cl");
    static String func_edge_contact         = read_src("functions/edge_contact.cl");

    /**
     * Core kernel files
     */
    static String kern_integrate            = read_src("kernels/integrate.cl");
    static String kern_sat_collide          = read_src("kernels/sat_collide.cl");
    static String kern_aabb_collide         = read_src("kernels/aabb_collide.cl");

    static String kern_scan_key_bank        = read_src("kernels/scan_key_bank.cl");
    static String kern_scan_int_array       = read_src("kernels/scan_int_array.cl");
    static String kern_scan_int_array_out   = read_src("kernels/scan_int_array_out.cl");
    static String kern_scan_candidates_out  = read_src("kernels/scan_key_candidates.cl");
    static String kern_generate_keys        = read_src("kernels/generate_keys.cl");
    static String kern_build_key_map        = read_src("kernels/build_key_map.cl");
    static String kern_resolve_constraints  = read_src("kernels/resolve_constraints.cl");
    static String kern_locate_in_bounds     = read_src("kernels/locate_in_bounds.cl");

    /**
     * Kernel function names
     */
    static String kn_locate_in_bounds                   = "locate_in_bounds";
    static String kn_count_candidates                   = "count_candidates";
    static String kn_finalize_candidates                = "finalize_candidates";
    static String kn_integrate                          = "integrate";
    static String kn_sat_collide                        = "sat_collide";
    static String kn_aabb_collide                       = "aabb_collide";
    static String kn_scan_bounds_single_block           = "scan_bounds_single_block";
    static String kn_scan_bounds_multi_block            = "scan_bounds_multi_block";
    static String kn_complete_bounds_multi_block        = "complete_bounds_multi_block";
    static String kn_scan_int_single_block              = "scan_int_single_block";
    static String kn_scan_int_multi_block               = "scan_int_multi_block";
    static String kn_complete_int_multi_block           = "complete_int_multi_block";
    static String kn_scan_int_single_block_out          = "scan_int_single_block_out";
    static String kn_scan_int_multi_block_out           = "scan_int_multi_block_out";
    static String kn_complete_int_multi_block_out       = "complete_int_multi_block_out";
    static String kn_scan_candidates_single_block       = "scan_candidates_single_block_out";
    static String kn_scan_candidates_multi_block        = "scan_candidates_multi_block_out";
    static String kn_complete_candidates_multi_block    = "complete_candidates_multi_block_out";
    static String kn_generate_keys                      = "generate_keys";
    static String kn_build_key_map                      = "build_key_map";
    static String kn_resolve_constraints                = "resolve_constraints";

    /**
     * CL Programs
     */
    static cl_program p_locate_in_bounds;
    static cl_program p_integrate;
    static cl_program p_sat_collide;
    static cl_program p_aabb_collide;
    static cl_program p_scan_key_bank;
    static cl_program p_scan_int_array;
    static cl_program p_scan_int_array_out;
    static cl_program p_scan_candidates;
    static cl_program p_generate_keys;
    static cl_program p_build_key_map;
    static cl_program p_resolve_constraints;

    /**
     * CL Kernels
     */
    static cl_kernel k_locate_in_bounds;
    static cl_kernel k_count_candidates;
    static cl_kernel k_finalize_candidates;
    static cl_kernel k_integrate;
    static cl_kernel k_sat_collide;
    static cl_kernel k_aabb_collide;
    static cl_kernel k_scan_bounds_single_block;
    static cl_kernel k_scan_bounds_multi_block;
    static cl_kernel k_complete_bounds_multi_block;
    static cl_kernel k_scan_int_single_block;
    static cl_kernel k_scan_int_multi_block;
    static cl_kernel k_complete_int_multi_block;
    static cl_kernel k_scan_int_single_block_out;
    static cl_kernel k_scan_int_multi_block_out;
    static cl_kernel k_complete_int_multi_block_out;
    static cl_kernel k_scan_candidates_single_block;
    static cl_kernel k_scan_candidates_multi_block;
    static cl_kernel k_complete_candidates_multi_block;
    static cl_kernel k_generate_keys;
    static cl_kernel k_build_key_map;
    static cl_kernel k_resolve_constraints;

    /**
     * During shutdown, these are used to release resources.
     */
    static List<cl_program> loaded_programs = new ArrayList<>();
    static List<cl_kernel> loaded_kernels = new ArrayList<>();

    private static final int wx = 256; // todo: query hardware for this limit
    private static final int m = wx * 2;
    private static final long[] local_work_default = new long[]{wx};
    private static cl_device_id[] device_init()
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

    public static cl_command_queue getCommandQueue()
    {
        return commandQueue;
    }

    private static cl_program cl_p(String ... src)
    {
        var program = OpenCLUtils.cl_p(context, device_ids, src);
        loaded_programs.add(program);
        return program;
    }

    private static cl_kernel cl_k(cl_program program, String kernel_name)
    {
        var kernel = OpenCLUtils.cl_k(program, kernel_name);
        loaded_kernels.add(kernel);
        return kernel;
    }

    private static int work_group_count(int n)
    {
        return (int) Math.ceil((float)n / (float)m);
    }

    public static void init()
    {
        device_ids = device_init();

        OpenCLUtils.printDeviceDetails(device_ids);

        /*
         * Programs
         */
        p_sat_collide = cl_p(
            func_project_polygon,
            func_polygon_distance,
            func_edge_contact,
            kern_sat_collide);

        p_aabb_collide = cl_p(
            func_do_bounds_intersect,
            func_calculate_key_index,
            kern_aabb_collide);

        p_integrate = cl_p(func_is_in_bounds,
            func_get_extents,
            func_get_key_for_point,
            kern_integrate);

        p_locate_in_bounds = cl_p(prag_int32_base_atomics,
            func_do_bounds_intersect,
            func_calculate_key_index,
            kern_locate_in_bounds);

        p_scan_key_bank = cl_p(func_exclusive_scan,
            kern_scan_key_bank);

        p_scan_int_array = cl_p(func_exclusive_scan,
            kern_scan_int_array);

        p_scan_int_array_out = cl_p(func_exclusive_scan,
            kern_scan_int_array_out);

        p_scan_candidates = cl_p(func_exclusive_scan,
            kern_scan_candidates_out);

        p_generate_keys = cl_p(prag_int32_base_atomics,
            func_calculate_key_index,
            kern_generate_keys);

        p_build_key_map = cl_p(prag_int32_base_atomics,
            func_calculate_key_index,
            kern_build_key_map);

        p_resolve_constraints = cl_p(kern_resolve_constraints);

        /*
         * Kernels
         */
        k_integrate                         = cl_k(p_integrate, kn_integrate);
        k_sat_collide                       = cl_k(p_sat_collide, kn_sat_collide);
        k_aabb_collide                      = cl_k(p_aabb_collide, kn_aabb_collide);
        k_locate_in_bounds                  = cl_k(p_locate_in_bounds, kn_locate_in_bounds);
        k_count_candidates                  = cl_k(p_locate_in_bounds, kn_count_candidates);
        k_finalize_candidates               = cl_k(p_locate_in_bounds, kn_finalize_candidates);
        k_scan_bounds_single_block          = cl_k(p_scan_key_bank, kn_scan_bounds_single_block);
        k_scan_bounds_multi_block           = cl_k(p_scan_key_bank, kn_scan_bounds_multi_block);
        k_complete_bounds_multi_block       = cl_k(p_scan_key_bank, kn_complete_bounds_multi_block);
        k_scan_int_single_block             = cl_k(p_scan_int_array, kn_scan_int_single_block);
        k_scan_int_multi_block              = cl_k(p_scan_int_array, kn_scan_int_multi_block);
        k_complete_int_multi_block          = cl_k(p_scan_int_array, kn_complete_int_multi_block);
        k_scan_int_single_block_out         = cl_k(p_scan_int_array_out, kn_scan_int_single_block_out);
        k_scan_int_multi_block_out          = cl_k(p_scan_int_array_out, kn_scan_int_multi_block_out);
        k_complete_int_multi_block_out      = cl_k(p_scan_int_array_out, kn_complete_int_multi_block_out);
        k_scan_candidates_single_block      = cl_k(p_scan_candidates, kn_scan_candidates_single_block);
        k_scan_candidates_multi_block       = cl_k(p_scan_candidates, kn_scan_candidates_multi_block);
        k_complete_candidates_multi_block   = cl_k(p_scan_candidates, kn_complete_candidates_multi_block);
        k_generate_keys                     = cl_k(p_generate_keys, kn_generate_keys);
        k_build_key_map                     = cl_k(p_build_key_map, kn_build_key_map);
        k_resolve_constraints               = cl_k(p_resolve_constraints, kn_resolve_constraints);
    }

    public static void destroy()
    {
        loaded_programs.forEach(CL::clReleaseProgram);
        loaded_kernels.forEach(CL::clReleaseKernel);
        clReleaseCommandQueue(commandQueue);
        clReleaseContext(context);
    }

    public static void makeBuffers(PhysicsBuffer physicsBuffer)
    {
        int bodiesSize = Main.Memory.bodyLength();
        int pointsSize = Main.Memory.pointLength();
        int boundsSize = Main.Memory.boundsLength();
        int edgesSize = Main.Memory.edgesLength();

        var bodyBuffer = FloatBuffer.wrap(Main.Memory.body_buffer, 0, bodiesSize);
        var pointBuffer = FloatBuffer.wrap(Main.Memory.point_buffer, 0, pointsSize);
        var boundsBuffer = FloatBuffer.wrap(Main.Memory.bounds_buffer, 0, boundsSize);
        var edgesBuffer = FloatBuffer.wrap(Main.Memory.edge_buffer, 0, edgesSize);

        Pointer srcBodies = Pointer.to(bodyBuffer);
        Pointer srcPoints = Pointer.to(pointBuffer);
        Pointer srcBounds = Pointer.to(boundsBuffer);
        Pointer srcEdges = Pointer.to(edgesBuffer);

        long bodyBufsize = (long)Sizeof.cl_float * bodiesSize;
        long pointBufsize = (long)Sizeof.cl_float * pointsSize;
        long boundsBufsize = (long)Sizeof.cl_float * boundsSize;
        long edgesBufsize = (long) Sizeof.cl_float * edgesSize;

        cl_mem srcMemBodies = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, bodyBufsize, srcBodies, null);
        cl_mem srcMemPoints = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, pointBufsize, srcPoints, null);
        cl_mem srcMemBounds = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, boundsBufsize, srcBounds, null);
        cl_mem srcMemEdges = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, edgesBufsize, srcEdges, null);


        physicsBuffer.bounds = new MemoryBuffer(srcMemBounds, boundsBufsize, srcBounds);
        physicsBuffer.bodies = new MemoryBuffer(srcMemBodies, bodyBufsize, srcBodies);
        physicsBuffer.points = new MemoryBuffer(srcMemPoints, pointBufsize, srcPoints);
        physicsBuffer.edges = new MemoryBuffer(srcMemEdges, edgesBufsize, srcEdges);
    }

    public static void finalize_candidates(PhysicsBuffer physicsBuffer)
    {
        if (physicsBuffer.get_candidate_count() > 0)
        {
            int[] finals = new int[physicsBuffer.get_candidate_count() * 2];
            Pointer dst_finals = Pointer.to(finals);
            long final_buf_size = (long)Sizeof.cl_int2 * physicsBuffer.get_candidate_count();
            cl_mem finals_data = CL.clCreateBuffer(context, CL.CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, final_buf_size, dst_finals, null);
            Pointer src_finals = Pointer.to(finals_data);

            int[] sz3 = new int[1];
            Pointer dst_size3 = Pointer.to(sz3);
            cl_mem size_data3 = CL.clCreateBuffer(context, CL.CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, Sizeof.cl_int, dst_size3, null);
            Pointer src_size3 = Pointer.to(size_data3);

            physicsBuffer.candidates = new MemoryBuffer(finals_data, final_buf_size, dst_finals);

            clSetKernelArg(k_finalize_candidates, 0, Sizeof.cl_mem, physicsBuffer.candidate_counts.pointer());
            clSetKernelArg(k_finalize_candidates, 1, Sizeof.cl_mem, physicsBuffer.candidate_offsets.pointer());
            clSetKernelArg(k_finalize_candidates, 2, Sizeof.cl_mem, physicsBuffer.matches.pointer());
            clSetKernelArg(k_finalize_candidates, 3, Sizeof.cl_mem, physicsBuffer.matches_used.pointer());
            clSetKernelArg(k_finalize_candidates, 4, Sizeof.cl_mem, src_size3);
            clSetKernelArg(k_finalize_candidates, 5, Sizeof.cl_mem, src_finals);

            clEnqueueNDRangeKernel(commandQueue, k_finalize_candidates, 1, null,
                new long[]{physicsBuffer.get_candidate_buffer_count()}, null, 0, null, null);

            clReleaseMemObject(size_data3);
        }
    }

    public static void aabb_collide(PhysicsBuffer physicsBuffer)
    {
        int[] matches = new int[physicsBuffer.get_candidate_match_count()];
        long matches_buf_size = (long)Sizeof.cl_int * matches.length;
        Pointer pnt_matches = Pointer.to(matches);
        cl_mem matches_data = CL.clCreateBuffer(context, CL.CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, matches_buf_size, pnt_matches, null);
        physicsBuffer.matches = new MemoryBuffer(matches_data, matches_buf_size, pnt_matches);

        int[] used = new int[physicsBuffer.get_candidate_buffer_count()];
        long used_buf_size = (long)Sizeof.cl_int * used.length;
        Pointer pnt_used = Pointer.to(used);
        cl_mem used_data = CL.clCreateBuffer(context, CL.CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, used_buf_size, pnt_used, null);
        physicsBuffer.matches_used = new MemoryBuffer(used_data, used_buf_size, pnt_used);

        int[] sz2 = new int[1];
        Pointer dst_size2 = Pointer.to(sz2);
        cl_mem size_data2 = CL.clCreateBuffer(context, CL.CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, Sizeof.cl_int, dst_size2, null);
        Pointer src_size2 = Pointer.to(size_data2);

        clSetKernelArg(k_aabb_collide, 0, Sizeof.cl_mem, physicsBuffer.bounds.pointer());
        clSetKernelArg(k_aabb_collide, 1, Sizeof.cl_mem, physicsBuffer.candidate_counts.pointer());
        clSetKernelArg(k_aabb_collide, 2, Sizeof.cl_mem, physicsBuffer.candidate_offsets.pointer());
        clSetKernelArg(k_aabb_collide, 3, Sizeof.cl_mem, physicsBuffer.key_map.pointer());
        clSetKernelArg(k_aabb_collide, 4, Sizeof.cl_mem, physicsBuffer.key_bank.pointer());
        clSetKernelArg(k_aabb_collide, 5, Sizeof.cl_mem, physicsBuffer.key_counts.pointer());
        clSetKernelArg(k_aabb_collide, 6, Sizeof.cl_mem, physicsBuffer.key_offsets.pointer());
        clSetKernelArg(k_aabb_collide, 7, Sizeof.cl_mem, physicsBuffer.matches.pointer());
        clSetKernelArg(k_aabb_collide, 8, Sizeof.cl_mem, physicsBuffer.matches_used.pointer());
        clSetKernelArg(k_aabb_collide, 9, Sizeof.cl_mem, src_size2);
        clSetKernelArg(k_aabb_collide, 10, Sizeof.cl_int, physicsBuffer.x_sub_divisions);
        clSetKernelArg(k_aabb_collide, 11, Sizeof.cl_int, physicsBuffer.key_count_length);

        clEnqueueNDRangeKernel(commandQueue, k_aabb_collide, 1, null,
            new long[]{physicsBuffer.get_candidate_buffer_count()}, null, 0, null, null);

        clEnqueueReadBuffer(commandQueue, size_data2, CL_TRUE, 0,
            Sizeof.cl_int, dst_size2, 0, null, null);

        clReleaseMemObject(size_data2);

        physicsBuffer.set_candidate_count(sz2[0]);
    }

    public static void count_matches(PhysicsBuffer physicsBuffer)
    {
        int n2 = physicsBuffer.get_candidate_buffer_count();
        int[] offsets = new int[physicsBuffer.get_candidate_buffer_count()];
        long offset_buf_size = (long)Sizeof.cl_int * n2;
        Pointer pnt_offset = Pointer.to(offsets);
        cl_mem offset_data = CL.clCreateBuffer(context, CL.CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, offset_buf_size, pnt_offset, null);
        physicsBuffer.candidate_offsets = new MemoryBuffer(offset_data, offset_buf_size, pnt_offset);

        int match_count = scan_key_candidates(physicsBuffer.candidate_counts.memory(), offset_data, n2);
        physicsBuffer.set_candidate_match_count(match_count);

    }

    public static void count_candidates(PhysicsBuffer physicsBuffer)
    {
        long cand_buf_size = (long)Sizeof.cl_int2 * physicsBuffer.get_candidate_buffer_count();
        int[] cand_counts = new int[physicsBuffer.get_candidate_buffer_count() * 2];
        Pointer pnt_cand_counts = Pointer.to(cand_counts);
        cl_mem cand_data = CL.clCreateBuffer(context, CL.CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, cand_buf_size, pnt_cand_counts, null);
        physicsBuffer.candidate_counts = new MemoryBuffer(cand_data, cand_buf_size, pnt_cand_counts);

        clSetKernelArg(k_count_candidates, 0, Sizeof.cl_mem, physicsBuffer.bounds.pointer());
        clSetKernelArg(k_count_candidates, 1, Sizeof.cl_mem, physicsBuffer.in_bounds.pointer());
        clSetKernelArg(k_count_candidates, 2, Sizeof.cl_mem, physicsBuffer.key_bank.pointer());
        clSetKernelArg(k_count_candidates, 3, Sizeof.cl_mem, physicsBuffer.key_counts.pointer());
        clSetKernelArg(k_count_candidates, 4, Sizeof.cl_mem, physicsBuffer.candidate_counts.pointer());
        clSetKernelArg(k_count_candidates, 5, Sizeof.cl_int, physicsBuffer.x_sub_divisions);
        clSetKernelArg(k_count_candidates, 6, Sizeof.cl_int, physicsBuffer.key_count_length);

        clEnqueueNDRangeKernel(commandQueue, k_count_candidates, 1, null,
            new long[]{physicsBuffer.get_candidate_buffer_count()}, null, 0, null, null);
    }

    public static void locate_in_bounds(PhysicsBuffer physicsBuffer, SpatialPartition spatialPartition)
    {
        // step 1: locate objects that are within bounds
        int x_subdivisions = spatialPartition.getX_subdivisions();
        physicsBuffer.x_sub_divisions = Pointer.to(new int[]{x_subdivisions});
        physicsBuffer.key_count_length = Pointer.to(new int[]{spatialPartition.getKey_counts().length});

        int n = Main.Memory.bodyCount();
        int[] in_bounds = new int[n];
        var pnt_inbound = Pointer.to(in_bounds);
        long inbound_buf_size = (long)Sizeof.cl_int * in_bounds.length;
        cl_mem inbound_data = CL.clCreateBuffer(context, CL.CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
            inbound_buf_size, pnt_inbound, null);

        physicsBuffer.in_bounds = new MemoryBuffer(inbound_data, inbound_buf_size, pnt_inbound);

        int[] sz = new int[1];
        Pointer dst_size = Pointer.to(sz);
        cl_mem size_data = CL.clCreateBuffer(context, CL.CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, Sizeof.cl_int, dst_size, null);
        Pointer src_size = Pointer.to(size_data);

        clSetKernelArg(k_locate_in_bounds, 0, Sizeof.cl_mem, physicsBuffer.bounds.pointer());
        clSetKernelArg(k_locate_in_bounds, 1, Sizeof.cl_mem, physicsBuffer.in_bounds.pointer());
        clSetKernelArg(k_locate_in_bounds, 2, Sizeof.cl_mem, src_size);

        clEnqueueNDRangeKernel(commandQueue, k_locate_in_bounds, 1, null,
            new long[]{n}, null, 0, null, null);

        clEnqueueReadBuffer(commandQueue, size_data, CL_TRUE, 0,
            Sizeof.cl_int, dst_size, 0, null, null);

        clReleaseMemObject(size_data);

        physicsBuffer.set_candidate_buffer_count(sz[0]);



    }

    public static void generate_key_map(PhysicsBuffer physicsBuffer, SpatialPartition spatialPartition)
    {
        int[] key_map = spatialPartition.getKey_map();
        // Note: this is not the same as the local counts in the spatial map.
        // This buffer is used only during calculations within the kernel
        int[] key_counts = new int[spatialPartition.getKey_offsets().length];
        int x_subdivisions = spatialPartition.getX_subdivisions();
        var minput = IntBuffer.wrap(key_map);
        var cinput = IntBuffer.wrap(key_counts);

        int n = Main.Memory.boundsCount();
        long map_buf_size = (long)Sizeof.cl_int * key_map.length;
        long counts_buf_size = (long)Sizeof.cl_int * key_counts.length;
        long flags = CL.CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR ;
        cl_mem map_data = CL.clCreateBuffer(context, flags, map_buf_size, Pointer.to(minput), null);
        cl_mem counts_data = CL.clCreateBuffer(context, flags, counts_buf_size, Pointer.to(cinput), null);

        Pointer src_data = Pointer.to(physicsBuffer.bounds.memory());
        Pointer dst_map = Pointer.to(minput);
        Pointer src_map = Pointer.to(map_data);
        Pointer src_offset = Pointer.to(physicsBuffer.key_offsets.memory());
        Pointer src_counts = Pointer.to(counts_data);

        physicsBuffer.key_map = new MemoryBuffer(map_data, map_buf_size, dst_map);

        clSetKernelArg(k_build_key_map, 0, Sizeof.cl_mem, src_data);
        clSetKernelArg(k_build_key_map, 1, Sizeof.cl_mem, src_map);
        clSetKernelArg(k_build_key_map, 2, Sizeof.cl_mem, src_offset);
        clSetKernelArg(k_build_key_map, 3, Sizeof.cl_mem, src_counts);
        clSetKernelArg(k_build_key_map, 4, Sizeof.cl_int, Pointer.to(new int[]{x_subdivisions}));
        clSetKernelArg(k_build_key_map, 5, Sizeof.cl_int, Pointer.to(new int[]{key_counts.length}));

        clEnqueueNDRangeKernel(commandQueue, k_build_key_map, 1, null,
            new long[]{n}, null, 0, null, null);

        clReleaseMemObject(counts_data);
    }

    public static void generate_key_bank(PhysicsBuffer physicsBuffer, SpatialPartition spatialPartition)
    {
        int[] key_bank = spatialPartition.getKey_bank();
        int[] key_counts = spatialPartition.getKey_counts();
        int x_subdivisions = spatialPartition.getX_subdivisions();
        var binput = IntBuffer.wrap(key_bank);
        var cinput = IntBuffer.wrap(key_counts);

        int n = Main.Memory.boundsCount();
        long bank_buf_size = (long)Sizeof.cl_int * key_bank.length;
        long counts_buf_size = (long)Sizeof.cl_int * key_counts.length;
        long flags = CL.CL_MEM_READ_WRITE | CL.CL_MEM_COPY_HOST_PTR;
        cl_mem bank_data = CL.clCreateBuffer(context, flags, bank_buf_size, Pointer.to(binput), null);
        cl_mem counts_data = CL.clCreateBuffer(context, flags, counts_buf_size, Pointer.to(cinput), null);

        Pointer src_data = Pointer.to(physicsBuffer.bounds.memory());
        Pointer dst_bank = Pointer.to(key_bank);
        Pointer src_bank = Pointer.to(bank_data);
        Pointer dst_counts = Pointer.to(key_counts);
        Pointer src_counts = Pointer.to(counts_data);

        physicsBuffer.key_counts = new MemoryBuffer(counts_data, counts_buf_size, dst_counts);
        physicsBuffer.key_bank = new MemoryBuffer(bank_data, bank_buf_size, dst_bank);

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
    }

    public static void calculate_map_offsets(PhysicsBuffer physicsBuffer, SpatialPartition spatialPartition)
    {
        int[] key_offsets = spatialPartition.getKey_offsets();
        int n = spatialPartition.getKey_counts().length;

        long data_buf_size = (long)Sizeof.cl_int * n;
        long flags = CL.CL_MEM_READ_WRITE | CL.CL_MEM_COPY_HOST_PTR;

        Pointer dst_data = Pointer.to(key_offsets);
        cl_mem o_data = CL.clCreateBuffer(context, flags, data_buf_size, dst_data, null);
        physicsBuffer.key_offsets = new MemoryBuffer(o_data, data_buf_size, dst_data);

        scan_int_out(physicsBuffer.key_counts.memory(), o_data, n);
    }

    public static void calculate_bank_offsets(PhysicsBuffer physicsBuffer, SpatialPartition spatialPartition)
    {
        int n = Main.Memory.boundsCount();
        int bank_size = scan_key_bounds(physicsBuffer.bounds.memory(), n);
        spatialPartition.resizeBank(bank_size);
    }

    public static void integrate(float delta_time, PhysicsBuffer physicsBuffer, SpatialPartition spatialPartition)
    {
        long[] global_work_size = new long[]{Main.Memory.bodyCount()};
        float[] args =
            {
                delta_time,
                spatialPartition.getX_spacing(),
                spatialPartition.getY_spacing(),
                spatialPartition.getX_origin(),
                spatialPartition.getY_origin(),
                spatialPartition.getWidth(),
                spatialPartition.getHeight(),
                (float)spatialPartition.getX_subdivisions(),
                (float)spatialPartition.getY_subdivisions(),
                physicsBuffer.get_gravity_x(),
                physicsBuffer.get_gravity_y(),
                physicsBuffer.get_friction()
            };

        Pointer srcDt = Pointer.to(FloatBuffer.wrap(args));
        cl_mem dtMem = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, Sizeof.cl_float * args.length, srcDt, null);

        int a = 0;
        clSetKernelArg(k_integrate, a++, Sizeof.cl_mem, Pointer.to(physicsBuffer.bodies.memory()));
        clSetKernelArg(k_integrate, a++, Sizeof.cl_mem, Pointer.to(physicsBuffer.points.memory()));
        clSetKernelArg(k_integrate, a++, Sizeof.cl_mem, Pointer.to(physicsBuffer.bounds.memory()));
        clSetKernelArg(k_integrate, a++, Sizeof.cl_mem, Pointer.to(dtMem));

        clEnqueueNDRangeKernel(commandQueue, k_integrate, 1, null,
            global_work_size, null, 0, null, null);

        clReleaseMemObject(dtMem);
    }

    public static void sat_collide(PhysicsBuffer physicsBuffer)
    {
        if (physicsBuffer.candidates == null) return;

        int candidatesSize = (int) physicsBuffer.candidates.getSize() / Sizeof.cl_int;
        //int reactionsSize = reactions.limit();

        // Set the work-item dimensions
        long global_work_size[] = new long[]{candidatesSize / Main.Memory.Width.COLLISION};

        // Set the arguments for the kernel
        clSetKernelArg(k_sat_collide, 0, Sizeof.cl_mem, Pointer.to(physicsBuffer.candidates.memory()));
        clSetKernelArg(k_sat_collide, 1, Sizeof.cl_mem, Pointer.to(physicsBuffer.bodies.memory()));
        clSetKernelArg(k_sat_collide, 2, Sizeof.cl_mem, Pointer.to(physicsBuffer.points.memory()));

        // Execute the kernel
        clEnqueueNDRangeKernel(commandQueue, k_sat_collide, 1, null,
            global_work_size, null, 0, null, null);
    }

    public static void resolve_constraints(PhysicsBuffer physicsBuffer, int edge_steps)
    {
        boolean lastStep;
        long global_work_size[] = new long[]{Main.Memory.bodyCount()};
        for (int i = 0; i < edge_steps; i++)
        {
            lastStep = i == edge_steps - 1;
            int n = lastStep ? 1 : 0;
            int a = 0;
            clSetKernelArg(k_resolve_constraints, a++, Sizeof.cl_mem, physicsBuffer.bodies.pointer());
            clSetKernelArg(k_resolve_constraints, a++, Sizeof.cl_mem, physicsBuffer.bounds.pointer());
            clSetKernelArg(k_resolve_constraints, a++, Sizeof.cl_mem, physicsBuffer.points.pointer());
            clSetKernelArg(k_resolve_constraints, a++, Sizeof.cl_mem, physicsBuffer.edges.pointer());
            clSetKernelArg(k_resolve_constraints, a++, Sizeof.cl_int, Pointer.to(new int[]{n}));

            // Execute the kernel
            clEnqueueNDRangeKernel(commandQueue, k_resolve_constraints, 1, null,
                global_work_size, null, 0, null, null);
        }
    }

    //#region Exclusive scan variants

    private static void scan_int(cl_mem d_data, int n)
    {
        int k = work_group_count(n);
        if (k == 1)
        {
            scan_single_block_int(d_data, n);
        }
        else
        {
            scan_multi_block_int(d_data, n, k);
        }
    }

    private static void scan_int_out(cl_mem d_data, cl_mem o_data, int n)
    {
        int k = work_group_count(n);
        if (k == 1)
        {
            scan_single_block_int_out(d_data, o_data, n);
        }
        else
        {
            scan_multi_block_int_out(d_data, o_data, n, k);
        }
    }

    private static int scan_key_bounds(cl_mem d_data, int n)
    {
        int k = work_group_count(n);
        if (k == 1)
        {
            return scan_single_block_key(d_data, n);
        }
        else
        {
            return scan_multi_block_key(d_data, n, k);
        }
    }

    private static int scan_key_candidates(cl_mem d_data, cl_mem o_data, int n)
    {
        int k = work_group_count(n);
        if (k == 1)
        {
            return scan_single_block_candidates_out(d_data, o_data, n);
        }
        else
        {
            return scan_multi_block_candidates_out(d_data, o_data, n, k);
        }
    }

    private static void scan_single_block_int(cl_mem d_data, int n)
    {
        int localBufferSize = Sizeof.cl_int * m;
        Pointer src_data = Pointer.to(d_data);

        clSetKernelArg(k_scan_int_single_block, 0, Sizeof.cl_mem, src_data);
        clSetKernelArg(k_scan_int_single_block, 1, localBufferSize,null);
        clSetKernelArg(k_scan_int_single_block, 2, Sizeof.cl_int, Pointer.to(new int[]{n}));

        clEnqueueNDRangeKernel(commandQueue, k_scan_int_single_block, 1, null,
            local_work_default, local_work_default, 0, null, null);
    }

    private static void scan_multi_block_int(cl_mem d_data, int n, int k)
    {
        int localBufferSize = Sizeof.cl_int * m;
        int gx = k * m;
        long part_buf_size = ((long)Sizeof.cl_int * ((long)k * 2));
        int[] partial_sums = new int[k * 2];
        cl_mem p_data = CL.clCreateBuffer(context, CL.CL_MEM_READ_WRITE | CL.CL_MEM_COPY_HOST_PTR, part_buf_size, Pointer.to(partial_sums), null);
        Pointer src_data = Pointer.to(d_data);
        Pointer src_part = Pointer.to(p_data);

        clSetKernelArg(k_scan_int_multi_block, 0, Sizeof.cl_mem, src_data);
        clSetKernelArg(k_scan_int_multi_block, 1, localBufferSize,null);
        clSetKernelArg(k_scan_int_multi_block, 2, Sizeof.cl_mem, src_part);
        clSetKernelArg(k_scan_int_multi_block, 3, Sizeof.cl_int, Pointer.to(new int[]{n}));

        clEnqueueNDRangeKernel(commandQueue, k_scan_int_multi_block, 1, null,
            new long[]{gx}, local_work_default, 0, null, null);

        int n2 = partial_sums.length;
        scan_int(p_data, n2);

        clSetKernelArg(k_complete_int_multi_block, 0, Sizeof.cl_mem, src_data);
        clSetKernelArg(k_complete_int_multi_block, 1, localBufferSize,null);
        clSetKernelArg(k_complete_int_multi_block, 2, Sizeof.cl_mem, src_part);
        clSetKernelArg(k_complete_int_multi_block, 3, Sizeof.cl_int, Pointer.to(new int[]{n}));

        clEnqueueNDRangeKernel(commandQueue, k_complete_int_multi_block, 1, null,
            new long[]{gx}, local_work_default, 0, null, null);

        clReleaseMemObject(p_data);
    }

    private static void scan_single_block_int_out(cl_mem d_data, cl_mem o_data, int n)
    {
        int localBufferSize = Sizeof.cl_int * m;
        Pointer src_data = Pointer.to(d_data);
        Pointer dst_data = Pointer.to(o_data);

        clSetKernelArg(k_scan_int_single_block_out, 0, Sizeof.cl_mem, src_data);
        clSetKernelArg(k_scan_int_single_block_out, 2, Sizeof.cl_mem, dst_data);
        clSetKernelArg(k_scan_int_single_block_out, 2, localBufferSize,null);
        clSetKernelArg(k_scan_int_single_block_out, 3, Sizeof.cl_int, Pointer.to(new int[]{n}));

        clEnqueueNDRangeKernel(commandQueue, k_scan_int_single_block_out, 1, null,
            local_work_default, local_work_default, 0, null, null);
    }

    private static void scan_multi_block_int_out(cl_mem d_data, cl_mem o_data, int n, int k)
    {
        int localBufferSize = Sizeof.cl_int * m;
        int gx = k * m;
        long part_buf_size = ((long)Sizeof.cl_int * ((long)k * 2));
        int[] partial_sums = new int[k * 2];
        cl_mem p_data = CL.clCreateBuffer(context, CL.CL_MEM_READ_WRITE | CL.CL_MEM_COPY_HOST_PTR, part_buf_size, Pointer.to(partial_sums), null);
        Pointer src_data = Pointer.to(d_data);
        Pointer src_part = Pointer.to(p_data);

        Pointer dst_data = Pointer.to(o_data);

        clSetKernelArg(k_scan_int_multi_block_out, 0, Sizeof.cl_mem, src_data);
        clSetKernelArg(k_scan_int_multi_block_out, 1, Sizeof.cl_mem, dst_data);
        clSetKernelArg(k_scan_int_multi_block_out, 2, localBufferSize,null);
        clSetKernelArg(k_scan_int_multi_block_out, 3, Sizeof.cl_mem, src_part);
        clSetKernelArg(k_scan_int_multi_block_out, 4, Sizeof.cl_int, Pointer.to(new int[]{n}));

        clEnqueueNDRangeKernel(commandQueue, k_scan_int_multi_block_out, 1, null,
            new long[]{gx}, local_work_default, 0, null, null);

        int n2 = partial_sums.length;
        scan_int(p_data, n2);

        clSetKernelArg(k_complete_int_multi_block_out, 0, Sizeof.cl_mem, dst_data);
        clSetKernelArg(k_complete_int_multi_block_out, 1, localBufferSize,null);
        clSetKernelArg(k_complete_int_multi_block_out, 2, Sizeof.cl_mem, src_part);
        clSetKernelArg(k_complete_int_multi_block_out, 3, Sizeof.cl_int, Pointer.to(new int[]{n}));

        clEnqueueNDRangeKernel(commandQueue, k_complete_int_multi_block_out, 1, null,
            new long[]{gx}, local_work_default, 0, null, null);

        clReleaseMemObject(p_data);
    }

    private static int scan_single_block_candidates_out(cl_mem d_data, cl_mem o_data, int n)
    {
        int localBufferSize = Sizeof.cl_int * m;
        Pointer src_data = Pointer.to(d_data);
        Pointer dst_data = Pointer.to(o_data);

        int[] sz = new int[1];
        Pointer dst_size = Pointer.to(sz);
        cl_mem size_data = CL.clCreateBuffer(context, CL.CL_MEM_READ_WRITE, Sizeof.cl_int, null, null);
        Pointer src_size = Pointer.to(size_data);

        clSetKernelArg(k_scan_candidates_single_block, 0, Sizeof.cl_mem, src_data);
        clSetKernelArg(k_scan_candidates_single_block, 1, Sizeof.cl_mem, dst_data);
        clSetKernelArg(k_scan_candidates_single_block, 2, Sizeof.cl_mem, src_size);
        clSetKernelArg(k_scan_candidates_single_block, 3, localBufferSize,null);
        clSetKernelArg(k_scan_candidates_single_block, 4, Sizeof.cl_int, Pointer.to(new int[]{n}));

        clEnqueueNDRangeKernel(commandQueue, k_scan_candidates_single_block, 1, null,
            local_work_default, local_work_default, 0, null, null);

        clEnqueueReadBuffer(commandQueue, size_data, CL_TRUE, 0,
            Sizeof.cl_int, dst_size, 0, null, null);

        clReleaseMemObject(size_data);

        return sz[0];
    }

    private static int scan_multi_block_candidates_out(cl_mem d_data, cl_mem o_data, int n, int k)
    {
        int localBufferSize = Sizeof.cl_int * m;
        int gx = k * m;
        long part_buf_size = ((long)Sizeof.cl_int * ((long)k * 2));
        int[] partial_sums = new int[k * 2];
        cl_mem p_data = CL.clCreateBuffer(context, CL.CL_MEM_READ_WRITE | CL.CL_MEM_COPY_HOST_PTR, part_buf_size, Pointer.to(partial_sums), null);
        Pointer src_data = Pointer.to(d_data);
        Pointer src_part = Pointer.to(p_data);

        Pointer dst_data = Pointer.to(o_data);

        clSetKernelArg(k_scan_candidates_multi_block, 0, Sizeof.cl_mem, src_data);
        clSetKernelArg(k_scan_candidates_multi_block, 1, Sizeof.cl_mem, dst_data);
        clSetKernelArg(k_scan_candidates_multi_block, 2, localBufferSize,null);
        clSetKernelArg(k_scan_candidates_multi_block, 3, Sizeof.cl_mem, src_part);
        clSetKernelArg(k_scan_candidates_multi_block, 4, Sizeof.cl_int, Pointer.to(new int[]{n}));

        clEnqueueNDRangeKernel(commandQueue, k_scan_candidates_multi_block, 1, null,
            new long[]{gx}, local_work_default, 0, null, null);

        int n2 = partial_sums.length;
        scan_int(p_data, n2);

        int[] sz = new int[1];
        Pointer dst_size = Pointer.to(sz);
        cl_mem size_data = CL.clCreateBuffer(context, CL.CL_MEM_READ_WRITE, Sizeof.cl_int, null, null);
        Pointer src_size = Pointer.to(size_data);

        clSetKernelArg(k_complete_candidates_multi_block, 0, Sizeof.cl_mem, src_data);
        clSetKernelArg(k_complete_candidates_multi_block, 1, Sizeof.cl_mem, dst_data);
        clSetKernelArg(k_complete_candidates_multi_block, 2, Sizeof.cl_mem, src_size);
        clSetKernelArg(k_complete_candidates_multi_block, 3, localBufferSize,null);
        clSetKernelArg(k_complete_candidates_multi_block, 4, Sizeof.cl_mem, src_part);
        clSetKernelArg(k_complete_candidates_multi_block, 5, Sizeof.cl_int, Pointer.to(new int[]{n}));

        clEnqueueNDRangeKernel(commandQueue, k_complete_candidates_multi_block, 1, null,
            new long[]{gx}, local_work_default, 0, null, null);

        clEnqueueReadBuffer(commandQueue, size_data, CL_TRUE, 0,
            Sizeof.cl_int, dst_size, 0, null, null);

        clReleaseMemObject(p_data);
        clReleaseMemObject(size_data);

        return sz[0];
    }

    private static int scan_single_block_key(cl_mem d_data, int n)
    {
        int localBufferSize = Sizeof.cl_int * m;
        Pointer src_data = Pointer.to(d_data);

        int[] sz = new int[1];
        Pointer dst_size = Pointer.to(sz);
        cl_mem size_data = CL.clCreateBuffer(context, CL.CL_MEM_READ_WRITE, Sizeof.cl_int, null, null);
        Pointer src_size = Pointer.to(size_data);

        clSetKernelArg(k_scan_bounds_single_block, 0, Sizeof.cl_mem, src_data);
        clSetKernelArg(k_scan_bounds_single_block, 1, Sizeof.cl_mem, src_size);
        clSetKernelArg(k_scan_bounds_single_block, 2, localBufferSize,null);
        clSetKernelArg(k_scan_bounds_single_block, 3, Sizeof.cl_int, Pointer.to(new int[]{n}));

        clEnqueueNDRangeKernel(commandQueue, k_scan_bounds_single_block, 1, null,
            local_work_default, local_work_default, 0, null, null);

        clEnqueueReadBuffer(commandQueue, size_data, CL_TRUE, 0,
            Sizeof.cl_int, dst_size, 0, null, null);

        clReleaseMemObject(size_data);

        return sz[0];
    }

    private static int scan_multi_block_key(cl_mem d_data, int n, int k)
    {
        int localBufferSize = Sizeof.cl_int * m;
        int gx = k * m;
        long part_buf_size = ((long)Sizeof.cl_int * ((long)k * 2));
        int[] partial_sums = new int[k * 2];
        cl_mem p_data = CL.clCreateBuffer(context, CL.CL_MEM_READ_WRITE | CL.CL_MEM_COPY_HOST_PTR, part_buf_size, Pointer.to(partial_sums), null);
        Pointer src_data = Pointer.to(d_data);
        Pointer src_part = Pointer.to(p_data);

        clSetKernelArg(k_scan_bounds_multi_block, 0, Sizeof.cl_mem, src_data);
        clSetKernelArg(k_scan_bounds_multi_block, 1, localBufferSize,null);
        clSetKernelArg(k_scan_bounds_multi_block, 2, Sizeof.cl_mem, src_part);
        clSetKernelArg(k_scan_bounds_multi_block, 3, Sizeof.cl_int, Pointer.to(new int[]{n}));

        clEnqueueNDRangeKernel(commandQueue, k_scan_bounds_multi_block, 1, null,
            new long[]{gx}, local_work_default, 0, null, null);

        int n2 = partial_sums.length;
        scan_int(p_data, n2);

        int[] sz = new int[1];
        Pointer dst_size = Pointer.to(sz);
        cl_mem size_data = CL.clCreateBuffer(context, CL.CL_MEM_READ_WRITE, Sizeof.cl_int, null, null);
        Pointer src_size = Pointer.to(size_data);

        clSetKernelArg(k_complete_bounds_multi_block, 0, Sizeof.cl_mem, src_data);
        clSetKernelArg(k_complete_bounds_multi_block, 1, Sizeof.cl_mem, src_size);
        clSetKernelArg(k_complete_bounds_multi_block, 2, localBufferSize,null);
        clSetKernelArg(k_complete_bounds_multi_block, 3, Sizeof.cl_mem, src_part);
        clSetKernelArg(k_complete_bounds_multi_block, 4, Sizeof.cl_int, Pointer.to(new int[]{n}));

        clEnqueueNDRangeKernel(commandQueue, k_complete_bounds_multi_block, 1, null,
            new long[]{gx}, local_work_default, 0, null, null);

        clEnqueueReadBuffer(commandQueue, size_data, CL_TRUE, 0,
            Sizeof.cl_int, dst_size, 0, null, null);

        clReleaseMemObject(size_data);
        clReleaseMemObject(p_data);

        return sz[0];
    }

    //#endregion
}
