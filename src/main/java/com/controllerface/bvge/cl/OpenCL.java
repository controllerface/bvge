package com.controllerface.bvge.cl;

import com.controllerface.bvge.Main;
import com.controllerface.bvge.ecs.systems.physics.MemoryBuffer;
import com.controllerface.bvge.ecs.systems.physics.PhysicsBuffer;
import com.controllerface.bvge.ecs.systems.physics.SpatialPartition;
import org.jocl.*;
import org.joml.Matrix4f;

import java.util.*;

import static com.controllerface.bvge.cl.OpenCLUtils.*;
import static org.jocl.CL.*;
import static org.lwjgl.opengl.WGL.wglGetCurrentContext;
import static org.lwjgl.opengl.WGL.wglGetCurrentDC;

public class OpenCL
{
    private static final int COLLISION_SIZE = 2;

    private static final long FLAGS_WRITE_GPU       = CL_MEM_READ_WRITE;
    private static final long FLAGS_WRITE_CPU_COPY  = CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR;
    private static final long FLAGS_READ_CPU_COPY   = CL_MEM_READ_ONLY  | CL_MEM_COPY_HOST_PTR;

    static cl_command_queue commandQueue;
    static cl_context context;
    static cl_device_id[] device_ids;

    static String prag_int32_base_atomics   = read_src("pragma/int32_base_atomics.cl");

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
    static String func_project_circle       = read_src("functions/project_circle.cl");
    static String func_polygon_distance     = read_src("functions/polygon_distance.cl");
    static String func_edge_contact         = read_src("functions/edge_contact.cl");
    static String func_rotate_point         = read_src("functions/rotate_point.cl");
    static String func_angle_between        = read_src("functions/angle_between.cl");
    static String func_closest_point_circle = read_src("functions/closest_point_circle.cl");

    /**
     * CRUD
     */

    static String kern_gpu_crud = read_src("kernels/gpu_crud.cl");

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
    static String kern_prepare_edges        = read_src("kernels/prepare_edges.cl");
    static String kern_prepare_transforms   = read_src("kernels/prepare_transforms.cl");
    static String kern_prepare_bounds       = read_src("kernels/prepare_bounds.cl");
    static String kern_animate_hulls        = read_src("kernels/animate_hulls.cl");

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
    static String kn_update_accel                       = "update_accel";
    static String kn_rotate_hull                        = "rotate_hull";
    static String kn_read_position                      = "read_position";
    static String kn_create_point                       = "create_point";
    static String kn_create_edge                        = "create_edge";
    static String kn_create_hull                        = "create_hull";
    static String kn_create_vertex_reference            = "create_vertex_reference";
    static String kn_create_bone_reference              = "create_bone_reference";
    static String kn_create_bone                        = "create_bone";
    static String kn_prepare_edges                      = "prepare_edges";
    static String kn_prepare_transforms                 = "prepare_transforms";
    static String kn_prepare_bounds                     = "prepare_bounds";
    static String kn_animate_hulls                      = "animate_hulls";

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
    static cl_program p_gpu_crud;
    static cl_program p_prepare_edges;
    static cl_program p_prepare_transforms;
    static cl_program p_prepare_bounds;
    static cl_program p_animate_hulls;

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
    static cl_kernel k_update_accel;
    static cl_kernel k_rotate_hull;
    static cl_kernel k_read_position;
    static cl_kernel k_create_point;
    static cl_kernel k_create_edge;
    static cl_kernel k_create_hull;
    static cl_kernel k_create_vertex_reference;
    static cl_kernel k_create_bone_reference;
    static cl_kernel k_create_bone;
    static cl_kernel k_prepare_edges;
    static cl_kernel k_prepare_transforms;
    static cl_kernel k_prepare_bounds;
    static cl_kernel k_animate_hulls;




    // memory objects
    private static final HashMap<Integer, cl_mem> shared_mem = new LinkedHashMap<>();
    private static cl_mem mem_points;
    private static cl_mem mem_edges;
    private static cl_mem mem_transform;
    private static cl_mem mem_aabb;
    private static cl_mem mem_hull_acceleration;
    private static cl_mem mem_hull_rotation;
    private static cl_mem mem_hull_element_tables;
    private static cl_mem mem_hull_flags;
    private static cl_mem mem_aabb_index;
    private static cl_mem mem_aabb_key_bank;

    private static cl_mem mem_vertex_references;
    private static cl_mem mem_vertex_table;
    private static cl_mem mem_vertex_index;

    private static cl_mem mem_bone_references;
    private static cl_mem mem_bone_instances;
    private static cl_mem mem_bone_index;




    /**
     * During shutdown, these are used to release resources.
     */
    static List<cl_program> loaded_programs = new ArrayList<>();
    static List<cl_kernel> loaded_kernels = new ArrayList<>();

    private static PhysicsBuffer physicsBuffer;

    private static final Pointer ZERO_PATTERN = Pointer.to(new int[]{0});

    // these are re-calculated at startup to match the user's hardware
    private static long wx = 0;
    private static long m = 0;
    private static long[] local_work_default = arg_long(0);

    // pre-made size array, used for kernels that have a single work item
    private static final long[] global_single_size = arg_long(1);

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
        int[] numPlatformsArray = new int[1];
        clGetPlatformIDs(0, null, numPlatformsArray);
        int numPlatforms = numPlatformsArray[0];

        // Obtain a platform ID
        cl_platform_id[] platforms = new cl_platform_id[numPlatforms];
        clGetPlatformIDs(platforms.length, platforms, null);
        cl_platform_id platform = platforms[platformIndex];

        // note: this portion is windows specific
        var dc = wglGetCurrentDC();
        var ctx = wglGetCurrentContext();

        // todo: add linux code path, should look something like this
//        var ctx = glXGetCurrentContext();
//        var dc = glXGetCurrentDrawable();
        //contextProperties.addProperty(CL_GLX_DISPLAY_KHR, dc);
        // Initialize the context properties

        cl_context_properties contextProperties = new cl_context_properties();
        contextProperties.addProperty(CL_CONTEXT_PLATFORM, platform);
        contextProperties.addProperty(CL_GL_CONTEXT_KHR, ctx);
        contextProperties.addProperty(CL_WGL_HDC_KHR, dc);

        // Obtain the number of devices for the platform
        int[] numDevicesArray = new int[1];
        clGetDeviceIDs(platform, deviceType, 0, null, numDevicesArray);
        int numDevices = numDevicesArray[0];

        // Obtain a device ID
        cl_device_id[] devices = new cl_device_id[numDevices];
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

    private static void cl_read_buffer(cl_mem src, long size, Pointer dst)
    {
        clEnqueueReadBuffer(commandQueue, src, CL_TRUE, 0, size, dst,
            0, null, null);
    }

    private static cl_mem cl_new_buffer(long flags, long size)
    {
       return CL.clCreateBuffer(context, flags, size, null, null);
    }

    private static cl_mem cl_new_buffer(long flags, long size, Pointer src)
    {
        return CL.clCreateBuffer(context, flags, size, src, null);
    }

    private static void cl_zero_buffer(cl_mem buffer, long buffer_size)
    {
        clEnqueueFillBuffer(commandQueue, buffer, ZERO_PATTERN, 1, 0, buffer_size,
            0, null, null);
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

    private static void k_call(cl_kernel kernel, long[] global_work_size)
    {
        clEnqueueNDRangeKernel(commandQueue, kernel, 1, null,
            global_work_size, null, 0, null, null);
    }

    private static void k_call(cl_kernel kernel, long[] global_work_size, long[] local_work_size)
    {
        clEnqueueNDRangeKernel(commandQueue, kernel, 1, null,
            global_work_size, local_work_size, 0, null, null);
    }

    private static void gl_acquire(cl_mem mem)
    {
        clEnqueueAcquireGLObjects(commandQueue, 1, new cl_mem[]{mem}, 0, null, null);
    }

    private static void gl_release(cl_mem mem)
    {
        clEnqueueReleaseGLObjects(commandQueue, 1, new cl_mem[]{ mem}, 0, null, null);
    }

    public static long[] arg_long(long arg)
    {
        return new long[]{ arg };
    }

    public static int[] arg_int(int arg)
    {
        return new int[]{ arg };
    }

    public static int[] arg_int2(int x, int y)
    {
        return new int[]{ x, y };
    }

    public static int[] arg_int4(int x, int y, int z, int w)
    {
        return new int[]{ x, y, z, w };
    }

    public static float[] arg_float(float arg)
    {
        return new float[]{ arg };
    }

    public static float[] arg_float2(float x, float y)
    {
        return new float[]{ x, y };
    }

    public static float[] arg_float4(float x, float y, float z, float w)
    {
        return new float[]{ x, y, z, w };
    }

    public static float[] arg_float16(float s0, float s1, float s2, float s3,
                                      float s4, float s5, float s6, float s7,
                                      float s8, float s9, float sA, float sB,
                                      float sC, float sD, float sE, float sF)
    {
        return new float[]{ s0, s1, s2, s3,
                            s4, s5, s6, s7,
                            s8, s9, sA, sB,
                            sC, sD, sE, sF };
    }

    public static float[] arg_float16_matrix(Matrix4f matrix)
    {
        return new float[]
            {
                matrix.m00(), matrix.m01(), matrix.m02(), matrix.m03(),
                matrix.m10(), matrix.m11(), matrix.m12(), matrix.m13(),
                matrix.m20(), matrix.m21(), matrix.m22(), matrix.m23(),
                matrix.m30(), matrix.m31(), matrix.m32(), matrix.m33()
            };
    }

    public static int work_group_count(int n)
    {
        return (int) Math.ceil((float)n / (float)m);
    }

    public static void init(int max_hulls, int max_points)
    {
        device_ids = device_init();

        var device = device_ids[0];

        System.out.println("-------- OPEN CL DEVICE -----------");
        System.out.println(getString(device, CL_DEVICE_VENDOR));
        System.out.println(getString(device, CL_DEVICE_NAME));
        System.out.println(getString(device, CL_DRIVER_VERSION));
        System.out.println("-----------------------------------\n");

        wx = getSize(device, CL_DEVICE_MAX_WORK_GROUP_SIZE);
        m = wx * 2;
        local_work_default = new long[]{ wx };

        //OpenCLUtils.debugDeviceDetails(device_ids);

        /*
         * Programs
         */
        p_sat_collide = cl_p(
            func_angle_between,
            func_closest_point_circle,
            func_project_circle,
            func_project_polygon,
            func_polygon_distance,
            func_edge_contact,
            kern_sat_collide);

        p_aabb_collide = cl_p(
            func_do_bounds_intersect,
            func_calculate_key_index,
            kern_aabb_collide);

        p_integrate = cl_p(func_angle_between,
            func_rotate_point,
            func_is_in_bounds,
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

        p_gpu_crud = cl_p(func_rotate_point, kern_gpu_crud);

        p_prepare_edges = cl_p(kern_prepare_edges);

        p_prepare_transforms = cl_p(kern_prepare_transforms);

        p_prepare_bounds = cl_p(kern_prepare_bounds);

        p_animate_hulls = cl_p(kern_animate_hulls);

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
        k_update_accel                      = cl_k(p_gpu_crud, kn_update_accel);
        k_rotate_hull                       = cl_k(p_gpu_crud, kn_rotate_hull);
        k_read_position                     = cl_k(p_gpu_crud, kn_read_position);
        k_create_hull                       = cl_k(p_gpu_crud, kn_create_hull);
        k_create_point                      = cl_k(p_gpu_crud, kn_create_point);
        k_create_edge                       = cl_k(p_gpu_crud, kn_create_edge);
        k_create_bone_reference             = cl_k(p_gpu_crud, kn_create_bone_reference);
        k_create_vertex_reference           = cl_k(p_gpu_crud, kn_create_vertex_reference);
        k_create_bone                       = cl_k(p_gpu_crud, kn_create_bone);
        k_prepare_edges                     = cl_k(p_prepare_edges, kn_prepare_edges);
        k_prepare_transforms                = cl_k(p_prepare_transforms, kn_prepare_transforms);
        k_prepare_bounds                    = cl_k(p_prepare_bounds, kn_prepare_bounds);
        k_animate_hulls                     = cl_k(p_animate_hulls, kn_animate_hulls);

        // init physics buffers here

        int transform_mem_size        = max_hulls * Sizeof.cl_float4;
        int accleration_mem_size      = max_hulls * Sizeof.cl_float2;
        int rotation_mem_size         = max_hulls * Sizeof.cl_float2;
        int element_table_mem_size    = max_hulls * Sizeof.cl_int4;
        int flags_mem_size            = max_hulls * Sizeof.cl_int2;
        int bounding_box_mem_size     = max_hulls * Sizeof.cl_float4;
        int spatial_index_mem_size    = max_hulls * Sizeof.cl_int4;
        int spatial_key_bank_mem_size = max_hulls * Sizeof.cl_int2;
        int points_mem_size           = max_points * Sizeof.cl_float4;
        int edges_mem_size            = max_points * Sizeof.cl_float4;

        int vertex_table_mem_size     = max_points * Sizeof.cl_int2;
        int vertex_reference_mem_size = max_points * Sizeof.cl_float2;
        int vertex_index_mem_size     = max_points * Sizeof.cl_int;

        int bone_reference_mem_size   = max_points * Sizeof.cl_float16;
        int bone_instance_mem_size    = max_points * Sizeof.cl_float16;
        int bone_index_mem_size       = max_points * Sizeof.cl_int;

        int total = transform_mem_size
            + accleration_mem_size
            + rotation_mem_size
            + element_table_mem_size
            + flags_mem_size
            + bounding_box_mem_size
            + spatial_index_mem_size
            + spatial_key_bank_mem_size
            + points_mem_size
            + edges_mem_size
            + vertex_table_mem_size
            + vertex_reference_mem_size
            + vertex_index_mem_size
            + bone_reference_mem_size
            + bone_instance_mem_size
            + bone_index_mem_size;

        System.out.println("------------- BUFFERS -------------");
        System.out.println("points            : " + points_mem_size);
        System.out.println("edges             : " + edges_mem_size);
        System.out.println("transforms        : " + transform_mem_size);
        System.out.println("acceleration      : " + accleration_mem_size);
        System.out.println("rotation          : " + rotation_mem_size);
        System.out.println("element table     : " + element_table_mem_size);
        System.out.println("flags             : " + flags_mem_size);
        System.out.println("bounding box      : " + bounding_box_mem_size);
        System.out.println("spatial index     : " + spatial_index_mem_size);
        System.out.println("spatial key bank  : " + spatial_key_bank_mem_size);
        System.out.println("vertex table      : " + vertex_table_mem_size);
        System.out.println("vertex references : " + vertex_reference_mem_size);
        System.out.println("vertex index      : " + vertex_index_mem_size);
        System.out.println("bone references   : " + bone_reference_mem_size);
        System.out.println("bone instances    : " + bone_instance_mem_size);
        System.out.println("bone index        : " + bone_index_mem_size);
        System.out.println("=====================================");
        System.out.println(" Total (Bytes)    : " + total);
        System.out.println("               KB : " + ((float)total / 1024f));
        System.out.println("               MB : " + ((float)total / 1024f / 1024f));
        System.out.println("               GB : " + ((float)total / 1024f / 1024f / 1024f));
        System.out.println("-----------------------------------\n");

        mem_hull_acceleration         = cl_new_buffer(FLAGS_WRITE_GPU, accleration_mem_size);
        mem_hull_rotation             = cl_new_buffer(FLAGS_WRITE_GPU, rotation_mem_size);
        mem_hull_element_tables       = cl_new_buffer(FLAGS_WRITE_GPU, element_table_mem_size);
        mem_hull_flags                = cl_new_buffer(FLAGS_WRITE_GPU, flags_mem_size);
        mem_aabb_index                = cl_new_buffer(FLAGS_WRITE_GPU, spatial_index_mem_size);
        mem_aabb_key_bank             = cl_new_buffer(FLAGS_WRITE_GPU, spatial_key_bank_mem_size);
        mem_transform                 = cl_new_buffer(FLAGS_WRITE_GPU, transform_mem_size);
        mem_aabb                      = cl_new_buffer(FLAGS_WRITE_GPU, bounding_box_mem_size);
        mem_points                    = cl_new_buffer(FLAGS_WRITE_GPU, points_mem_size);
        mem_edges                     = cl_new_buffer(FLAGS_WRITE_GPU, edges_mem_size);
        mem_vertex_table              = cl_new_buffer(FLAGS_WRITE_GPU, vertex_table_mem_size);
        mem_vertex_references         = cl_new_buffer(FLAGS_WRITE_GPU, vertex_reference_mem_size);
        mem_vertex_index              = cl_new_buffer(FLAGS_WRITE_GPU, vertex_index_mem_size);
        mem_bone_references           = cl_new_buffer(FLAGS_WRITE_GPU, bone_reference_mem_size);
        mem_bone_instances            = cl_new_buffer(FLAGS_WRITE_GPU, bone_instance_mem_size);
        mem_bone_index                = cl_new_buffer(FLAGS_WRITE_GPU, bone_index_mem_size);

        cl_zero_buffer(mem_hull_acceleration, accleration_mem_size);
        cl_zero_buffer(mem_hull_rotation, rotation_mem_size);
        cl_zero_buffer(mem_hull_element_tables, element_table_mem_size);
        cl_zero_buffer(mem_hull_flags, flags_mem_size);
        cl_zero_buffer(mem_aabb_index, spatial_index_mem_size);
        cl_zero_buffer(mem_aabb_key_bank, spatial_key_bank_mem_size);
        cl_zero_buffer(mem_transform, transform_mem_size);
        cl_zero_buffer(mem_aabb, bounding_box_mem_size);
        cl_zero_buffer(mem_points, points_mem_size);
        cl_zero_buffer(mem_edges, edges_mem_size);
        cl_zero_buffer(mem_vertex_table, vertex_table_mem_size);
        cl_zero_buffer(mem_vertex_references, vertex_reference_mem_size);
        cl_zero_buffer(mem_vertex_index, vertex_index_mem_size);
        cl_zero_buffer(mem_bone_references, bone_reference_mem_size);
        cl_zero_buffer(mem_bone_instances, bone_instance_mem_size);
        cl_zero_buffer(mem_bone_index, bone_index_mem_size);
    }

    public static void destroy()
    {
        //clReleaseMemObject(mem_points);
        //clReleaseMemObject(mem_transform);
        //clReleaseMemObject(mem_aabb);
        //clReleaseMemObject(mem_edges);
        // todo: destroy more/track it better
        //shared_mem.values().forEach(CL::clReleaseMemObject);

        loaded_programs.forEach(CL::clReleaseProgram);
        loaded_kernels.forEach(CL::clReleaseKernel);
        clReleaseCommandQueue(commandQueue);
        clReleaseContext(context);
    }

    // GL Interop

    /**
     * Called from Gl code to share a buffer object with CL. This allows kernels to process
     * buffer data for shaders. The size of the shared data is determined automatically by the
     * vboID and is set in the GL context.
     *
     * @param vboID GL buffer ID of the data to share.
     */
    public static void share_memory(int vboID)
    {
        cl_mem vbo_mem = clCreateFromGLBuffer(context, FLAGS_WRITE_GPU, vboID, null);
        shared_mem.put(vboID, vbo_mem);
    }






    /**
     * Transfers a subset of all bounding boxes from CL memory into GL memory, converting the bounds
     * into a vertex structure that can be rendered as a line loop.
     *
     * @param vboID
     * @param vboOffset
     * @param batchSize
     */
    public static void GL_bounds(int vboID, int vboOffset, int batchSize)
    {
        var vbo_mem = shared_mem.get(vboID);
        long[] global_work_size = arg_long(batchSize);
        long[] edge_offset = arg_long(vboOffset);

        clSetKernelArg(k_prepare_bounds, 0, Sizeof.cl_mem, physicsBuffer.bounds.pointer());
        clSetKernelArg(k_prepare_bounds, 1, Sizeof.cl_mem, Pointer.to(vbo_mem));
        clSetKernelArg(k_prepare_bounds, 2, Sizeof.cl_int, Pointer.to(edge_offset));

        gl_acquire(vbo_mem);
        k_call(k_prepare_bounds, global_work_size);
        gl_release(vbo_mem);
    }

    public static void GL_edges(int vboID, int vboOffset, int batchSize)
    {
        var vbo_mem = shared_mem.get(vboID);
        long[] global_work_size = arg_long(batchSize);
        long[] edge_offset = arg_long(vboOffset);

        clSetKernelArg(k_prepare_edges, 0, Sizeof.cl_mem, physicsBuffer.points.pointer());
        clSetKernelArg(k_prepare_edges, 1, Sizeof.cl_mem, physicsBuffer.edges.pointer());
        clSetKernelArg(k_prepare_edges, 2, Sizeof.cl_mem, Pointer.to(vbo_mem));
        clSetKernelArg(k_prepare_edges, 3, Sizeof.cl_int, Pointer.to(edge_offset));

        gl_acquire(vbo_mem);
        k_call(k_prepare_edges, global_work_size);
        gl_release(vbo_mem);
    }

    public static void GL_transforms(int vboID, int transforms_id, int size)
    {
        var vbo_mem = shared_mem.get(vboID);
        var vbo_mem2 = shared_mem.get(transforms_id);
        long[] global_work_size = arg_long(size);

        clSetKernelArg(k_prepare_transforms, 0, Sizeof.cl_mem, Pointer.to(mem_transform));
        clSetKernelArg(k_prepare_transforms, 1, Sizeof.cl_mem, Pointer.to(mem_hull_rotation));
        clSetKernelArg(k_prepare_transforms, 2, Sizeof.cl_mem, Pointer.to(vbo_mem));
        clSetKernelArg(k_prepare_transforms, 3, Sizeof.cl_mem, Pointer.to(vbo_mem2));

        gl_acquire(vbo_mem);
        gl_acquire(vbo_mem2);
        k_call(k_prepare_transforms, global_work_size);
        gl_release(vbo_mem);
        gl_release(vbo_mem2);
    }





    public static void initPhysicsBuffer(PhysicsBuffer physicsBuffer)
    {
        // todo: the hull and bounds data needs to be sliced up into smaller arrays for better
        //  efficiency. Where possible, 4 dimensional vectors are preferable, followed by
        //  2 dimensional vectors, and then scalars.

        physicsBuffer.bounds = new MemoryBuffer(mem_aabb);
        physicsBuffer.hulls = new MemoryBuffer(mem_transform);
        physicsBuffer.points = new MemoryBuffer(mem_points);
        physicsBuffer.edges  = new MemoryBuffer(mem_edges);
        physicsBuffer.elements = new MemoryBuffer(mem_hull_element_tables);
        physicsBuffer.acceleration = new MemoryBuffer(mem_hull_acceleration);
        physicsBuffer.rotation = new MemoryBuffer(mem_hull_rotation);
        physicsBuffer.flags = new MemoryBuffer(mem_hull_flags);
        physicsBuffer.index = new MemoryBuffer(mem_aabb_index);
        physicsBuffer.bank = new MemoryBuffer(mem_aabb_key_bank);
    }






    public static void create_point(int point_index, float pos_x, float pos_y, float prv_x, float prv_y, int v, int b)
    {
        var pnt_index = Pointer.to(arg_int(point_index));
        var pnt_point = Pointer.to(arg_float4(pos_x, pos_y, prv_x, prv_y));
        var pnt_table = Pointer.to(arg_int2(v, b));

        clSetKernelArg(k_create_point, 0, Sizeof.cl_mem, Pointer.to(mem_points));
        clSetKernelArg(k_create_point, 1, Sizeof.cl_mem, Pointer.to(mem_vertex_table));
        clSetKernelArg(k_create_point, 2, Sizeof.cl_int, pnt_index);
        clSetKernelArg(k_create_point, 3, Sizeof.cl_float4, pnt_point);
        clSetKernelArg(k_create_point, 4, Sizeof.cl_int2, pnt_table);

        k_call(k_create_point, global_single_size);
    }

    public static void create_edge(int edge_index, float p1, float p2, float l, int flags)
    {
        var pnt_index = Pointer.to(arg_int(edge_index));
        var pnt_edge = Pointer.to(arg_float4(p1, p2, l, flags));

        clSetKernelArg(k_create_edge, 0, Sizeof.cl_mem, Pointer.to(mem_edges));
        clSetKernelArg(k_create_edge, 1, Sizeof.cl_int, pnt_index);
        clSetKernelArg(k_create_edge, 2, Sizeof.cl_float4, pnt_edge);

        k_call(k_create_edge, global_single_size);
    }


    public static void create_vertex_reference(int vert_ref_index, float x, float y)
    {
        var pnt_index = Pointer.to(arg_int(vert_ref_index));
        var pnt_vert_ref = Pointer.to(arg_float2(x, y));

        clSetKernelArg(k_create_vertex_reference, 0, Sizeof.cl_mem, Pointer.to(mem_vertex_references));
        clSetKernelArg(k_create_vertex_reference, 1, Sizeof.cl_int, pnt_index);
        clSetKernelArg(k_create_vertex_reference, 2, Sizeof.cl_float2, pnt_vert_ref);

        k_call(k_create_vertex_reference, global_single_size);
    }

    public static void create_bone_reference(int bone_ref_index, float[] matrix)
    {
        var pnt_index = Pointer.to(arg_int(bone_ref_index));
        var pnt_bone_ref = Pointer.to(matrix);

        clSetKernelArg(k_create_bone_reference, 0, Sizeof.cl_mem, Pointer.to(mem_bone_references));
        clSetKernelArg(k_create_bone_reference, 1, Sizeof.cl_int, pnt_index);
        clSetKernelArg(k_create_bone_reference, 2, Sizeof.cl_float16, pnt_bone_ref);

        k_call(k_create_bone_reference, global_single_size);
    }

    public static void create_bone(int bone_index, int bone_ref_index, float[] matrix)
    {
        var pnt_index = Pointer.to(arg_int(bone_index));
        var pnt_ref_index = Pointer.to(arg_int(bone_ref_index));
        var pnt_bone_ref = Pointer.to(matrix);

        clSetKernelArg(k_create_bone, 0, Sizeof.cl_mem, Pointer.to(mem_bone_instances));
        clSetKernelArg(k_create_bone, 1, Sizeof.cl_mem, Pointer.to(mem_bone_index));
        clSetKernelArg(k_create_bone, 2, Sizeof.cl_int, pnt_index);
        clSetKernelArg(k_create_bone, 3, Sizeof.cl_float16, pnt_bone_ref);
        clSetKernelArg(k_create_bone, 4, Sizeof.cl_int, pnt_ref_index);

        k_call(k_create_bone, global_single_size);
    }


    public static void create_hull(int hull_index, float[] hull, float[] rotation, int[] table, int[] flags)
    {
        var pnt_index = Pointer.to(arg_int(hull_index));
        var pnt_flags = Pointer.to(flags);
        var pnt_table = Pointer.to(table);
        var pnt_rotation = Pointer.to(rotation);
        var pnt_hull = Pointer.to(hull);

        clSetKernelArg(k_create_hull, 0, Sizeof.cl_mem, Pointer.to(mem_transform));
        clSetKernelArg(k_create_hull, 1, Sizeof.cl_mem, Pointer.to(mem_hull_rotation));
        clSetKernelArg(k_create_hull, 2, Sizeof.cl_mem, Pointer.to(mem_hull_element_tables));
        clSetKernelArg(k_create_hull, 3, Sizeof.cl_mem, Pointer.to(mem_hull_flags));
        clSetKernelArg(k_create_hull, 4, Sizeof.cl_int, pnt_index);
        clSetKernelArg(k_create_hull, 5, Sizeof.cl_float4, pnt_hull);
        clSetKernelArg(k_create_hull, 6, Sizeof.cl_float2, pnt_rotation);
        clSetKernelArg(k_create_hull, 7, Sizeof.cl_int4, pnt_table);
        clSetKernelArg(k_create_hull, 8, Sizeof.cl_int2, pnt_flags);

        k_call(k_create_hull, global_single_size);
    }

    public static void update_accel(int hull_index, float acc_x, float acc_y)
    {
        var pnt_index = Pointer.to(arg_int(hull_index));
        var pnt_acc = Pointer.to(arg_float2(acc_x, acc_y));

        clSetKernelArg(k_update_accel, 0, Sizeof.cl_mem, Pointer.to(mem_hull_acceleration));
        clSetKernelArg(k_update_accel, 1, Sizeof.cl_int, pnt_index);
        clSetKernelArg(k_update_accel, 2, Sizeof.cl_float2, pnt_acc);

        k_call(k_update_accel, global_single_size);
    }

    public static void rotate_hull(int hull_index, float angle)
    {
        var pnt_index = Pointer.to(arg_int(hull_index));
        var pnt_angle = Pointer.to(arg_float(angle));

        clSetKernelArg(k_rotate_hull, 0, Sizeof.cl_mem, physicsBuffer.hulls.pointer());
        clSetKernelArg(k_rotate_hull, 1, Sizeof.cl_mem, physicsBuffer.elements.pointer());
        clSetKernelArg(k_rotate_hull, 2, Sizeof.cl_mem, physicsBuffer.points.pointer());
        clSetKernelArg(k_rotate_hull, 3, Sizeof.cl_int, pnt_index);
        clSetKernelArg(k_rotate_hull, 4, Sizeof.cl_float, pnt_angle);

        k_call(k_rotate_hull, global_single_size);
    }

    public static float[] read_position(int hull_index)
    {
        if (physicsBuffer == null) return null;

        int[] index = arg_int(hull_index);

        cl_mem result_data = cl_new_buffer(FLAGS_WRITE_GPU, Sizeof.cl_float2);
        cl_zero_buffer(result_data, Sizeof.cl_float2);
        Pointer src_result = Pointer.to(result_data);

        clSetKernelArg(k_read_position, 0, Sizeof.cl_mem, physicsBuffer.hulls.pointer());
        clSetKernelArg(k_read_position, 1, Sizeof.cl_float2, src_result);
        clSetKernelArg(k_read_position, 2, Sizeof.cl_int, Pointer.to(index));

        k_call(k_read_position, global_single_size);

        float[] result = arg_float2(0, 0);
        Pointer dst_result = Pointer.to(result);
        cl_read_buffer(result_data, Sizeof.cl_float2, dst_result);
        clReleaseMemObject(result_data);

        return result;
    }

    public static void setPhysicsBuffer(PhysicsBuffer physicsBuffer)
    {
        OpenCL.physicsBuffer = physicsBuffer;
    }


    //#region Physics Simulation


    public static void animate_hulls()
    {
        long[] global_work_size = new long[]{Main.Memory.point_count()};

        clSetKernelArg(k_animate_hulls, 0, Sizeof.cl_mem, Pointer.to(mem_points));
        clSetKernelArg(k_animate_hulls, 1, Sizeof.cl_mem, Pointer.to(mem_transform));
        clSetKernelArg(k_animate_hulls, 2, Sizeof.cl_mem, Pointer.to(mem_hull_flags));
        clSetKernelArg(k_animate_hulls, 3, Sizeof.cl_mem, Pointer.to(mem_vertex_table));
        clSetKernelArg(k_animate_hulls, 4, Sizeof.cl_mem, Pointer.to(mem_vertex_references));
        clSetKernelArg(k_animate_hulls, 5, Sizeof.cl_mem, Pointer.to(mem_bone_instances));

        k_call(k_animate_hulls, global_work_size);
    }



    public static void integrate(float delta_time, SpatialPartition spatialPartition)
    {
        long[] global_work_size = new long[]{Main.Memory.hull_count()};
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

        Pointer srcArgs = Pointer.to(args);

        long size = Sizeof.cl_float * args.length;
        cl_mem argMem = cl_new_buffer(FLAGS_READ_CPU_COPY, size, srcArgs);

        clSetKernelArg(k_integrate, 0, Sizeof.cl_mem, Pointer.to(physicsBuffer.hulls.memory()));
        clSetKernelArg(k_integrate, 1, Sizeof.cl_mem, Pointer.to(physicsBuffer.elements.memory()));
        clSetKernelArg(k_integrate, 2, Sizeof.cl_mem, Pointer.to(physicsBuffer.acceleration.memory()));
        clSetKernelArg(k_integrate, 3, Sizeof.cl_mem, Pointer.to(physicsBuffer.rotation.memory()));
        clSetKernelArg(k_integrate, 4, Sizeof.cl_mem, Pointer.to(physicsBuffer.points.memory()));
        clSetKernelArg(k_integrate, 5, Sizeof.cl_mem, Pointer.to(physicsBuffer.bounds.memory()));
        clSetKernelArg(k_integrate, 6, Sizeof.cl_mem, Pointer.to(physicsBuffer.index.memory()));
        clSetKernelArg(k_integrate, 7, Sizeof.cl_mem, Pointer.to(physicsBuffer.bank.memory()));
        clSetKernelArg(k_integrate, 8, Sizeof.cl_mem, Pointer.to(physicsBuffer.flags.memory()));
        clSetKernelArg(k_integrate, 9, Sizeof.cl_mem, Pointer.to(argMem));

        k_call(k_integrate, global_work_size);

        clReleaseMemObject(argMem);
    }

    public static void calculate_bank_offsets(SpatialPartition spatialPartition)
    {
        int n = Main.Memory.hull_count();
        int bank_size = scan_key_bounds(physicsBuffer.bank.memory(), n);
        spatialPartition.resizeBank(bank_size);
    }

    public static void generate_keys(SpatialPartition spatialPartition)
    {
        if (spatialPartition.getKey_bank_size() < 1)
        {
            return;
        }
        int n = Main.Memory.hull_count();
        long bank_buf_size = (long)Sizeof.cl_int * spatialPartition.getKey_bank_size();
        long counts_buf_size = (long)Sizeof.cl_int * spatialPartition.getDirectoryLength();

        cl_mem bank_data = cl_new_buffer(FLAGS_WRITE_GPU, bank_buf_size);
        cl_mem counts_data = cl_new_buffer(FLAGS_WRITE_GPU, counts_buf_size);
        cl_zero_buffer(counts_data, counts_buf_size);

        Pointer src_bank = Pointer.to(bank_data);
        Pointer src_counts = Pointer.to(counts_data);
        Pointer src_kb_len = Pointer.to(arg_int(spatialPartition.getKey_bank_size()));
        Pointer src_kc_len = Pointer.to(arg_int(spatialPartition.getDirectoryLength()));
        Pointer src_x_subs = Pointer.to(arg_int(spatialPartition.getX_subdivisions()));

        // key counts get transferred out right now for use in the spatial index renderer
        // todo: remove this transfer after moving over to CL/GL interop renderer
        physicsBuffer.key_counts = new MemoryBuffer(counts_data);
        physicsBuffer.key_bank = new MemoryBuffer(bank_data);

        // pass in arguments
        clSetKernelArg(k_generate_keys, 0, Sizeof.cl_mem, physicsBuffer.index.pointer());
        clSetKernelArg(k_generate_keys, 1, Sizeof.cl_mem, physicsBuffer.bank.pointer());
        clSetKernelArg(k_generate_keys, 2, Sizeof.cl_mem, src_bank);
        clSetKernelArg(k_generate_keys, 3, Sizeof.cl_mem, src_counts);
        clSetKernelArg(k_generate_keys, 4, Sizeof.cl_int, src_x_subs);
        clSetKernelArg(k_generate_keys, 5, Sizeof.cl_int, src_kb_len);
        clSetKernelArg(k_generate_keys, 6, Sizeof.cl_int, src_kc_len);

        k_call(k_generate_keys, arg_long(n));
    }

    public static void calculate_map_offsets(SpatialPartition spatialPartition)
    {
        int n = spatialPartition.getDirectoryLength();
        long data_buf_size = (long)Sizeof.cl_int * n;
        cl_mem o_data = cl_new_buffer(FLAGS_WRITE_GPU, data_buf_size);
        physicsBuffer.key_offsets = new MemoryBuffer(o_data);
        scan_int_out(physicsBuffer.key_counts.memory(), o_data, n);
    }

    public static void build_key_map(SpatialPartition spatialPartition)
    {
        int n = Main.Memory.hull_count();
        long map_buf_size = (long)Sizeof.cl_int * spatialPartition.getKey_map_size();
        long counts_buf_size = (long)Sizeof.cl_int * spatialPartition.getDirectoryLength();

        cl_mem map_data = cl_new_buffer(FLAGS_WRITE_GPU, map_buf_size);
        cl_mem counts_data = cl_new_buffer(FLAGS_WRITE_GPU, counts_buf_size);

        // the counts buffer needs to start off filled with all zeroes
        cl_zero_buffer(counts_data, counts_buf_size);

        Pointer src_map = Pointer.to(map_data);
        Pointer src_counts = Pointer.to(counts_data);
        Pointer src_x_subs = Pointer.to(arg_int(spatialPartition.getX_subdivisions()));
        Pointer src_c_len = Pointer.to(arg_int(spatialPartition.getDirectoryLength()));

        physicsBuffer.key_map = new MemoryBuffer(map_data);

        clSetKernelArg(k_build_key_map, 0, Sizeof.cl_mem, physicsBuffer.index.pointer());
        clSetKernelArg(k_build_key_map, 1, Sizeof.cl_mem, physicsBuffer.bank.pointer());
        clSetKernelArg(k_build_key_map, 2, Sizeof.cl_mem, src_map);
        clSetKernelArg(k_build_key_map, 3, Sizeof.cl_mem, physicsBuffer.key_offsets.pointer());
        clSetKernelArg(k_build_key_map, 4, Sizeof.cl_mem, src_counts);
        clSetKernelArg(k_build_key_map, 5, Sizeof.cl_int, src_x_subs );
        clSetKernelArg(k_build_key_map, 6, Sizeof.cl_int, src_c_len);

        k_call(k_build_key_map, arg_long(n));

        clReleaseMemObject(counts_data);
    }

    public static void locate_in_bounds(SpatialPartition spatialPartition)
    {
        int n = Main.Memory.hull_count();

        // step 1: locate objects that are within bounds
        int x_subdivisions = spatialPartition.getX_subdivisions();
        physicsBuffer.x_sub_divisions = Pointer.to(arg_int(x_subdivisions));
        physicsBuffer.key_count_length = Pointer.to(arg_int(spatialPartition.getDirectoryLength()));

        long inbound_buf_size = (long)Sizeof.cl_int * n;
        cl_mem inbound_data = cl_new_buffer(FLAGS_WRITE_GPU, inbound_buf_size);

        physicsBuffer.in_bounds = new MemoryBuffer(inbound_data);

        int[] size = arg_int(0);
        Pointer dst_size = Pointer.to(size);
        cl_mem size_data = cl_new_buffer(FLAGS_WRITE_CPU_COPY, Sizeof.cl_int, dst_size);
        Pointer src_size = Pointer.to(size_data);

        clSetKernelArg(k_locate_in_bounds, 0, Sizeof.cl_mem, physicsBuffer.bank.pointer());
        clSetKernelArg(k_locate_in_bounds, 1, Sizeof.cl_mem, physicsBuffer.in_bounds.pointer());
        clSetKernelArg(k_locate_in_bounds, 2, Sizeof.cl_mem, src_size);

        k_call(k_locate_in_bounds, arg_long(n));

        cl_read_buffer(size_data, Sizeof.cl_int, dst_size);

        clReleaseMemObject(size_data);

        physicsBuffer.set_candidate_buffer_count(size[0]);
    }

    public static void count_candidates()
    {
        long cand_buf_size = (long)Sizeof.cl_int2 * physicsBuffer.get_candidate_buffer_count();
        cl_mem cand_data = cl_new_buffer(FLAGS_WRITE_GPU, cand_buf_size);
        physicsBuffer.candidate_counts = new MemoryBuffer(cand_data);

        clSetKernelArg(k_count_candidates, 0, Sizeof.cl_mem, physicsBuffer.bank.pointer());
        clSetKernelArg(k_count_candidates, 1, Sizeof.cl_mem, physicsBuffer.in_bounds.pointer());
        clSetKernelArg(k_count_candidates, 2, Sizeof.cl_mem, physicsBuffer.key_bank.pointer());
        clSetKernelArg(k_count_candidates, 3, Sizeof.cl_mem, physicsBuffer.key_counts.pointer());
        clSetKernelArg(k_count_candidates, 4, Sizeof.cl_mem, physicsBuffer.candidate_counts.pointer());
        clSetKernelArg(k_count_candidates, 5, Sizeof.cl_int, physicsBuffer.x_sub_divisions);
        clSetKernelArg(k_count_candidates, 6, Sizeof.cl_int, physicsBuffer.key_count_length);

        k_call(k_count_candidates, arg_long(physicsBuffer.get_candidate_buffer_count()));
    }

    public static void count_matches()
    {
        int n = physicsBuffer.get_candidate_buffer_count();
        long offset_buf_size = (long)Sizeof.cl_int * n;
        cl_mem offset_data = cl_new_buffer(FLAGS_WRITE_GPU, offset_buf_size);
        physicsBuffer.candidate_offsets = new MemoryBuffer(offset_data);

        int match_count = scan_key_candidates(physicsBuffer.candidate_counts.memory(), offset_data, n);
        physicsBuffer.set_candidate_match_count(match_count);

    }

    public static void aabb_collide()
    {
        long matches_buf_size = (long)Sizeof.cl_int * physicsBuffer.get_candidate_match_count();
        cl_mem matches_data = cl_new_buffer(FLAGS_WRITE_GPU, matches_buf_size);
        physicsBuffer.matches = new MemoryBuffer(matches_data);

        long used_buf_size = (long)Sizeof.cl_int * physicsBuffer.get_candidate_buffer_count();
        cl_mem used_data = cl_new_buffer(FLAGS_WRITE_GPU, used_buf_size);
        physicsBuffer.matches_used = new MemoryBuffer(used_data);

        // this buffer will contain the total number of candidates that were found
        int[] count = arg_int(0);
        Pointer dst_count = Pointer.to(count);
        cl_mem count_data = cl_new_buffer(FLAGS_WRITE_CPU_COPY, Sizeof.cl_int, dst_count);
        Pointer src_count = Pointer.to(count_data);

        clSetKernelArg(k_aabb_collide, 0, Sizeof.cl_mem, physicsBuffer.bounds.pointer());
        clSetKernelArg(k_aabb_collide, 1, Sizeof.cl_mem, physicsBuffer.bank.pointer());
        clSetKernelArg(k_aabb_collide, 2, Sizeof.cl_mem, physicsBuffer.flags.pointer());
        clSetKernelArg(k_aabb_collide, 3, Sizeof.cl_mem, physicsBuffer.candidate_counts.pointer());
        clSetKernelArg(k_aabb_collide, 4, Sizeof.cl_mem, physicsBuffer.candidate_offsets.pointer());
        clSetKernelArg(k_aabb_collide, 5, Sizeof.cl_mem, physicsBuffer.key_map.pointer());
        clSetKernelArg(k_aabb_collide, 6, Sizeof.cl_mem, physicsBuffer.key_bank.pointer());
        clSetKernelArg(k_aabb_collide, 7, Sizeof.cl_mem, physicsBuffer.key_counts.pointer());
        clSetKernelArg(k_aabb_collide, 8, Sizeof.cl_mem, physicsBuffer.key_offsets.pointer());
        clSetKernelArg(k_aabb_collide, 9, Sizeof.cl_mem, physicsBuffer.matches.pointer());
        clSetKernelArg(k_aabb_collide, 10, Sizeof.cl_mem, physicsBuffer.matches_used.pointer());
        clSetKernelArg(k_aabb_collide, 11, Sizeof.cl_mem, src_count);
        clSetKernelArg(k_aabb_collide, 12, Sizeof.cl_int, physicsBuffer.x_sub_divisions);
        clSetKernelArg(k_aabb_collide, 13, Sizeof.cl_int, physicsBuffer.key_count_length);

        k_call(k_aabb_collide, arg_long(physicsBuffer.get_candidate_buffer_count()));

        cl_read_buffer(count_data, Sizeof.cl_int, dst_count);

        clReleaseMemObject(count_data);

        physicsBuffer.set_candidate_count(count[0]);
    }

    public static void finalize_candidates()
    {
        if (physicsBuffer.get_candidate_count() > 0)
        {
            // create an empty buffer that the kernel will use to store finalized candidates
            long final_buf_size = (long)Sizeof.cl_int2 * physicsBuffer.get_candidate_count();
            cl_mem finals_data = cl_new_buffer(FLAGS_WRITE_GPU, final_buf_size);
            Pointer src_finals = Pointer.to(finals_data);

            // the kernel will use this value as an internal atomic counter, always initialize to zero
            int[] counter = new int[]{ 0 };
            Pointer dst_counter = Pointer.to(counter);
            cl_mem counter_data = cl_new_buffer(FLAGS_WRITE_CPU_COPY, Sizeof.cl_int, dst_counter);
            Pointer src_counter = Pointer.to(counter_data);

            physicsBuffer.set_final_size(final_buf_size);

            physicsBuffer.candidates = new MemoryBuffer(finals_data);

            clSetKernelArg(k_finalize_candidates, 0, Sizeof.cl_mem, physicsBuffer.candidate_counts.pointer());
            clSetKernelArg(k_finalize_candidates, 1, Sizeof.cl_mem, physicsBuffer.candidate_offsets.pointer());
            clSetKernelArg(k_finalize_candidates, 2, Sizeof.cl_mem, physicsBuffer.matches.pointer());
            clSetKernelArg(k_finalize_candidates, 3, Sizeof.cl_mem, physicsBuffer.matches_used.pointer());
            clSetKernelArg(k_finalize_candidates, 4, Sizeof.cl_mem, src_counter);
            clSetKernelArg(k_finalize_candidates, 5, Sizeof.cl_mem, src_finals);

            k_call(k_finalize_candidates, arg_long(physicsBuffer.get_candidate_buffer_count()));

            clReleaseMemObject(counter_data);
        }
    }

    public static void sat_collide()
    {
        if (physicsBuffer.candidates == null) return;

        int candidatesSize = (int) physicsBuffer.get_final_size() / Sizeof.cl_int;

        // Set the work-item dimensions
        long[] global_work_size = new long[]{candidatesSize / COLLISION_SIZE};

        // Set the arguments for the kernel
        clSetKernelArg(k_sat_collide, 0, Sizeof.cl_mem, physicsBuffer.candidates.pointer());
        clSetKernelArg(k_sat_collide, 1, Sizeof.cl_mem, physicsBuffer.hulls.pointer());
        clSetKernelArg(k_sat_collide, 2, Sizeof.cl_mem, physicsBuffer.elements.pointer());
        clSetKernelArg(k_sat_collide, 3, Sizeof.cl_mem, physicsBuffer.flags.pointer());
        clSetKernelArg(k_sat_collide, 4, Sizeof.cl_mem, physicsBuffer.points.pointer());
        clSetKernelArg(k_sat_collide, 5, Sizeof.cl_mem, physicsBuffer.edges.pointer());

        k_call(k_sat_collide, global_work_size);
    }

    public static void resolve_constraints(int edge_steps)
    {
        boolean lastStep;
        long[] global_work_size = new long[]{Main.Memory.hull_count()};
        for (int i = 0; i < edge_steps; i++)
        {
            lastStep = i == edge_steps - 1;
            int n = lastStep ? 1 : 0;
            int a = 0;
            clSetKernelArg(k_resolve_constraints, a++, Sizeof.cl_mem, physicsBuffer.elements.pointer());
            clSetKernelArg(k_resolve_constraints, a++, Sizeof.cl_mem, physicsBuffer.bank.pointer());
            clSetKernelArg(k_resolve_constraints, a++, Sizeof.cl_mem, physicsBuffer.points.pointer());
            clSetKernelArg(k_resolve_constraints, a++, Sizeof.cl_mem, physicsBuffer.edges.pointer());
            clSetKernelArg(k_resolve_constraints, a++, Sizeof.cl_int, Pointer.to(new int[]{n}));

            //gl_acquire(vertex_mem);
            k_call(k_resolve_constraints, global_work_size);
            //gl_release(vertex_mem);
        }
    }

    //#endregion

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

    private static int scan_key_bounds(cl_mem d_data2, int n)
    {
        int k = work_group_count(n);
        if (k == 1)
        {
            return scan_single_block_key(d_data2, n);
        }
        else
        {
            return scan_multi_block_key(d_data2, n, k);
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
        long localBufferSize = Sizeof.cl_int * m;

        clSetKernelArg(k_scan_int_single_block, 0, Sizeof.cl_mem, Pointer.to(d_data));
        clSetKernelArg(k_scan_int_single_block, 1, localBufferSize,null);
        clSetKernelArg(k_scan_int_single_block, 2, Sizeof.cl_int, Pointer.to(new int[]{n}));

        k_call(k_scan_int_single_block, local_work_default, local_work_default);
    }

    private static void scan_multi_block_int(cl_mem d_data, int n, int k)
    {
        long localBufferSize = Sizeof.cl_int * m;
        long gx = k * m;
        long[] global_work_size = arg_long(gx);
        int part_size = k * 2;
        long part_buf_size = ((long)Sizeof.cl_int * ((long)part_size));
        cl_mem part_data = cl_new_buffer(FLAGS_WRITE_GPU, part_buf_size);
        Pointer src_data = Pointer.to(d_data);
        Pointer src_part = Pointer.to(part_data);
        Pointer src_n = Pointer.to(new int[]{n});

        clSetKernelArg(k_scan_int_multi_block, 0, Sizeof.cl_mem, src_data);
        clSetKernelArg(k_scan_int_multi_block, 1, localBufferSize,null);
        clSetKernelArg(k_scan_int_multi_block, 2, Sizeof.cl_mem, src_part);
        clSetKernelArg(k_scan_int_multi_block, 3, Sizeof.cl_int, src_n);

        k_call(k_scan_int_multi_block, global_work_size, local_work_default);

        scan_int(part_data, part_size);

        clSetKernelArg(k_complete_int_multi_block, 0, Sizeof.cl_mem, src_data);
        clSetKernelArg(k_complete_int_multi_block, 1, localBufferSize,null);
        clSetKernelArg(k_complete_int_multi_block, 2, Sizeof.cl_mem, src_part);
        clSetKernelArg(k_complete_int_multi_block, 3, Sizeof.cl_int, src_n);

        k_call(k_complete_int_multi_block, global_work_size, local_work_default);

        clReleaseMemObject(part_data);
    }

    private static void scan_single_block_int_out(cl_mem d_data, cl_mem o_data, int n)
    {
        long localBufferSize = Sizeof.cl_int * m;
        Pointer src_data = Pointer.to(d_data);
        Pointer dst_data = Pointer.to(o_data);

        clSetKernelArg(k_scan_int_single_block_out, 0, Sizeof.cl_mem, src_data);
        clSetKernelArg(k_scan_int_single_block_out, 2, Sizeof.cl_mem, dst_data);
        clSetKernelArg(k_scan_int_single_block_out, 2, localBufferSize,null);
        clSetKernelArg(k_scan_int_single_block_out, 3, Sizeof.cl_int, Pointer.to(new int[]{n}));

        k_call(k_scan_int_single_block_out, local_work_default, local_work_default);
    }

    private static void scan_multi_block_int_out(cl_mem d_data, cl_mem o_data, int n, int k)
    {
        long localBufferSize = Sizeof.cl_int * m;
        long gx = k * m;
        long[] global_work_size = arg_long(gx);
        int part_size = k * 2;
        long part_buf_size = ((long)Sizeof.cl_int * ((long)part_size));
        cl_mem part_data = cl_new_buffer(FLAGS_WRITE_GPU, part_buf_size);
        Pointer src_data = Pointer.to(d_data);
        Pointer src_part = Pointer.to(part_data);
        Pointer dst_data = Pointer.to(o_data);
        Pointer src_n = Pointer.to(new int[]{n});

        clSetKernelArg(k_scan_int_multi_block_out, 0, Sizeof.cl_mem, src_data);
        clSetKernelArg(k_scan_int_multi_block_out, 1, Sizeof.cl_mem, dst_data);
        clSetKernelArg(k_scan_int_multi_block_out, 2, localBufferSize,null);
        clSetKernelArg(k_scan_int_multi_block_out, 3, Sizeof.cl_mem, src_part);
        clSetKernelArg(k_scan_int_multi_block_out, 4, Sizeof.cl_int, src_n);

        k_call(k_scan_int_multi_block_out, global_work_size, local_work_default);

        scan_int(part_data, part_size);

        clSetKernelArg(k_complete_int_multi_block_out, 0, Sizeof.cl_mem, dst_data);
        clSetKernelArg(k_complete_int_multi_block_out, 1, localBufferSize,null);
        clSetKernelArg(k_complete_int_multi_block_out, 2, Sizeof.cl_mem, src_part);
        clSetKernelArg(k_complete_int_multi_block_out, 3, Sizeof.cl_int, src_n);

        k_call(k_complete_int_multi_block_out, global_work_size, local_work_default);

        clReleaseMemObject(part_data);
    }

    private static int scan_single_block_candidates_out(cl_mem d_data, cl_mem o_data, int n)
    {
        long localBufferSize = Sizeof.cl_int * m;
        Pointer src_data = Pointer.to(d_data);
        Pointer dst_data = Pointer.to(o_data);

        int[] sz = new int[]{ 0 };
        Pointer dst_size = Pointer.to(sz);
        cl_mem size_data = cl_new_buffer(FLAGS_WRITE_GPU, Sizeof.cl_int);
        Pointer src_size = Pointer.to(size_data);

        clSetKernelArg(k_scan_candidates_single_block, 0, Sizeof.cl_mem, src_data);
        clSetKernelArg(k_scan_candidates_single_block, 1, Sizeof.cl_mem, dst_data);
        clSetKernelArg(k_scan_candidates_single_block, 2, Sizeof.cl_mem, src_size);
        clSetKernelArg(k_scan_candidates_single_block, 3, localBufferSize,null);
        clSetKernelArg(k_scan_candidates_single_block, 4, Sizeof.cl_int, Pointer.to(new int[]{n}));

        k_call(k_scan_candidates_single_block, local_work_default, local_work_default);

        cl_read_buffer(size_data, Sizeof.cl_int, dst_size);

        clReleaseMemObject(size_data);

        return sz[0];
    }

    private static int scan_multi_block_candidates_out(cl_mem d_data, cl_mem o_data, int n, int k)
    {
        long localBufferSize = Sizeof.cl_int * m;
        long gx = k * m;
        long[] global_work_size = arg_long(gx);
        int part_size = k * 2;
        long part_buf_size = ((long)Sizeof.cl_int * ((long)part_size));
        cl_mem p_data = cl_new_buffer(FLAGS_WRITE_GPU, part_buf_size);
        Pointer src_data = Pointer.to(d_data);
        Pointer src_part = Pointer.to(p_data);
        Pointer dst_data = Pointer.to(o_data);
        Pointer src_n = Pointer.to(new int[]{n});

        clSetKernelArg(k_scan_candidates_multi_block, 0, Sizeof.cl_mem, src_data);
        clSetKernelArg(k_scan_candidates_multi_block, 1, Sizeof.cl_mem, dst_data);
        clSetKernelArg(k_scan_candidates_multi_block, 2, localBufferSize,null);
        clSetKernelArg(k_scan_candidates_multi_block, 3, Sizeof.cl_mem, src_part);
        clSetKernelArg(k_scan_candidates_multi_block, 4, Sizeof.cl_int, src_n);

        k_call(k_scan_candidates_multi_block, global_work_size, local_work_default);

        scan_int(p_data, part_size);

        int[] sz = new int[]{ 0 };
        Pointer dst_size = Pointer.to(sz);
        cl_mem size_data = cl_new_buffer(FLAGS_WRITE_GPU, Sizeof.cl_int);
        Pointer src_size = Pointer.to(size_data);

        clSetKernelArg(k_complete_candidates_multi_block, 0, Sizeof.cl_mem, src_data);
        clSetKernelArg(k_complete_candidates_multi_block, 1, Sizeof.cl_mem, dst_data);
        clSetKernelArg(k_complete_candidates_multi_block, 2, Sizeof.cl_mem, src_size);
        clSetKernelArg(k_complete_candidates_multi_block, 3, localBufferSize,null);
        clSetKernelArg(k_complete_candidates_multi_block, 4, Sizeof.cl_mem, src_part);
        clSetKernelArg(k_complete_candidates_multi_block, 5, Sizeof.cl_int, src_n);

        k_call(k_complete_candidates_multi_block, global_work_size, local_work_default);

        cl_read_buffer(size_data, Sizeof.cl_int, dst_size);

        clReleaseMemObject(p_data);
        clReleaseMemObject(size_data);

        return sz[0];
    }

    private static int scan_single_block_key(cl_mem d_data2, int n)
    {
        long localBufferSize = Sizeof.cl_int * m;
        Pointer src_data2 = Pointer.to(d_data2);

        int[] sz = new int[]{ 0 };
        Pointer dst_size = Pointer.to(sz);
        cl_mem size_data = cl_new_buffer(FLAGS_WRITE_GPU, Sizeof.cl_int);
        Pointer src_size = Pointer.to(size_data);
        Pointer src_n = Pointer.to(new int[]{n});

        clSetKernelArg(k_scan_bounds_single_block, 0, Sizeof.cl_mem, src_data2);
        clSetKernelArg(k_scan_bounds_single_block, 1, Sizeof.cl_mem, src_size);
        clSetKernelArg(k_scan_bounds_single_block, 2, localBufferSize,null);
        clSetKernelArg(k_scan_bounds_single_block, 3, Sizeof.cl_int, src_n);

        k_call(k_scan_bounds_single_block, local_work_default, local_work_default);

        cl_read_buffer(size_data, Sizeof.cl_int, dst_size);

        clReleaseMemObject(size_data);

        return sz[0];
    }

    private static int scan_multi_block_key(cl_mem d_data2, int n, int k)
    {
        long localBufferSize = Sizeof.cl_int * m;
        long gx = k * m;
        long[] global_work_size = arg_long(gx);
        int part_size = k * 2;
        long part_buf_size = ((long)Sizeof.cl_int * ((long)part_size));
        cl_mem p_data = cl_new_buffer(FLAGS_WRITE_GPU, part_buf_size);
        Pointer src_data2 = Pointer.to(d_data2);
        Pointer src_part = Pointer.to(p_data);
        Pointer src_n = Pointer.to(new int[]{n});

        clSetKernelArg(k_scan_bounds_multi_block, 0, Sizeof.cl_mem, src_data2);
        clSetKernelArg(k_scan_bounds_multi_block, 1, localBufferSize,null);
        clSetKernelArg(k_scan_bounds_multi_block, 2, Sizeof.cl_mem, src_part);
        clSetKernelArg(k_scan_bounds_multi_block, 3, Sizeof.cl_int, src_n);

        k_call(k_scan_bounds_multi_block, global_work_size, local_work_default);

        scan_int(p_data, part_size);

        int[] sz = new int[1];
        Pointer dst_size = Pointer.to(sz);
        cl_mem size_data = cl_new_buffer(FLAGS_WRITE_GPU, Sizeof.cl_int);
        Pointer src_size = Pointer.to(size_data);

        clSetKernelArg(k_complete_bounds_multi_block, 0, Sizeof.cl_mem, src_data2);
        clSetKernelArg(k_complete_bounds_multi_block, 1, Sizeof.cl_mem, src_size);
        clSetKernelArg(k_complete_bounds_multi_block, 2, localBufferSize,null);
        clSetKernelArg(k_complete_bounds_multi_block, 3, Sizeof.cl_mem, src_part);
        clSetKernelArg(k_complete_bounds_multi_block, 4, Sizeof.cl_int, src_n);

        k_call(k_complete_bounds_multi_block, global_work_size, local_work_default);

        cl_read_buffer(size_data, Sizeof.cl_int, dst_size);

        clReleaseMemObject(size_data);
        clReleaseMemObject(p_data);

        return sz[0];
    }

    //#endregion
}
