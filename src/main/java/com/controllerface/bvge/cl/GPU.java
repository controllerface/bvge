package com.controllerface.bvge.cl;

import com.controllerface.bvge.Main;
import com.controllerface.bvge.cl.programs.*;
import com.controllerface.bvge.ecs.systems.physics.MemoryBuffer;
import com.controllerface.bvge.ecs.systems.physics.PhysicsBuffer;
import com.controllerface.bvge.ecs.systems.physics.SpatialPartition;
import org.jocl.*;

import java.util.*;

import static com.controllerface.bvge.cl.CLUtils.*;
import static org.jocl.CL.*;
import static org.lwjgl.opengl.WGL.wglGetCurrentContext;
import static org.lwjgl.opengl.WGL.wglGetCurrentDC;

/**
 * Core class used for executing GPU programs.
 * -
 * This class provides core functionality used by the engine to execute parallelized workloads,
 * primarily for physics calculations and pre-processing for rendering operations. This class has
 * two main organizational themes, one internal and one external. Internally, the core components
 * are the GPU programs, GPU kernels, and memory buffers used with them. Externally, this class
 * mainly provides a series of functions that are used to execute predefined GPU programs. This
 * class uses the OpenCL and OpenGL APIs to provide all features.
 */
public class GPU
{
    private static final long FLAGS_WRITE_GPU = CL_MEM_READ_WRITE;
    private static final long FLAGS_WRITE_CPU_COPY = CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR;
    private static final long FLAGS_READ_CPU_COPY = CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR;

    private static final Pointer ZERO_PATTERN = Pointer.to(new int[]{ 0 });

    // these are re-calculated at startup to match the user's hardware
    private static long max_work_group_size = 0;
    private static long max_scan_block_size = 0;

    private static long[] local_work_default = arg_long(0);

    // pre-made size array, used for kernels that have a single work item
    private static final long[] global_single_size = arg_long(1);

    static cl_command_queue command_queue;
    static cl_context context;
    static cl_device_id[] device_ids;

    private static PhysicsBuffer physicsBuffer;

    /**
     * After init, all kernels are loaded into this map, making named access of them simple.
     */
    private static final Map<Kernel, cl_kernel> _k = new HashMap<>();

    /**
     * During shutdown, these are used to release resources. todo: remove these, delegate to program objects
     */
    static List<cl_program> loaded_programs = new ArrayList<>();
    static List<cl_kernel> loaded_kernels = new ArrayList<>();

    /**
     * Enumerates all exiting GPU programs. Programs contain one or more "kernels". A kernel
     * is effectively an entry point into a small program, in reality simply a function.
     * Program implementations are responsible for creating and registering these kernel objects.
     * At runtime, the kernels are called using the Open CL API.
     */
    private enum Program
    {
        aabb_collide(new AabbCollide()),
        animate_hulls(new AnimateHulls()),
        build_key_map(new BuildKeyMap()),
        generate_keys(new GenerateKeys()),
        gpu_crud(new GpuCrud()),
        integrate(new Integrate()),
        locate_in_bounds(new LocateInBounds()),
        prepare_bones(new PrepareBones()),
        prepare_bounds(new PrepareBounds()),
        prepare_edges(new PrepareEdges()),
        prepare_transforms(new PrepareTransforms()),
        resolve_constraints(new ResolveConstraints()),
        sat_collide(new SatCollide()),
        scan_candidates(new ScanCandidates()),
        scan_int_array(new ScanIntArray()),
        scan_int_array_out(new ScanIntArrayOut()),
        scan_key_bank(new ScanKeyBank());

        private final GPUProgram program;

        Program(GPUProgram program)
        {
            this.program = program;
        }
    }

    /**
     * Kernel function names. Program implementations use this enum to instantiate kernel objects
     * with a specific name, which are then called using the various methods of the GPU class.
     */
    public enum Kernel
    {
        aabb_collide,
        animate_hulls,
        build_key_map,
        complete_bounds_multi_block,
        complete_candidates_multi_block_out,
        complete_int_multi_block,
        complete_int_multi_block_out,
        count_candidates,
        create_armature,
        create_bone,
        create_bone_reference,
        create_edge,
        create_hull,
        create_point,
        create_vertex_reference,
        finalize_candidates,
        generate_keys,
        integrate,
        locate_in_bounds,
        prepare_bones,
        prepare_bounds,
        prepare_edges,
        prepare_transforms,
        read_position,
        resolve_constraints,
        rotate_hull,
        sat_collide,
        scan_bounds_multi_block,
        scan_bounds_single_block,
        scan_candidates_multi_block_out,
        scan_candidates_single_block_out,
        scan_int_multi_block,
        scan_int_multi_block_out,
        scan_int_single_block,
        scan_int_single_block_out,
        update_accel,

        ;

        GPUKernel gpu;

        public void set_kernel(GPUKernel gpuKernel)
        {
            this.gpu = gpuKernel;
        }
    }


    //#region Memory Objects

    /**
     * Memory that is shared between Open CL and Open GL contexts
     */
    private static final HashMap<Integer, cl_mem> shared_mem = new LinkedHashMap<>();

    /**
     * Individual points (vertices) of tracked physics hulls. Values are float4 with the following mappings:
     * -
     * x: current x position
     * y: current y position
     * z: previous x position
     * w: previous y position
     * -
     */
    private static cl_mem mem_points;

    /**
     * Edges of tracked physics hulls. Values are float4 with the following mappings:
     * -
     * x: point 1 index
     * y: point 2 index
     * z: distance constraint
     * w: edge flags
     * -
     * note: x, y, and w values are cast to int during use
     */
    private static cl_mem mem_edges;

    /**
     * Positions of tracked hulls. Values are float4 with the following mappings:
     * -
     * x: current x position
     * y: current y position
     * z: scale x
     * w: scale y
     * -
     */
    private static cl_mem mem_hulls;

    /**
     * Axis-aligned bounding boxes of tracked physics hulls. Values are float4 with the following mappings:
     * -
     * x: corner x position
     * y: corner y position
     * z: width
     * w: height
     * -
     */
    private static cl_mem mem_aabb;

    /**
     * Rotation information about tracked physics hulls. Values are float2 with the following mappings:
     * -
     * x: initial reference angle
     * y: current rotation
     * -
     */
    private static cl_mem mem_hull_rotation;

    /**
     * Indexing table for tracked physics hulls. Values are int4 with the following mappings:
     * -
     * x: start point index
     * y: end point index
     * z: start edge index
     * w: end edge index
     * -
     */
    private static cl_mem mem_hull_element_tables;

    /**
     * Flags that related to tracked physics hulls. Values are int2 with the following mappings:
     * -
     * x: hull flags
     * y: armature id
     * -
     */
    private static cl_mem mem_hull_flags;

    /**
     * Spatial partition index information for tracked physics hulls. Values are int4 with the following mappings:
     * -
     * x: minimum x key index
     * y: maximum x key index
     * z: minimum y key index
     * w: maximum y key index
     * -
     */
    private static cl_mem mem_aabb_index;

    /**
     * Spatial partition key bank information for tracked physics hulls. Values are int2 with the following mappings:
     * -
     * x: key bank offset
     * y: key bank size
     * -
     */
    private static cl_mem mem_aabb_key_bank;

    /**
     * Vertex information for loaded models. Values are float2 with the following mappings:
     * -
     * x: x position
     * y: y position
     * -
     */
    private static cl_mem mem_vertex_references;

    /**
     * Indexing table for points of tracked physics hulls. Values are int2 with the following mappings:
     * -
     * x: reference vertex index
     * y: bone index (todo: also used as a proxy for hull ID, based on alignment, but they should be separate)
     * -
     */
    private static cl_mem mem_vertex_table;

    /**
     * Bone offset reference matrices of loaded models. Values are float16 with the following mappings:
     * s0: (m00) transformation matrix column 1 row 1
     * s1: (m01) transformation matrix column 1 row 2
     * s2: (m02) transformation matrix column 1 row 3
     * s3: (m03) transformation matrix column 1 row 4
     * s4: (m10) transformation matrix column 2 row 1
     * s5: (m11) transformation matrix column 2 row 2
     * s6: (m12) transformation matrix column 2 row 3
     * s7: (m13) transformation matrix column 2 row 4
     * s8: (m20) transformation matrix column 3 row 1
     * s9: (m21) transformation matrix column 3 row 2
     * sA: (m22) transformation matrix column 3 row 3
     * sB: (m23) transformation matrix column 3 row 4
     * sC: (m30) transformation matrix column 4 row 1
     * sD: (m31) transformation matrix column 4 row 2
     * sE: (m32) transformation matrix column 4 row 3
     * sF: (m33) transformation matrix column 4 row 4
     */
    private static cl_mem mem_bone_references;

    /**
     * Bone offset animation matrices of tracked physics hulls. Values are float16 with the following mappings:
     * s0: (m00) transformation matrix column 1 row 1
     * s1: (m01) transformation matrix column 1 row 2
     * s2: (m02) transformation matrix column 1 row 3
     * s3: (m03) transformation matrix column 1 row 4
     * s4: (m10) transformation matrix column 2 row 1
     * s5: (m11) transformation matrix column 2 row 2
     * s6: (m12) transformation matrix column 2 row 3
     * s7: (m13) transformation matrix column 2 row 4
     * s8: (m20) transformation matrix column 3 row 1
     * s9: (m21) transformation matrix column 3 row 2
     * sA: (m22) transformation matrix column 3 row 3
     * sB: (m23) transformation matrix column 3 row 4
     * sC: (m30) transformation matrix column 4 row 1
     * sD: (m31) transformation matrix column 4 row 2
     * sE: (m32) transformation matrix column 4 row 3
     * sF: (m33) transformation matrix column 4 row 4
     */
    private static cl_mem mem_bone_instances;

    /**
     * Reference bone index of bones used for tracked physics hulls. Values are int with the following mapping:
     * value: bone reference index
     */
    private static cl_mem mem_bone_index;

    /**
     * Armature information for tracked physics hulls. Values are float4 with the following mappings:
     * x: current x position
     * y: current y position
     * z: previous x position
     * w: previous y position
     */
    private static cl_mem mem_armatures;

    /**
     * Hull index of root hull for an armature. Values are int with the following mapping:
     * value: hull index
     */
    private static cl_mem mem_armature_flags;

    /**
     * Acceleration value of an armature. Values are float2 with the following mappings:
     * x: current x acceleration
     * y: current y acceleration
     */
    private static cl_mem mem_armature_acceleration;

    //#endregion


    private static void cl_read_buffer(cl_mem src, long size, Pointer dst)
    {
        clEnqueueReadBuffer(command_queue, src, CL_TRUE, 0, size, dst,
            0, null, null);
    }

    private static cl_mem cl_new_buffer(long size)
    {
        return clCreateBuffer(context, FLAGS_WRITE_GPU, size, null, null);
    }

    private static cl_mem cl_new_int_arg_buffer(Pointer src)
    {
        return clCreateBuffer(context, FLAGS_WRITE_CPU_COPY, Sizeof.cl_int, src, null);
    }

    private static cl_mem cl_new_read_only_buffer(long size, Pointer src)
    {
        return clCreateBuffer(context, FLAGS_READ_CPU_COPY, size, src, null);
    }

    private static void cl_zero_buffer(cl_mem buffer, long buffer_size)
    {
        clEnqueueFillBuffer(command_queue, buffer, ZERO_PATTERN, 1, 0, buffer_size,
            0, null, null);
    }

    public static cl_program cl_p(List<String> src_strings)
    {
        return cl_p(src_strings.toArray(new String[]{}));
    }

    public static cl_program cl_p(String... src)
    {
        var program = CLUtils.cl_p(context, device_ids, src);
        loaded_programs.add(program);
        return program;
    }

    public static cl_kernel cl_k(cl_program program, String kernel_name)
    {
        var kernel = CLUtils.cl_k(program, kernel_name);
        loaded_kernels.add(kernel);
        return kernel;
    }

    public static int work_group_count(int n)
    {
        return (int) Math.ceil((float) n / (float) max_scan_block_size);
    }

    public static void setPhysicsBuffer(PhysicsBuffer physicsBuffer)
    {
        GPU.physicsBuffer = physicsBuffer;
    }

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

        // Obtain the number of devices for the platform
        int[] numDevicesArray = new int[1];
        clGetDeviceIDs(platform, deviceType, 0, null, numDevicesArray);
        int numDevices = numDevicesArray[0];

        // Obtain a device ID
        cl_device_id[] devices = new cl_device_id[numDevices];
        clGetDeviceIDs(platform, deviceType, numDevices, devices, null);
        cl_device_id device = devices[deviceIndex];

        var device_ids = new cl_device_id[]{device};

        var dc = wglGetCurrentDC();
        var ctx = wglGetCurrentContext();

        // todo: the above code is windows specific add linux code path,
        //  should look something like this:
        // var ctx = glXGetCurrentContext();
        // var dc = glXGetCurrentDrawable();
        // contextProperties.addProperty(CL_GLX_DISPLAY_KHR, dc);

        // Initialize the context properties
        cl_context_properties contextProperties = new cl_context_properties();
        contextProperties.addProperty(CL_CONTEXT_PLATFORM, platform);
        contextProperties.addProperty(CL_GL_CONTEXT_KHR, ctx);
        contextProperties.addProperty(CL_WGL_HDC_KHR, dc);

//        OpenCL.printDeviceDetails(device_ids);
        // Create a context for the selected device
        context = clCreateContext(
            contextProperties, 1, device_ids,
            null, null, null);

        // Create a command-queue for the selected device
        cl_queue_properties properties = new cl_queue_properties();
        command_queue = clCreateCommandQueueWithProperties(
            context, device, properties, null);

        return device_ids;

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

        max_work_group_size = getSize(device, CL_DEVICE_MAX_WORK_GROUP_SIZE);
        max_scan_block_size = max_work_group_size * 2;
        local_work_default = new long[]{max_work_group_size};

        // initialize kernel programs
        for (Program program : Program.values())
        {
            program.program.init();
            _k.putAll(program.program.kernels);
        }

        //OpenCLUtils.debugDeviceDetails(device_ids);

        // create memory buffers
        prepare_memory(max_hulls, max_points);

        // Create re-usable kernel objects
        prepare_kernels();
    }

    private static void prepare_memory(int max_hulls, int max_points)
    {
        int transform_mem_size        = max_hulls * Sizeof.cl_float4;
        int acceleration_mem_size     = max_hulls * Sizeof.cl_float2;
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
        int bone_reference_mem_size   = max_points * Sizeof.cl_float16;
        int bone_instance_mem_size    = max_points * Sizeof.cl_float16;
        int bone_index_mem_size       = max_points * Sizeof.cl_int;
        int armature_mem_size         = max_points * Sizeof.cl_float4;
        int armature_flags_mem_size   = max_points * Sizeof.cl_int;

        mem_armature_acceleration = cl_new_buffer(acceleration_mem_size);
        mem_hull_rotation         = cl_new_buffer(rotation_mem_size);
        mem_hull_element_tables   = cl_new_buffer(element_table_mem_size);
        mem_hull_flags            = cl_new_buffer(flags_mem_size);
        mem_aabb_index            = cl_new_buffer(spatial_index_mem_size);
        mem_aabb_key_bank         = cl_new_buffer(spatial_key_bank_mem_size);
        mem_hulls                 = cl_new_buffer(transform_mem_size);
        mem_aabb                  = cl_new_buffer(bounding_box_mem_size);
        mem_points                = cl_new_buffer(points_mem_size);
        mem_edges                 = cl_new_buffer(edges_mem_size);
        mem_vertex_table          = cl_new_buffer(vertex_table_mem_size);
        mem_vertex_references     = cl_new_buffer(vertex_reference_mem_size);
        mem_bone_references       = cl_new_buffer(bone_reference_mem_size);
        mem_bone_instances        = cl_new_buffer(bone_instance_mem_size);
        mem_bone_index            = cl_new_buffer(bone_index_mem_size);
        mem_armatures             = cl_new_buffer(armature_mem_size);
        mem_armature_flags        = cl_new_buffer(armature_flags_mem_size);

        cl_zero_buffer(mem_armature_acceleration, acceleration_mem_size);
        cl_zero_buffer(mem_hull_rotation, rotation_mem_size);
        cl_zero_buffer(mem_hull_element_tables, element_table_mem_size);
        cl_zero_buffer(mem_hull_flags, flags_mem_size);
        cl_zero_buffer(mem_aabb_index, spatial_index_mem_size);
        cl_zero_buffer(mem_aabb_key_bank, spatial_key_bank_mem_size);
        cl_zero_buffer(mem_hulls, transform_mem_size);
        cl_zero_buffer(mem_aabb, bounding_box_mem_size);
        cl_zero_buffer(mem_points, points_mem_size);
        cl_zero_buffer(mem_edges, edges_mem_size);
        cl_zero_buffer(mem_vertex_table, vertex_table_mem_size);
        cl_zero_buffer(mem_vertex_references, vertex_reference_mem_size);
        cl_zero_buffer(mem_bone_references, bone_reference_mem_size);
        cl_zero_buffer(mem_bone_instances, bone_instance_mem_size);
        cl_zero_buffer(mem_bone_index, bone_index_mem_size);
        cl_zero_buffer(mem_armatures, armature_mem_size);
        cl_zero_buffer(mem_armature_flags, armature_flags_mem_size);

        // Debugging info
        int total = transform_mem_size
            + acceleration_mem_size
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
            + bone_reference_mem_size
            + bone_instance_mem_size
            + bone_index_mem_size
            + armature_mem_size
            + armature_flags_mem_size;

        System.out.println("------------- BUFFERS -------------");
        System.out.println("points            : " + points_mem_size);
        System.out.println("edges             : " + edges_mem_size);
        System.out.println("transforms        : " + transform_mem_size);
        System.out.println("acceleration      : " + acceleration_mem_size);
        System.out.println("rotation          : " + rotation_mem_size);
        System.out.println("element table     : " + element_table_mem_size);
        System.out.println("flags             : " + flags_mem_size);
        System.out.println("bounding box      : " + bounding_box_mem_size);
        System.out.println("spatial index     : " + spatial_index_mem_size);
        System.out.println("spatial key bank  : " + spatial_key_bank_mem_size);
        System.out.println("vertex table      : " + vertex_table_mem_size);
        System.out.println("vertex references : " + vertex_reference_mem_size);
        System.out.println("bone references   : " + bone_reference_mem_size);
        System.out.println("bone instances    : " + bone_instance_mem_size);
        System.out.println("bone index        : " + bone_index_mem_size);
        System.out.println("armatures         : " + armature_mem_size);
        System.out.println("armature flags    : " + armature_flags_mem_size);
        System.out.println("=====================================");
        System.out.println(" Total (Bytes)    : " + total);
        System.out.println("               KB : " + ((float) total / 1024f));
        System.out.println("               MB : " + ((float) total / 1024f / 1024f));
        System.out.println("               GB : " + ((float) total / 1024f / 1024f / 1024f));
        System.out.println("-----------------------------------\n");
    }

    /**
     * Creates reusable GPUKernel objects that the individual API methods use to implement
     * the CPU-GPU transition layer. Pre-generating kernels this way helps to reduce calls
     * to set kernel arguments, which can be expensive. Where possible, kernel arguments
     * can be set once, and then subsequent calls to that kernel do not require setting
     * the argument again. Only arguments with data that changes need to be updated.
     * Generally, kernel functions operate on large arrays of data, which can be defined
     * as arguments only once, even if the contents of these arrays changes often.
     */
    private static void prepare_kernels()
    {
        var gpu_prepare_bounds = new GPUKernel(command_queue, _k.get(Kernel.prepare_bounds), 3);
        gpu_prepare_bounds.set_arg(0, Sizeof.cl_mem, Pointer.to(mem_aabb));
        gpu_prepare_bounds.def_arg(1, Sizeof.cl_mem);
        gpu_prepare_bounds.def_arg(2, Sizeof.cl_int);
        Kernel.prepare_bounds.set_kernel(gpu_prepare_bounds);

        var gpu_prepare_transforms = new GPUKernel(command_queue, _k.get(Kernel.prepare_transforms), 4);
        gpu_prepare_transforms.set_arg(0, Sizeof.cl_mem, Pointer.to(mem_hulls));
        gpu_prepare_transforms.set_arg(1, Sizeof.cl_mem, Pointer.to(mem_hull_rotation));
        gpu_prepare_transforms.def_arg(2, Sizeof.cl_mem);
        gpu_prepare_transforms.def_arg(3, Sizeof.cl_mem);
        Kernel.prepare_transforms.set_kernel(gpu_prepare_transforms);

        var gpu_prepare_edges = new GPUKernel(command_queue, _k.get(Kernel.prepare_edges), 4);
        gpu_prepare_edges.set_arg(0, Sizeof.cl_mem, Pointer.to(mem_points));
        gpu_prepare_edges.set_arg(1, Sizeof.cl_mem, Pointer.to(mem_edges));
        gpu_prepare_edges.def_arg(2, Sizeof.cl_mem);
        gpu_prepare_edges.def_arg(3, Sizeof.cl_int);
        Kernel.prepare_edges.set_kernel(gpu_prepare_edges);

        var gpu_prepare_bones = new GPUKernel(command_queue, _k.get(Kernel.prepare_bones), 8);
        gpu_prepare_bones.set_arg(0, Sizeof.cl_mem, Pointer.to(mem_bone_instances));
        gpu_prepare_bones.set_arg(1, Sizeof.cl_mem, Pointer.to(mem_bone_references));
        gpu_prepare_bones.set_arg(2, Sizeof.cl_mem, Pointer.to(mem_bone_index));
        gpu_prepare_bones.set_arg(3, Sizeof.cl_mem, Pointer.to(mem_hulls));
        gpu_prepare_bones.set_arg(4, Sizeof.cl_mem, Pointer.to(mem_armatures));
        gpu_prepare_bones.set_arg(5, Sizeof.cl_mem, Pointer.to(mem_hull_flags));
        gpu_prepare_bones.def_arg(6, Sizeof.cl_mem);
        gpu_prepare_bones.def_arg(7, Sizeof.cl_int);
        Kernel.prepare_bones.set_kernel(gpu_prepare_bones);

        var gpu_create_point = new GPUKernel(command_queue, _k.get(Kernel.create_point), 5);
        gpu_create_point.set_arg(0, Sizeof.cl_mem, Pointer.to(mem_points));
        gpu_create_point.set_arg(1, Sizeof.cl_mem, Pointer.to(mem_vertex_table));
        gpu_create_point.def_arg(2, Sizeof.cl_int);
        gpu_create_point.def_arg(3, Sizeof.cl_float4);
        gpu_create_point.def_arg(4, Sizeof.cl_int2);
        Kernel.create_point.set_kernel(gpu_create_point);

        var gpu_create_edge = new GPUKernel(command_queue, _k.get(Kernel.create_edge), 3);
        gpu_create_edge.set_arg(0, Sizeof.cl_mem, Pointer.to(mem_edges));
        gpu_create_edge.def_arg(1, Sizeof.cl_int);
        gpu_create_edge.def_arg(2, Sizeof.cl_float4);
        Kernel.create_edge.set_kernel(gpu_create_edge);

        var create_armature = new GPUKernel(command_queue, _k.get(Kernel.create_armature), 5);
        create_armature.set_arg(0, Sizeof.cl_mem, Pointer.to(mem_armatures));
        create_armature.set_arg(1, Sizeof.cl_mem, Pointer.to(mem_armature_flags));
        create_armature.def_arg(2, Sizeof.cl_int);
        create_armature.def_arg(3, Sizeof.cl_float4);
        create_armature.def_arg(4, Sizeof.cl_int);
        Kernel.create_armature.set_kernel(create_armature);

        var create_vertex_ref = new GPUKernel(command_queue, _k.get(Kernel.create_vertex_reference), 3);
        create_vertex_ref.set_arg(0, Sizeof.cl_mem, Pointer.to(mem_vertex_references));
        create_vertex_ref.def_arg(1, Sizeof.cl_int);
        create_vertex_ref.def_arg(2, Sizeof.cl_float2);
        Kernel.create_vertex_reference.set_kernel(create_vertex_ref);

        var create_bone_ref = new GPUKernel(command_queue, _k.get(Kernel.create_bone_reference), 3);
        create_bone_ref.set_arg(0, Sizeof.cl_mem, Pointer.to(mem_bone_references));
        create_bone_ref.def_arg(1, Sizeof.cl_int);
        create_bone_ref.def_arg(2, Sizeof.cl_float16);
        Kernel.create_bone_reference.set_kernel(create_bone_ref);

        var create_bone = new GPUKernel(command_queue, _k.get(Kernel.create_bone), 5);
        create_bone.set_arg(0, Sizeof.cl_mem, Pointer.to(mem_bone_instances));
        create_bone.set_arg(1, Sizeof.cl_mem, Pointer.to(mem_bone_index));
        create_bone.def_arg(2, Sizeof.cl_int);
        create_bone.def_arg(3, Sizeof.cl_float16);
        create_bone.def_arg(4, Sizeof.cl_int);
        Kernel.create_bone.set_kernel(create_bone);

        var create_hull = new GPUKernel(command_queue, _k.get(Kernel.create_hull), 9);
        create_hull.set_arg(0, Sizeof.cl_mem, Pointer.to(mem_hulls));
        create_hull.set_arg(1, Sizeof.cl_mem, Pointer.to(mem_hull_rotation));
        create_hull.set_arg(2, Sizeof.cl_mem, Pointer.to(mem_hull_element_tables));
        create_hull.set_arg(3, Sizeof.cl_mem, Pointer.to(mem_hull_flags));
        create_hull.def_arg(4, Sizeof.cl_int);
        create_hull.def_arg(5, Sizeof.cl_float4);
        create_hull.def_arg(6, Sizeof.cl_float2);
        create_hull.def_arg(7, Sizeof.cl_int4);
        create_hull.def_arg(8, Sizeof.cl_int2);
        Kernel.create_hull.set_kernel(create_hull);

        var update_accel = new GPUKernel(command_queue, _k.get(Kernel.update_accel), 3);
        update_accel.set_arg(0, Sizeof.cl_mem, Pointer.to(mem_armature_acceleration));
        update_accel.def_arg(1, Sizeof.cl_int);
        update_accel.def_arg(2, Sizeof.cl_float2);
        Kernel.update_accel.set_kernel(update_accel);

        var read_position = new GPUKernel(command_queue, _k.get(Kernel.read_position), 3);
        read_position.set_arg(0, Sizeof.cl_mem, Pointer.to(mem_armatures));
        read_position.def_arg(1, Sizeof.cl_float2);
        read_position.def_arg(2, Sizeof.cl_int);
        Kernel.read_position.set_kernel(read_position);

        var animate_hulls = new GPUKernel(command_queue, _k.get(Kernel.animate_hulls), 8);
        animate_hulls.set_arg(0, Sizeof.cl_mem, Pointer.to(mem_points));
        animate_hulls.set_arg(1, Sizeof.cl_mem, Pointer.to(mem_hulls));
        animate_hulls.set_arg(2, Sizeof.cl_mem, Pointer.to(mem_hull_flags));
        animate_hulls.set_arg(3, Sizeof.cl_mem, Pointer.to(mem_vertex_table));
        animate_hulls.set_arg(4, Sizeof.cl_mem, Pointer.to(mem_armatures));
        animate_hulls.set_arg(5, Sizeof.cl_mem, Pointer.to(mem_armature_flags));
        animate_hulls.set_arg(6, Sizeof.cl_mem, Pointer.to(mem_vertex_references));
        animate_hulls.set_arg(7, Sizeof.cl_mem, Pointer.to(mem_bone_instances));
        Kernel.animate_hulls.set_kernel(animate_hulls);

        var integrate = new GPUKernel(command_queue, _k.get(Kernel.integrate), 12);
        integrate.set_arg(0, Sizeof.cl_mem, Pointer.to(mem_hulls));
        integrate.set_arg(1, Sizeof.cl_mem, Pointer.to(mem_armatures));
        integrate.set_arg(2, Sizeof.cl_mem, Pointer.to(mem_armature_flags));
        integrate.set_arg(3, Sizeof.cl_mem, Pointer.to(mem_hull_element_tables));
        integrate.set_arg(4, Sizeof.cl_mem, Pointer.to(mem_armature_acceleration));
        integrate.set_arg(5, Sizeof.cl_mem, Pointer.to(mem_hull_rotation));
        integrate.set_arg(6, Sizeof.cl_mem, Pointer.to(mem_points));
        integrate.set_arg(7, Sizeof.cl_mem, Pointer.to(mem_aabb));
        integrate.set_arg(8, Sizeof.cl_mem, Pointer.to(mem_aabb_index));
        integrate.set_arg(9, Sizeof.cl_mem, Pointer.to(mem_aabb_key_bank));
        integrate.set_arg(10, Sizeof.cl_mem, Pointer.to(mem_hull_flags));
        integrate.def_arg(11, Sizeof.cl_mem);
        Kernel.integrate.set_kernel(integrate);
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
        clReleaseCommandQueue(command_queue);
        clReleaseContext(context);
    }

    //#region GL Interop

    /**
     * Called from Gl code to share a buffer object with CL. This allows kernels to process
     * buffer data for shaders. The size of the shared data is determined automatically by the
     * vboID and is set in the GL context.
     *
     * @param vboID GL buffer ID of the data to share.
     */
    public static void share_memory(int vboID)
    {
        var vbo_mem = clCreateFromGLBuffer(context, FLAGS_WRITE_GPU, vboID, null);
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
        var gpu_kernel = Kernel.prepare_bounds.gpu;

        var vbo_mem = shared_mem.get(vboID);
        long[] edge_offset = arg_long(vboOffset);

        gpu_kernel.share_mem(vbo_mem);
        gpu_kernel.update_arg(1, Pointer.to(vbo_mem));
        gpu_kernel.update_arg(2, Pointer.to(edge_offset));
        gpu_kernel.call(arg_long(batchSize));
    }

    public static void GL_bones(int vboID, int vboOffset, int batchSize)
    {
        var gpu_kernel = Kernel.prepare_bones.gpu;

        var vbo_mem = shared_mem.get(vboID);
        long[] bone_offset = arg_long(vboOffset);

        gpu_kernel.share_mem(vbo_mem);
        gpu_kernel.update_arg(6, Pointer.to(vbo_mem));
        gpu_kernel.update_arg(7, Pointer.to(bone_offset));
        gpu_kernel.call(arg_long(batchSize));
    }

    public static void GL_edges(int vboID, int vboOffset, int batchSize)
    {
        var gpu_kernel = Kernel.prepare_edges.gpu;

        var vbo_mem = shared_mem.get(vboID);
        long[] edge_offset = arg_long(vboOffset);

        gpu_kernel.share_mem(vbo_mem);
        gpu_kernel.update_arg(2, Pointer.to(vbo_mem));
        gpu_kernel.update_arg(3, Pointer.to(edge_offset));
        gpu_kernel.call(arg_long(batchSize));
    }

    public static void GL_transforms(int index_buffer_id, int transforms_id, int size)
    {
        var gpu_kernel = Kernel.prepare_transforms.gpu;

        var vbo_index_buffer = shared_mem.get(index_buffer_id);
        var vbo_transforms = shared_mem.get(transforms_id);

        gpu_kernel.share_mem(vbo_index_buffer);
        gpu_kernel.share_mem(vbo_transforms);
        gpu_kernel.update_arg(2, Pointer.to(vbo_index_buffer));
        gpu_kernel.update_arg(3, Pointer.to(vbo_transforms));
        gpu_kernel.call(arg_long(size));
    }

    //#endregion

    //#region CPU Create/Read/Update/Delete Functions

    public static void create_point(int point_index, float pos_x, float pos_y, float prv_x, float prv_y, int v, int b)
    {
        var gpu_kernel = Kernel.create_point.gpu;

        var pnt_index = Pointer.to(arg_int(point_index));
        var pnt_point = Pointer.to(arg_float4(pos_x, pos_y, prv_x, prv_y));
        var pnt_table = Pointer.to(arg_int2(v, b));

        gpu_kernel.update_arg(2, pnt_index);
        gpu_kernel.update_arg(3, pnt_point);
        gpu_kernel.update_arg(4, pnt_table);
        gpu_kernel.call(global_single_size);
    }

    public static void create_edge(int edge_index, float p1, float p2, float l, int flags)
    {
        var gpu_kernel = Kernel.create_edge.gpu;

        var pnt_index = Pointer.to(arg_int(edge_index));
        var pnt_edge = Pointer.to(arg_float4(p1, p2, l, flags));

        gpu_kernel.update_arg(1, pnt_index);
        gpu_kernel.update_arg(2, pnt_edge);
        gpu_kernel.call(global_single_size);
    }

    public static void create_armature(int armature_index, float x, float y, int flags)
    {
        var gpu_kernel = Kernel.create_armature.gpu;

        var pnt_index = Pointer.to(arg_int(armature_index));
        var pnt_armature = Pointer.to(arg_float4(x, y, x, y));
        var pnt_flags = Pointer.to(arg_int(flags));

        gpu_kernel.update_arg(2, pnt_index);
        gpu_kernel.update_arg(3, pnt_armature);
        gpu_kernel.update_arg(4, pnt_flags);
        gpu_kernel.call(global_single_size);
    }

    public static void create_vertex_reference(int vert_ref_index, float x, float y)
    {
        var gpu_kernel = Kernel.create_vertex_reference.gpu;

        var pnt_index = Pointer.to(arg_int(vert_ref_index));
        var pnt_vert_ref = Pointer.to(arg_float2(x, y));

        gpu_kernel.update_arg(1, pnt_index);
        gpu_kernel.update_arg(2, pnt_vert_ref);
        gpu_kernel.call(global_single_size);
    }

    public static void create_bone_reference(int bone_ref_index, float[] matrix)
    {
        var gpu_kernel = Kernel.create_bone_reference.gpu;

        var pnt_index = Pointer.to(arg_int(bone_ref_index));
        var pnt_bone_ref = Pointer.to(matrix);

        gpu_kernel.update_arg(1, pnt_index);
        gpu_kernel.update_arg(2, pnt_bone_ref);
        gpu_kernel.call(global_single_size);
    }

    public static void create_bone(int bone_index, int bone_ref_index, float[] matrix)
    {
        var gpu_kernel = Kernel.create_bone.gpu;

        var pnt_index = Pointer.to(arg_int(bone_index));
        var pnt_ref_index = Pointer.to(arg_int(bone_ref_index));
        var pnt_bone_ref = Pointer.to(matrix);

        gpu_kernel.update_arg(2, pnt_index);
        gpu_kernel.update_arg(3, pnt_bone_ref);
        gpu_kernel.update_arg(4, pnt_ref_index);
        gpu_kernel.call(global_single_size);
    }

    public static void create_hull(int hull_index, float[] hull, float[] rotation, int[] table, int[] flags)
    {
        var gpu_kernel = Kernel.create_hull.gpu;

        var pnt_index = Pointer.to(arg_int(hull_index));
        var pnt_flags = Pointer.to(flags);
        var pnt_table = Pointer.to(table);
        var pnt_rotation = Pointer.to(rotation);
        var pnt_hull = Pointer.to(hull);

        gpu_kernel.update_arg(4, pnt_index);
        gpu_kernel.update_arg(5, pnt_hull);
        gpu_kernel.update_arg(6, pnt_rotation);
        gpu_kernel.update_arg(7, pnt_table);
        gpu_kernel.update_arg(8, pnt_flags);
        gpu_kernel.call(global_single_size);
    }

    public static void update_accel(int armature_index, float acc_x, float acc_y)
    {
        var gpu_kernel = Kernel.update_accel.gpu;

        var pnt_index = Pointer.to(arg_int(armature_index));
        var pnt_acc = Pointer.to(arg_float2(acc_x, acc_y));

        gpu_kernel.update_arg(1, pnt_index);
        gpu_kernel.update_arg(2, pnt_acc);
        gpu_kernel.call(global_single_size);
    }

    // todo: implement armature rotations and update this
    public static void rotate_hull(int hull_index, float angle)
    {
        var pnt_index = Pointer.to(arg_int(hull_index));
        var pnt_angle = Pointer.to(arg_float(angle));

        clSetKernelArg(_k.get(Kernel.rotate_hull), 0, Sizeof.cl_mem, Pointer.to(mem_hulls));
        clSetKernelArg(_k.get(Kernel.rotate_hull), 1, Sizeof.cl_mem, Pointer.to(mem_hull_element_tables));
        clSetKernelArg(_k.get(Kernel.rotate_hull), 2, Sizeof.cl_mem, Pointer.to(mem_points));
        clSetKernelArg(_k.get(Kernel.rotate_hull), 3, Sizeof.cl_int, pnt_index);
        clSetKernelArg(_k.get(Kernel.rotate_hull), 4, Sizeof.cl_float, pnt_angle);

        k_call(command_queue, _k.get(Kernel.rotate_hull), global_single_size);
    }

    public static float[] read_position(int armature_index)
    {
        if (physicsBuffer == null)
        {
            return null;
        }

        var gpu_kernel = Kernel.read_position.gpu;

        int[] index = arg_int(armature_index);

        cl_mem result_data = cl_new_buffer(Sizeof.cl_float2);
        cl_zero_buffer(result_data, Sizeof.cl_float2);
        Pointer src_result = Pointer.to(result_data);

        gpu_kernel.update_arg(1, src_result);
        gpu_kernel.update_arg(2, Pointer.to(index));
        gpu_kernel.call(global_single_size);

        float[] result = arg_float2(0, 0);
        Pointer dst_result = Pointer.to(result);
        cl_read_buffer(result_data, Sizeof.cl_float2, dst_result);
        clReleaseMemObject(result_data);

        return result;
    }

    //#endregion

    //#region Physics Simulation

    public static void animate_hulls()
    {
        Kernel.animate_hulls.gpu.call(arg_long(Main.Memory.point_count()));
    }

    public static void integrate(float delta_time, SpatialPartition spatialPartition)
    {
        var gpu_kernel = Kernel.integrate.gpu;

        float[] args =
            {
                delta_time,
                spatialPartition.getX_spacing(),
                spatialPartition.getY_spacing(),
                spatialPartition.getX_origin(),
                spatialPartition.getY_origin(),
                spatialPartition.getWidth(),
                spatialPartition.getHeight(),
                (float) spatialPartition.getX_subdivisions(),
                (float) spatialPartition.getY_subdivisions(),
                physicsBuffer.get_gravity_x(),
                physicsBuffer.get_gravity_y(),
                physicsBuffer.get_friction()
            };

        Pointer srcArgs = Pointer.to(args);

        long size = Sizeof.cl_float * args.length;
        cl_mem argMem = cl_new_read_only_buffer(size, srcArgs);

        gpu_kernel.update_arg(11, Pointer.to(argMem));
        gpu_kernel.call(arg_long(Main.Memory.hull_count()));

        clReleaseMemObject(argMem);
    }

    public static void calculate_bank_offsets(SpatialPartition spatialPartition)
    {
        int n = Main.Memory.hull_count();
        int bank_size = scan_key_bounds(mem_aabb_key_bank, n);
        spatialPartition.resizeBank(bank_size);
    }

    public static void generate_keys(SpatialPartition spatialPartition)
    {
        if (spatialPartition.getKey_bank_size() < 1)
        {
            return;
        }
        int n = Main.Memory.hull_count();
        long bank_buf_size = (long) Sizeof.cl_int * spatialPartition.getKey_bank_size();
        long counts_buf_size = (long) Sizeof.cl_int * spatialPartition.getDirectoryLength();

        cl_mem bank_data = cl_new_buffer(bank_buf_size);
        cl_mem counts_data = cl_new_buffer(counts_buf_size);
        cl_zero_buffer(counts_data, counts_buf_size);

        Pointer src_bank = Pointer.to(bank_data);
        Pointer src_counts = Pointer.to(counts_data);
        Pointer src_kb_len = Pointer.to(arg_int(spatialPartition.getKey_bank_size()));
        Pointer src_kc_len = Pointer.to(arg_int(spatialPartition.getDirectoryLength()));
        Pointer src_x_subs = Pointer.to(arg_int(spatialPartition.getX_subdivisions()));

        physicsBuffer.key_counts = new MemoryBuffer(counts_data);
        physicsBuffer.key_bank = new MemoryBuffer(bank_data);

        // pass in arguments
        clSetKernelArg(_k.get(Kernel.generate_keys), 0, Sizeof.cl_mem, Pointer.to(mem_aabb_index));
        clSetKernelArg(_k.get(Kernel.generate_keys), 1, Sizeof.cl_mem, Pointer.to(mem_aabb_key_bank));
        clSetKernelArg(_k.get(Kernel.generate_keys), 2, Sizeof.cl_mem, src_bank);
        clSetKernelArg(_k.get(Kernel.generate_keys), 3, Sizeof.cl_mem, src_counts);
        clSetKernelArg(_k.get(Kernel.generate_keys), 4, Sizeof.cl_int, src_x_subs);
        clSetKernelArg(_k.get(Kernel.generate_keys), 5, Sizeof.cl_int, src_kb_len);
        clSetKernelArg(_k.get(Kernel.generate_keys), 6, Sizeof.cl_int, src_kc_len);

        k_call(command_queue, _k.get(Kernel.generate_keys), arg_long(n));
    }

    public static void calculate_map_offsets(SpatialPartition spatialPartition)
    {
        int n = spatialPartition.getDirectoryLength();
        long data_buf_size = (long) Sizeof.cl_int * n;
        cl_mem o_data = cl_new_buffer(data_buf_size);
        physicsBuffer.key_offsets = new MemoryBuffer(o_data);
        scan_int_out(physicsBuffer.key_counts.memory(), o_data, n);
    }

    public static void build_key_map(SpatialPartition spatialPartition)
    {
        int n = Main.Memory.hull_count();
        long map_buf_size = (long) Sizeof.cl_int * spatialPartition.getKey_map_size();
        long counts_buf_size = (long) Sizeof.cl_int * spatialPartition.getDirectoryLength();

        cl_mem map_data = cl_new_buffer(map_buf_size);
        cl_mem counts_data = cl_new_buffer(counts_buf_size);

        // the counts buffer needs to start off filled with all zeroes
        cl_zero_buffer(counts_data, counts_buf_size);

        Pointer src_map = Pointer.to(map_data);
        Pointer src_counts = Pointer.to(counts_data);
        Pointer src_x_subs = Pointer.to(arg_int(spatialPartition.getX_subdivisions()));
        Pointer src_c_len = Pointer.to(arg_int(spatialPartition.getDirectoryLength()));

        physicsBuffer.key_map = new MemoryBuffer(map_data);

        clSetKernelArg(_k.get(Kernel.build_key_map), 0, Sizeof.cl_mem, Pointer.to(mem_aabb_index));
        clSetKernelArg(_k.get(Kernel.build_key_map), 1, Sizeof.cl_mem, Pointer.to(mem_aabb_key_bank));
        clSetKernelArg(_k.get(Kernel.build_key_map), 2, Sizeof.cl_mem, src_map);
        clSetKernelArg(_k.get(Kernel.build_key_map), 3, Sizeof.cl_mem, physicsBuffer.key_offsets.pointer());
        clSetKernelArg(_k.get(Kernel.build_key_map), 4, Sizeof.cl_mem, src_counts);
        clSetKernelArg(_k.get(Kernel.build_key_map), 5, Sizeof.cl_int, src_x_subs);
        clSetKernelArg(_k.get(Kernel.build_key_map), 6, Sizeof.cl_int, src_c_len);

        k_call(command_queue, _k.get(Kernel.build_key_map), arg_long(n));

        clReleaseMemObject(counts_data);
    }

    public static void locate_in_bounds(SpatialPartition spatialPartition)
    {
        int n = Main.Memory.hull_count();

        // step 1: locate objects that are within bounds
        int x_subdivisions = spatialPartition.getX_subdivisions();
        physicsBuffer.x_sub_divisions = Pointer.to(arg_int(x_subdivisions));
        physicsBuffer.key_count_length = Pointer.to(arg_int(spatialPartition.getDirectoryLength()));

        long inbound_buf_size = (long) Sizeof.cl_int * n;
        cl_mem inbound_data = cl_new_buffer(inbound_buf_size);

        physicsBuffer.in_bounds = new MemoryBuffer(inbound_data);

        int[] size = arg_int(0);
        Pointer dst_size = Pointer.to(size);
        cl_mem size_data = cl_new_int_arg_buffer(dst_size);
        Pointer src_size = Pointer.to(size_data);

        clSetKernelArg(_k.get(Kernel.locate_in_bounds), 0, Sizeof.cl_mem, Pointer.to(mem_aabb_key_bank));
        clSetKernelArg(_k.get(Kernel.locate_in_bounds), 1, Sizeof.cl_mem, physicsBuffer.in_bounds.pointer());
        clSetKernelArg(_k.get(Kernel.locate_in_bounds), 2, Sizeof.cl_mem, src_size);

        k_call(command_queue, _k.get(Kernel.locate_in_bounds), arg_long(n));

        cl_read_buffer(size_data, Sizeof.cl_int, dst_size);

        clReleaseMemObject(size_data);

        physicsBuffer.set_candidate_buffer_count(size[0]);
    }

    public static void count_candidates()
    {
        long cand_buf_size = (long) Sizeof.cl_int2 * physicsBuffer.get_candidate_buffer_count();
        cl_mem cand_data = cl_new_buffer(cand_buf_size);
        physicsBuffer.candidate_counts = new MemoryBuffer(cand_data);

        clSetKernelArg(_k.get(Kernel.count_candidates), 0, Sizeof.cl_mem, Pointer.to(mem_aabb_key_bank));
        clSetKernelArg(_k.get(Kernel.count_candidates), 1, Sizeof.cl_mem, physicsBuffer.in_bounds.pointer());
        clSetKernelArg(_k.get(Kernel.count_candidates), 2, Sizeof.cl_mem, physicsBuffer.key_bank.pointer());
        clSetKernelArg(_k.get(Kernel.count_candidates), 3, Sizeof.cl_mem, physicsBuffer.key_counts.pointer());
        clSetKernelArg(_k.get(Kernel.count_candidates), 4, Sizeof.cl_mem, physicsBuffer.candidate_counts.pointer());
        clSetKernelArg(_k.get(Kernel.count_candidates), 5, Sizeof.cl_int, physicsBuffer.x_sub_divisions);
        clSetKernelArg(_k.get(Kernel.count_candidates), 6, Sizeof.cl_int, physicsBuffer.key_count_length);

        k_call(command_queue, _k.get(Kernel.count_candidates), arg_long(physicsBuffer.get_candidate_buffer_count()));
    }

    public static void count_matches()
    {
        int n = physicsBuffer.get_candidate_buffer_count();
        long offset_buf_size = (long) Sizeof.cl_int * n;
        cl_mem offset_data = cl_new_buffer(offset_buf_size);
        physicsBuffer.candidate_offsets = new MemoryBuffer(offset_data);

        int match_count = scan_key_candidates(physicsBuffer.candidate_counts.memory(), offset_data, n);
        physicsBuffer.set_candidate_match_count(match_count);

    }

    public static void aabb_collide()
    {
        long matches_buf_size = (long) Sizeof.cl_int * physicsBuffer.get_candidate_match_count();
        cl_mem matches_data = cl_new_buffer(matches_buf_size);
        physicsBuffer.matches = new MemoryBuffer(matches_data);

        long used_buf_size = (long) Sizeof.cl_int * physicsBuffer.get_candidate_buffer_count();
        cl_mem used_data = cl_new_buffer(used_buf_size);
        physicsBuffer.matches_used = new MemoryBuffer(used_data);

        // this buffer will contain the total number of candidates that were found
        int[] count = arg_int(0);
        Pointer dst_count = Pointer.to(count);
        cl_mem count_data = cl_new_int_arg_buffer(dst_count);
        Pointer src_count = Pointer.to(count_data);

        clSetKernelArg(_k.get(Kernel.aabb_collide), 0, Sizeof.cl_mem, Pointer.to(mem_aabb));
        clSetKernelArg(_k.get(Kernel.aabb_collide), 1, Sizeof.cl_mem, Pointer.to(mem_aabb_key_bank));
        clSetKernelArg(_k.get(Kernel.aabb_collide), 2, Sizeof.cl_mem, Pointer.to(mem_hull_flags));
        clSetKernelArg(_k.get(Kernel.aabb_collide), 3, Sizeof.cl_mem, physicsBuffer.candidate_counts.pointer());
        clSetKernelArg(_k.get(Kernel.aabb_collide), 4, Sizeof.cl_mem, physicsBuffer.candidate_offsets.pointer());
        clSetKernelArg(_k.get(Kernel.aabb_collide), 5, Sizeof.cl_mem, physicsBuffer.key_map.pointer());
        clSetKernelArg(_k.get(Kernel.aabb_collide), 6, Sizeof.cl_mem, physicsBuffer.key_bank.pointer());
        clSetKernelArg(_k.get(Kernel.aabb_collide), 7, Sizeof.cl_mem, physicsBuffer.key_counts.pointer());
        clSetKernelArg(_k.get(Kernel.aabb_collide), 8, Sizeof.cl_mem, physicsBuffer.key_offsets.pointer());
        clSetKernelArg(_k.get(Kernel.aabb_collide), 9, Sizeof.cl_mem, physicsBuffer.matches.pointer());
        clSetKernelArg(_k.get(Kernel.aabb_collide), 10, Sizeof.cl_mem, physicsBuffer.matches_used.pointer());
        clSetKernelArg(_k.get(Kernel.aabb_collide), 11, Sizeof.cl_mem, src_count);
        clSetKernelArg(_k.get(Kernel.aabb_collide), 12, Sizeof.cl_int, physicsBuffer.x_sub_divisions);
        clSetKernelArg(_k.get(Kernel.aabb_collide), 13, Sizeof.cl_int, physicsBuffer.key_count_length);

        k_call(command_queue, _k.get(Kernel.aabb_collide), arg_long(physicsBuffer.get_candidate_buffer_count()));

        cl_read_buffer(count_data, Sizeof.cl_int, dst_count);

        clReleaseMemObject(count_data);

        physicsBuffer.set_candidate_count(count[0]);
    }

    public static void finalize_candidates()
    {
        if (physicsBuffer.get_candidate_count() > 0)
        {
            // create an empty buffer that the kernel will use to store finalized candidates
            long final_buf_size = (long) Sizeof.cl_int2 * physicsBuffer.get_candidate_count();
            cl_mem finals_data = cl_new_buffer(final_buf_size);
            Pointer src_finals = Pointer.to(finals_data);

            // the kernel will use this value as an internal atomic counter, always initialize to zero
            int[] counter = new int[]{0};
            Pointer dst_counter = Pointer.to(counter);
            cl_mem counter_data = cl_new_int_arg_buffer(dst_counter);
            Pointer src_counter = Pointer.to(counter_data);

            physicsBuffer.set_final_size(final_buf_size);

            physicsBuffer.candidates = new MemoryBuffer(finals_data);

            clSetKernelArg(_k.get(Kernel.finalize_candidates), 0, Sizeof.cl_mem, physicsBuffer.candidate_counts.pointer());
            clSetKernelArg(_k.get(Kernel.finalize_candidates), 1, Sizeof.cl_mem, physicsBuffer.candidate_offsets.pointer());
            clSetKernelArg(_k.get(Kernel.finalize_candidates), 2, Sizeof.cl_mem, physicsBuffer.matches.pointer());
            clSetKernelArg(_k.get(Kernel.finalize_candidates), 3, Sizeof.cl_mem, physicsBuffer.matches_used.pointer());
            clSetKernelArg(_k.get(Kernel.finalize_candidates), 4, Sizeof.cl_mem, src_counter);
            clSetKernelArg(_k.get(Kernel.finalize_candidates), 5, Sizeof.cl_mem, src_finals);

            k_call(command_queue, _k.get(Kernel.finalize_candidates), arg_long(physicsBuffer.get_candidate_buffer_count()));

            clReleaseMemObject(counter_data);
        }
    }

    public static void sat_collide()
    {
        if (physicsBuffer.candidates == null)
        {
            return;
        }

        int candidatesSize = (int) physicsBuffer.get_final_size() / Sizeof.cl_int;

        // Set the work-item dimensions
        long[] global_work_size = new long[]{ candidatesSize / 2 }; // candidates are pairs of integer indices

        // Set the arguments for the kernel
        clSetKernelArg(_k.get(Kernel.sat_collide), 0, Sizeof.cl_mem, physicsBuffer.candidates.pointer());
        clSetKernelArg(_k.get(Kernel.sat_collide), 1, Sizeof.cl_mem, Pointer.to(mem_hulls));
        clSetKernelArg(_k.get(Kernel.sat_collide), 2, Sizeof.cl_mem, Pointer.to(mem_armatures));
        clSetKernelArg(_k.get(Kernel.sat_collide), 3, Sizeof.cl_mem, Pointer.to(mem_hull_element_tables));
        clSetKernelArg(_k.get(Kernel.sat_collide), 4, Sizeof.cl_mem, Pointer.to(mem_hull_flags));
        clSetKernelArg(_k.get(Kernel.sat_collide), 5, Sizeof.cl_mem, Pointer.to(mem_points));
        clSetKernelArg(_k.get(Kernel.sat_collide), 6, Sizeof.cl_mem, Pointer.to(mem_edges));

        k_call(command_queue, _k.get(Kernel.sat_collide), global_work_size);
    }

    public static void resolve_constraints(int edge_steps)
    {
        boolean lastStep;
        long[] global_work_size = new long[]{Main.Memory.hull_count()};
        for (int i = 0; i < edge_steps; i++)
        {
            lastStep = i == edge_steps - 1;
            int n = lastStep
                ? 1
                : 0;
            int a = 0;
            clSetKernelArg(_k.get(Kernel.resolve_constraints), a++, Sizeof.cl_mem, Pointer.to(mem_hull_element_tables));
            clSetKernelArg(_k.get(Kernel.resolve_constraints), a++, Sizeof.cl_mem, Pointer.to(mem_aabb_key_bank));
            clSetKernelArg(_k.get(Kernel.resolve_constraints), a++, Sizeof.cl_mem, Pointer.to(mem_points));
            clSetKernelArg(_k.get(Kernel.resolve_constraints), a++, Sizeof.cl_mem, Pointer.to(mem_edges));
            clSetKernelArg(_k.get(Kernel.resolve_constraints), a++, Sizeof.cl_int, Pointer.to(new int[]{n}));

            //gl_acquire(vertex_mem);
            k_call(command_queue, _k.get(Kernel.resolve_constraints), global_work_size);
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
        long local_buffer_size = Sizeof.cl_int * max_scan_block_size;

        clSetKernelArg(_k.get(Kernel.scan_int_single_block), 0, Sizeof.cl_mem, Pointer.to(d_data));
        clSetKernelArg(_k.get(Kernel.scan_int_single_block), 1, local_buffer_size, null);
        clSetKernelArg(_k.get(Kernel.scan_int_single_block), 2, Sizeof.cl_int, Pointer.to(new int[]{n}));

        k_call(command_queue, _k.get(Kernel.scan_int_single_block), local_work_default, local_work_default);
    }

    private static void scan_multi_block_int(cl_mem d_data, int n, int k)
    {
        long local_buffer_size = Sizeof.cl_int * max_scan_block_size;
        long gx = k * max_scan_block_size;
        long[] global_work_size = arg_long(gx);
        int part_size = k * 2;
        long part_buf_size = ((long) Sizeof.cl_int * ((long) part_size));
        cl_mem part_data = cl_new_buffer(part_buf_size);
        Pointer src_data = Pointer.to(d_data);
        Pointer src_part = Pointer.to(part_data);
        Pointer src_n = Pointer.to(new int[]{n});

        clSetKernelArg(_k.get(Kernel.scan_int_multi_block), 0, Sizeof.cl_mem, src_data);
        clSetKernelArg(_k.get(Kernel.scan_int_multi_block), 1, local_buffer_size, null);
        clSetKernelArg(_k.get(Kernel.scan_int_multi_block), 2, Sizeof.cl_mem, src_part);
        clSetKernelArg(_k.get(Kernel.scan_int_multi_block), 3, Sizeof.cl_int, src_n);

        k_call(command_queue, _k.get(Kernel.scan_int_multi_block), global_work_size, local_work_default);

        scan_int(part_data, part_size);

        clSetKernelArg(_k.get(Kernel.complete_int_multi_block), 0, Sizeof.cl_mem, src_data);
        clSetKernelArg(_k.get(Kernel.complete_int_multi_block), 1, local_buffer_size, null);
        clSetKernelArg(_k.get(Kernel.complete_int_multi_block), 2, Sizeof.cl_mem, src_part);
        clSetKernelArg(_k.get(Kernel.complete_int_multi_block), 3, Sizeof.cl_int, src_n);

        k_call(command_queue, _k.get(Kernel.complete_int_multi_block), global_work_size, local_work_default);

        clReleaseMemObject(part_data);
    }

    private static void scan_single_block_int_out(cl_mem d_data, cl_mem o_data, int n)
    {
        long local_buffer_size = Sizeof.cl_int * max_scan_block_size;
        Pointer src_data = Pointer.to(d_data);
        Pointer dst_data = Pointer.to(o_data);

        clSetKernelArg(_k.get(Kernel.scan_int_single_block_out), 0, Sizeof.cl_mem, src_data);
        clSetKernelArg(_k.get(Kernel.scan_int_single_block_out), 2, Sizeof.cl_mem, dst_data);
        clSetKernelArg(_k.get(Kernel.scan_int_single_block_out), 2, local_buffer_size, null);
        clSetKernelArg(_k.get(Kernel.scan_int_single_block_out), 3, Sizeof.cl_int, Pointer.to(new int[]{n}));

        k_call(command_queue, _k.get(Kernel.scan_int_single_block_out), local_work_default, local_work_default);
    }

    private static void scan_multi_block_int_out(cl_mem d_data, cl_mem o_data, int n, int k)
    {
        long local_buffer_size = Sizeof.cl_int * max_scan_block_size;
        long gx = k * max_scan_block_size;
        long[] global_work_size = arg_long(gx);
        int part_size = k * 2;
        long part_buf_size = ((long) Sizeof.cl_int * ((long) part_size));
        cl_mem part_data = cl_new_buffer(part_buf_size);
        Pointer src_data = Pointer.to(d_data);
        Pointer src_part = Pointer.to(part_data);
        Pointer dst_data = Pointer.to(o_data);
        Pointer src_n = Pointer.to(new int[]{n});

        clSetKernelArg(_k.get(Kernel.scan_int_multi_block_out), 0, Sizeof.cl_mem, src_data);
        clSetKernelArg(_k.get(Kernel.scan_int_multi_block_out), 1, Sizeof.cl_mem, dst_data);
        clSetKernelArg(_k.get(Kernel.scan_int_multi_block_out), 2, local_buffer_size, null);
        clSetKernelArg(_k.get(Kernel.scan_int_multi_block_out), 3, Sizeof.cl_mem, src_part);
        clSetKernelArg(_k.get(Kernel.scan_int_multi_block_out), 4, Sizeof.cl_int, src_n);

        k_call(command_queue, _k.get(Kernel.scan_int_multi_block_out), global_work_size, local_work_default);

        scan_int(part_data, part_size);

        clSetKernelArg(_k.get(Kernel.complete_int_multi_block_out), 0, Sizeof.cl_mem, dst_data);
        clSetKernelArg(_k.get(Kernel.complete_int_multi_block_out), 1, local_buffer_size, null);
        clSetKernelArg(_k.get(Kernel.complete_int_multi_block_out), 2, Sizeof.cl_mem, src_part);
        clSetKernelArg(_k.get(Kernel.complete_int_multi_block_out), 3, Sizeof.cl_int, src_n);

        k_call(command_queue, _k.get(Kernel.complete_int_multi_block_out), global_work_size, local_work_default);

        clReleaseMemObject(part_data);
    }

    private static int scan_single_block_candidates_out(cl_mem d_data, cl_mem o_data, int n)
    {
        long local_buffer_size = Sizeof.cl_int * max_scan_block_size;
        Pointer src_data = Pointer.to(d_data);
        Pointer dst_data = Pointer.to(o_data);

        int[] sz = new int[]{0};
        Pointer dst_size = Pointer.to(sz);
        cl_mem size_data = cl_new_buffer(Sizeof.cl_int);
        Pointer src_size = Pointer.to(size_data);

        clSetKernelArg(_k.get(Kernel.scan_candidates_single_block_out), 0, Sizeof.cl_mem, src_data);
        clSetKernelArg(_k.get(Kernel.scan_candidates_single_block_out), 1, Sizeof.cl_mem, dst_data);
        clSetKernelArg(_k.get(Kernel.scan_candidates_single_block_out), 2, Sizeof.cl_mem, src_size);
        clSetKernelArg(_k.get(Kernel.scan_candidates_single_block_out), 3, local_buffer_size, null);
        clSetKernelArg(_k.get(Kernel.scan_candidates_single_block_out), 4, Sizeof.cl_int, Pointer.to(new int[]{n}));

        k_call(command_queue, _k.get(Kernel.scan_candidates_single_block_out), local_work_default, local_work_default);

        cl_read_buffer(size_data, Sizeof.cl_int, dst_size);

        clReleaseMemObject(size_data);

        return sz[0];
    }

    private static int scan_multi_block_candidates_out(cl_mem d_data, cl_mem o_data, int n, int k)
    {
        long local_buffer_size = Sizeof.cl_int * max_scan_block_size;
        long gx = k * max_scan_block_size;
        long[] global_work_size = arg_long(gx);
        int part_size = k * 2;
        long part_buf_size = ((long) Sizeof.cl_int * ((long) part_size));
        cl_mem p_data = cl_new_buffer(part_buf_size);
        Pointer src_data = Pointer.to(d_data);
        Pointer src_part = Pointer.to(p_data);
        Pointer dst_data = Pointer.to(o_data);
        Pointer src_n = Pointer.to(new int[]{n});

        clSetKernelArg(_k.get(Kernel.scan_candidates_multi_block_out), 0, Sizeof.cl_mem, src_data);
        clSetKernelArg(_k.get(Kernel.scan_candidates_multi_block_out), 1, Sizeof.cl_mem, dst_data);
        clSetKernelArg(_k.get(Kernel.scan_candidates_multi_block_out), 2, local_buffer_size, null);
        clSetKernelArg(_k.get(Kernel.scan_candidates_multi_block_out), 3, Sizeof.cl_mem, src_part);
        clSetKernelArg(_k.get(Kernel.scan_candidates_multi_block_out), 4, Sizeof.cl_int, src_n);

        k_call(command_queue, _k.get(Kernel.scan_candidates_multi_block_out), global_work_size, local_work_default);

        scan_int(p_data, part_size);

        int[] sz = new int[]{0};
        Pointer dst_size = Pointer.to(sz);
        cl_mem size_data = cl_new_buffer(Sizeof.cl_int);
        Pointer src_size = Pointer.to(size_data);

        clSetKernelArg(_k.get(Kernel.complete_candidates_multi_block_out), 0, Sizeof.cl_mem, src_data);
        clSetKernelArg(_k.get(Kernel.complete_candidates_multi_block_out), 1, Sizeof.cl_mem, dst_data);
        clSetKernelArg(_k.get(Kernel.complete_candidates_multi_block_out), 2, Sizeof.cl_mem, src_size);
        clSetKernelArg(_k.get(Kernel.complete_candidates_multi_block_out), 3, local_buffer_size, null);
        clSetKernelArg(_k.get(Kernel.complete_candidates_multi_block_out), 4, Sizeof.cl_mem, src_part);
        clSetKernelArg(_k.get(Kernel.complete_candidates_multi_block_out), 5, Sizeof.cl_int, src_n);

        k_call(command_queue, _k.get(Kernel.complete_candidates_multi_block_out), global_work_size, local_work_default);

        cl_read_buffer(size_data, Sizeof.cl_int, dst_size);

        clReleaseMemObject(p_data);
        clReleaseMemObject(size_data);

        return sz[0];
    }

    private static int scan_single_block_key(cl_mem d_data2, int n)
    {
        long local_buffer_size = Sizeof.cl_int * max_scan_block_size;
        Pointer src_data2 = Pointer.to(d_data2);

        int[] sz = new int[]{0};
        Pointer dst_size = Pointer.to(sz);
        cl_mem size_data = cl_new_buffer(Sizeof.cl_int);
        Pointer src_size = Pointer.to(size_data);
        Pointer src_n = Pointer.to(new int[]{n});

        clSetKernelArg(_k.get(Kernel.scan_bounds_single_block), 0, Sizeof.cl_mem, src_data2);
        clSetKernelArg(_k.get(Kernel.scan_bounds_single_block), 1, Sizeof.cl_mem, src_size);
        clSetKernelArg(_k.get(Kernel.scan_bounds_single_block), 2, local_buffer_size, null);
        clSetKernelArg(_k.get(Kernel.scan_bounds_single_block), 3, Sizeof.cl_int, src_n);

        k_call(command_queue, _k.get(Kernel.scan_bounds_single_block), local_work_default, local_work_default);

        cl_read_buffer(size_data, Sizeof.cl_int, dst_size);

        clReleaseMemObject(size_data);

        return sz[0];
    }

    private static int scan_multi_block_key(cl_mem d_data2, int n, int k)
    {
        long local_buffer_size = Sizeof.cl_int * max_scan_block_size;
        long gx = k * max_scan_block_size;
        long[] global_work_size = arg_long(gx);
        int part_size = k * 2;
        long part_buf_size = ((long) Sizeof.cl_int * ((long) part_size));
        cl_mem p_data = cl_new_buffer(part_buf_size);
        Pointer src_data2 = Pointer.to(d_data2);
        Pointer src_part = Pointer.to(p_data);
        Pointer src_n = Pointer.to(new int[]{n});

        clSetKernelArg(_k.get(Kernel.scan_bounds_multi_block), 0, Sizeof.cl_mem, src_data2);
        clSetKernelArg(_k.get(Kernel.scan_bounds_multi_block), 1, local_buffer_size, null);
        clSetKernelArg(_k.get(Kernel.scan_bounds_multi_block), 2, Sizeof.cl_mem, src_part);
        clSetKernelArg(_k.get(Kernel.scan_bounds_multi_block), 3, Sizeof.cl_int, src_n);

        k_call(command_queue, _k.get(Kernel.scan_bounds_multi_block), global_work_size, local_work_default);

        scan_int(p_data, part_size);

        int[] sz = new int[1];
        Pointer dst_size = Pointer.to(sz);
        cl_mem size_data = cl_new_buffer(Sizeof.cl_int);
        Pointer src_size = Pointer.to(size_data);

        clSetKernelArg(_k.get(Kernel.complete_bounds_multi_block), 0, Sizeof.cl_mem, src_data2);
        clSetKernelArg(_k.get(Kernel.complete_bounds_multi_block), 1, Sizeof.cl_mem, src_size);
        clSetKernelArg(_k.get(Kernel.complete_bounds_multi_block), 2, local_buffer_size, null);
        clSetKernelArg(_k.get(Kernel.complete_bounds_multi_block), 3, Sizeof.cl_mem, src_part);
        clSetKernelArg(_k.get(Kernel.complete_bounds_multi_block), 4, Sizeof.cl_int, src_n);

        k_call(command_queue, _k.get(Kernel.complete_bounds_multi_block), global_work_size, local_work_default);

        cl_read_buffer(size_data, Sizeof.cl_int, dst_size);

        clReleaseMemObject(size_data);
        clReleaseMemObject(p_data);

        return sz[0];
    }

    //#endregion
}
