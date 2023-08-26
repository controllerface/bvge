package com.controllerface.bvge.cl;

import com.controllerface.bvge.Main;
import com.controllerface.bvge.cl.kernels.*;
import com.controllerface.bvge.cl.programs.*;
import com.controllerface.bvge.ecs.systems.physics.GPUMemory;
import com.controllerface.bvge.ecs.systems.physics.PhysicsBuffer;
import com.controllerface.bvge.ecs.systems.physics.UniformGrid;
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
    //#region Constants

    private static final long FLAGS_WRITE_GPU = CL_MEM_READ_WRITE;
    private static final long FLAGS_WRITE_CPU_COPY = CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR;
    private static final long FLAGS_READ_CPU_COPY = CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR;

    private static final Pointer ZERO_PATTERN = Pointer.to(new int[]{0});

    //#endregion

    //#region Workgroup Variables

    /**
     * These values are re-calculated at startup to match the user's hardware. The max work group is the
     * largest group of calculations that can be done in a single "warp" or "wave" of GPU processing.
     * Related to this, we store a max scan block, which is used for variants of the prefix scan kernels.
     * The local work default is simply the max group size formatted as a single element argument array,
     * making it simpler to use for Open Cl calls which expect that format.
     */
    private static long max_work_group_size = 0;
    private static long max_scan_block_size = 0;
    private static long[] local_work_default = arg_long(0);

    /**
     * This convenience array defines a work group size of 1, which is primarily used for setting up
     * data buffers at startup. Generally speaking, kernels of this size should be used sparingly, and
     * code should favor making bulk calls, however there are certain use cases where it makes sense to
     * perform a single operation on some GPU memory.
     */
    private static final long[] global_single_size = arg_long(1);

    //#endregion

    //#region Class Variables

    /**
     * The Open CL command queue that this class uses to issue GPU commands.
     */
    private static cl_command_queue command_queue;

    /**
     * The Open CL context associated with this class.
     */
    private static cl_context context;

    /**
     * An array of devices that support being used with Open CL. In practice, this should
     * only ever have single element, and that device should be the main GPU in the system.
     */
    private static cl_device_id[] device_ids;

    /**
     * Assists in managing data buffers and other variables used for physics calculations.
     * These properties and buffers are co-located within this single structure to make it
     * easier to reason about the logic and add or remove objects as needed for new features.
     */
    private static PhysicsBuffer physics_buffer;

    //#endregion

    //#region GPU Program Objects

    /**
     * Enumerates all existing GPU programs. Programs contain one or more "kernels". A kernel is
     * effectively an entry point into a small, self-contained, function that operates on memory.
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
        prepare_bounds(new com.controllerface.bvge.cl.programs.PrepareBounds()),
        prepare_edges(new PrepareEdges()),
        prepare_transforms(new PrepareTransforms()),
        resolve_constraints(new ResolveConstraints()),
        sat_collide(new SatCollide()),
        scan_candidates(new ScanCandidates()),
        scan_int_array(new ScanIntArray()),
        scan_int_array_out(new ScanIntArrayOut()),
        scan_key_bank(new ScanKeyBank()),

        ;

        private final GPUProgram gpu;

        Program(GPUProgram program)
        {
            this.gpu = program;
        }

        cl_kernel get_kernel(Kernel kernel)
        {
            return gpu.kernels.get(kernel);
        }
    }

    //#endregion

    //#region GPU Kernel Objects

    /**
     * After init, all kernels are loaded into this map, making named access of them simple.
     */
    // todo: this will be removed in favor of explicit uses of kernels in defined programs
    private static final Map<Kernel, cl_kernel> _k = new HashMap<>();

    /**
     * Kernel function names. Program implementations use this enum to instantiate kernel objects
     * with a specific name, which are then called using the various methods of the GPU class.
     */
    public enum Kernel
    {
        aabb_collide,
        animate_hulls,
        apply_reactions,
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
        sort_reactions,
        update_accel,

        ;

        GPUKernel gpu;

        public void set_kernel(GPUKernel gpuKernel)
        {
            this.gpu = gpuKernel;
        }
    }

    //#endregion

    //#region GPU Memory Objects

    /**
     * Memory that is shared between Open CL and Open GL contexts
     */
    private static final HashMap<Integer, cl_mem> shared_mem = new LinkedHashMap<>();

    /**
     * Memory buffers that store data used within the various kernel functions. Each buffer
     * has a different layout, but will align to an Open CL supported primitive type, such as
     * int, float or some vectorized type like, int2 or float4.
     */
    private enum Memory
    {
        /**
         * Individual points (vertices) of tracked physics hulls. Values are float4 with the following mappings:
         * -
         * x: current x position
         * y: current y position
         * z: previous x position
         * w: previous y position
         * -
         */
        points(Sizeof.cl_float4),

        /**
         * reaction counts for points on tracked physics hulls. Values are int with the following mapping:
         * -
         * value: reaction count
         * -
         */
        point_reactions(Sizeof.cl_int),

        /**
         * reaction offsets for points on tracked physics hulls. Values are int with the following mapping:
         * -
         * value: reaction buffer offset
         * -
         */
        point_offsets(Sizeof.cl_int),

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
        edges(Sizeof.cl_float4),

        /**
         * Positions of tracked hulls. Values are float4 with the following mappings:
         * -
         * x: current x position
         * y: current y position
         * z: scale x
         * w: scale y
         * -
         */
        hulls(Sizeof.cl_float4),

        /**
         * Axis-aligned bounding boxes of tracked physics hulls. Values are float4 with the following mappings:
         * -
         * x: corner x position
         * y: corner y position
         * z: width
         * w: height
         * -
         */
        aabb(Sizeof.cl_float4),

        /**
         * Rotation information about tracked physics hulls. Values are float2 with the following mappings:
         * -
         * x: initial reference angle
         * y: current rotation
         * -
         */
        hull_rotation(Sizeof.cl_float2),

        /**
         * Indexing table for tracked physics hulls. Values are int4 with the following mappings:
         * -
         * x: start point index
         * y: end point index
         * z: start edge index
         * w: end edge index
         * -
         */
        hull_element_table(Sizeof.cl_int4),

        /**
         * Flags that related to tracked physics hulls. Values are int2 with the following mappings:
         * -
         * x: hull flags
         * y: armature id
         * -
         */
        hull_flags(Sizeof.cl_int2),

        /**
         * Spatial partition index information for tracked physics hulls. Values are int4 with the following mappings:
         * -
         * x: minimum x key index
         * y: maximum x key index
         * z: minimum y key index
         * w: maximum y key index
         * -
         */
        aabb_index(Sizeof.cl_int4),

        /**
         * Spatial partition key bank information for tracked physics hulls. Values are int2 with the following mappings:
         * -
         * x: key bank offset
         * y: key bank size
         * -
         */
        aabb_key_table(Sizeof.cl_int2),

        /**
         * Vertex information for loaded models. Values are float2 with the following mappings:
         * -
         * x: x position
         * y: y position
         * -
         */
        vertex_references(Sizeof.cl_float2),

        /**
         * Indexing table for points of tracked physics hulls. Values are int2 with the following mappings:
         * -
         * x: reference vertex index
         * y: bone index (todo: also used as a proxy for hull ID, based on alignment, but they should be separate)
         * -
         */
        vertex_table(Sizeof.cl_int2),

        /**
         * Bone offset reference matrices of loaded models. Values are float16 with the following mappings:
         * -
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
         * -
         */
        bone_references(Sizeof.cl_float16),

        /**
         * Bone offset animation matrices of tracked physics hulls. Values are float16 with the following mappings:
         * -
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
         * -
         */
        bone_instances(Sizeof.cl_float16),

        /**
         * Reference bone index of bones used for tracked physics hulls. Values are int with the following mapping:
         * -
         * value: bone reference index
         * -
         */
        bone_index(Sizeof.cl_int),

        /**
         * Armature information for tracked physics hulls. Values are float4 with the following mappings:
         * -
         * x: current x position
         * y: current y position
         * z: previous x position
         * w: previous y position
         * -
         */
        armatures(Sizeof.cl_float4),

        /**
         * Hull index of root hull for an armature. Values are int with the following mapping:
         * -
         * value: hull index
         * -
         */
        armature_flags(Sizeof.cl_int),

        /**
         * Acceleration value of an armature. Values are float2 with the following mappings:
         * -
         * x: current x acceleration
         * y: current y acceleration
         * -
         */
        armature_accel(Sizeof.cl_float2),

        ;

        GPUMemory gpu;
        final int size;
        int length;

        Memory(int valueSize)
        {
            size = valueSize;
        }

        public void init(int buffer_length)
        {
            this.length = buffer_length * size;
            var mem = cl_new_buffer(this.length);
            this.gpu = new GPUMemory(mem);
            clear();
        }

        public void clear()
        {
            cl_zero_buffer(this.gpu.memory(), this.length);
        }
    }

    //#endregion

    //#region Public API

    public static cl_program gpu_p(List<String> src_strings)
    {
        String[] src = src_strings.toArray(new String[]{});
        return CLUtils.cl_p(context, device_ids, src);
    }

    public static void set_physics_buffer(PhysicsBuffer physics_buffer)
    {
        GPU.physics_buffer = physics_buffer;
    }

    public static void init(int max_hulls, int max_points)
    {
        device_ids = init_device();

        var device = device_ids[0];

        System.out.println("-------- OPEN CL DEVICE -----------");
        System.out.println(getString(device, CL_DEVICE_VENDOR));
        System.out.println(getString(device, CL_DEVICE_NAME));
        System.out.println(getString(device, CL_DRIVER_VERSION));
        System.out.println("-----------------------------------\n");

        max_work_group_size = getSize(device, CL_DEVICE_MAX_WORK_GROUP_SIZE);
        max_scan_block_size = max_work_group_size * 2;
        local_work_default = arg_long(max_work_group_size);

        // initialize kernel programs
        for (var program : Program.values())
        {
            program.gpu.init();
            _k.putAll(program.gpu.kernels);
        }

        //OpenCLUtils.debugDeviceDetails(device_ids);

        // create memory buffers
        init_memory(max_hulls, max_points);

        // Create re-usable kernel objects
        init_kernels();
    }

    public static void destroy()
    {
        for (Memory buffer : Memory.values())
        {
            Optional.ofNullable(buffer.gpu)
                .ifPresent(GPUMemory::release);
        }

        shared_mem.values().forEach(CL::clReleaseMemObject);

        for (Program program : Program.values())
        {
            Optional.ofNullable(program.gpu)
                .ifPresent(GPUProgram::destroy);
        }
        clReleaseCommandQueue(command_queue);
        clReleaseContext(context);
    }

    //#endregion

    //#region Utility Methods

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

    private static int work_group_count(int n)
    {
        return (int) Math.ceil((float) n / (float) max_scan_block_size);
    }

    //#endregion

    //#region Init Methods

    private static cl_device_id[] init_device()
    {
        // The platform, device type and device number
        // that will be used
        int platformIndex = 0;
        long deviceType = CL_DEVICE_TYPE_ALL;
        int deviceIndex = 0;

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
        var device = devices[deviceIndex];

        var device_ids = new cl_device_id[]{device};

        var dc = wglGetCurrentDC();
        var ctx = wglGetCurrentContext();

        // todo: the above code is windows specific add linux code path,
        //  should look something like this:
        // var ctx = glXGetCurrentContext();
        // var dc = glXGetCurrentDrawable();
        // contextProperties.addProperty(CL_GLX_DISPLAY_KHR, dc);

        // Initialize the context properties
        var contextProperties = new cl_context_properties();
        contextProperties.addProperty(CL_CONTEXT_PLATFORM, platform);
        contextProperties.addProperty(CL_GL_CONTEXT_KHR, ctx);
        contextProperties.addProperty(CL_WGL_HDC_KHR, dc);

//        OpenCL.printDeviceDetails(device_ids);
        // Create a context for the selected device
        context = clCreateContext(
            contextProperties, 1, device_ids,
            null, null, null);

        // Create a command-queue for the selected device
        var properties = new cl_queue_properties();
        command_queue = clCreateCommandQueueWithProperties(
            context, device, properties, null);

        return device_ids;

    }

    private static void init_memory(int max_hulls, int max_points)
    {
        // todo: there should be more granularity than just max hulls and points. There should be
        //  limits on armatures and other data types.

        Memory.armature_accel.init(max_hulls);
        Memory.hull_rotation.init(max_hulls);
        Memory.hull_element_table.init(max_hulls);
        Memory.hull_flags.init(max_hulls);
        Memory.aabb_index.init(max_hulls);
        Memory.aabb_key_table.init(max_hulls);
        Memory.hulls.init(max_hulls);
        Memory.aabb.init(max_hulls);
        Memory.points.init(max_points);
        Memory.point_reactions.init(max_points);
        Memory.point_offsets.init(max_points);
        Memory.edges.init(max_points);
        Memory.vertex_table.init(max_points);
        Memory.vertex_references.init(max_points);
        Memory.bone_references.init(max_points);
        Memory.bone_instances.init(max_points);
        Memory.bone_index.init(max_points);
        Memory.armatures.init(max_points);
        Memory.armature_flags.init(max_points);

        // Debugging info
        int total = Memory.hulls.length
            + Memory.armature_accel.length
            + Memory.hull_rotation.length
            + Memory.hull_element_table.length
            + Memory.hull_flags.length
            + Memory.aabb.length
            + Memory.aabb_index.length
            + Memory.aabb_key_table.length
            + Memory.points.length
            + Memory.point_reactions.length
            + Memory.point_offsets.length
            + Memory.edges.length
            + Memory.vertex_table.length
            + Memory.vertex_references.length
            + Memory.bone_references.length
            + Memory.bone_instances.length
            + Memory.bone_index.length
            + Memory.armatures.length
            + Memory.armature_flags.length;

        System.out.println("------------- BUFFERS -------------");
        System.out.println("points            : " + Memory.points.length);
        System.out.println("edges             : " + Memory.edges.length);
        System.out.println("hulls             : " + Memory.hulls.length);
        System.out.println("acceleration      : " + Memory.armature_accel.length);
        System.out.println("rotation          : " + Memory.hull_rotation.length);
        System.out.println("element table     : " + Memory.hull_element_table.length);
        System.out.println("flags             : " + Memory.hull_flags.length);
        System.out.println("point reactions   : " + Memory.point_reactions.length);
        System.out.println("point offsets     : " + Memory.point_offsets.length);
        System.out.println("bounding box      : " + Memory.aabb.length);
        System.out.println("spatial index     : " + Memory.aabb_index.length);
        System.out.println("spatial key bank  : " + Memory.aabb_key_table.length);
        System.out.println("vertex table      : " + Memory.vertex_table.length);
        System.out.println("vertex references : " + Memory.vertex_references.length);
        System.out.println("bone references   : " + Memory.bone_references.length);
        System.out.println("bone instances    : " + Memory.bone_instances.length);
        System.out.println("bone index        : " + Memory.bone_index.length);
        System.out.println("armatures         : " + Memory.armatures.length);
        System.out.println("armature flags    : " + Memory.armature_flags.length);
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
    private static void init_kernels()
    {
        var bounds_k = new PrepareBounds_k(command_queue, Program.prepare_bounds.gpu);
        bounds_k.set_aabb(Memory.aabb.gpu.pointer());
        Kernel.prepare_bounds.set_kernel(bounds_k);

        var transforms_k = new PrepareTransforms_k(command_queue, Program.prepare_transforms.gpu);
        transforms_k.set_hulls(Memory.hulls.gpu.pointer());
        transforms_k.set_rotations(Memory.hull_rotation.gpu.pointer());
        Kernel.prepare_transforms.set_kernel(transforms_k);

        var edges_k = new PrepareEdges_k(command_queue, Program.prepare_edges.gpu);
        edges_k.set_points(Memory.points.gpu.pointer());
        edges_k.set_edges(Memory.edges.gpu.pointer());
        Kernel.prepare_edges.set_kernel(edges_k);

        var bones_k = new PrepareBones_k(command_queue, Program.prepare_bones.gpu);
        bones_k.set_bone_instances(Memory.bone_instances.gpu.pointer());
        bones_k.set_bone_references(Memory.bone_references.gpu.pointer());
        bones_k.set_bone_index(Memory.bone_index.gpu.pointer());
        bones_k.set_hulls(Memory.hulls.gpu.pointer());
        bones_k.set_armatures(Memory.armatures.gpu.pointer());
        bones_k.set_hull_flags(Memory.hull_flags.gpu.pointer());
        Kernel.prepare_bones.set_kernel(bones_k);

        var sat_collide_k = new SatCollide_k(command_queue, Program.sat_collide.gpu);
        sat_collide_k.set_hulls(Memory.hulls.gpu.pointer());
        sat_collide_k.set_armatures(Memory.armatures.gpu.pointer());
        sat_collide_k.set_element_tables(Memory.hull_element_table.gpu.pointer());
        sat_collide_k.set_hull_flags(Memory.hull_flags.gpu.pointer());
        sat_collide_k.set_points(Memory.points.gpu.pointer());
        sat_collide_k.set_edges(Memory.edges.gpu.pointer());
        sat_collide_k.set_reactions(Memory.point_reactions.gpu.pointer());
        Kernel.sat_collide.set_kernel(sat_collide_k);

        var sort_reactions_k = new SortReactions_k(command_queue, Program.sat_collide.gpu);
        sort_reactions_k.set_reactions(Memory.point_reactions.gpu.pointer());
        sort_reactions_k.set_offsets(Memory.point_offsets.gpu.pointer());
        Kernel.sort_reactions.set_kernel(sort_reactions_k);

        var apply_reactions_k = new ApplyReactions_k(command_queue, Program.sat_collide.gpu);
        apply_reactions_k.set_points(Memory.points.gpu.pointer());
        apply_reactions_k.set_point_reactions(Memory.point_reactions.gpu.pointer());
        apply_reactions_k.set_point_offsets(Memory.point_offsets.gpu.pointer());
        Kernel.apply_reactions.set_kernel(apply_reactions_k);


        // todo: convert those below the lines to concrete classes





        var gpu_create_point = new GPUKernel(command_queue, _k.get(Kernel.create_point), 5);
        gpu_create_point.new_arg(0, Sizeof.cl_mem, Memory.points.gpu.pointer());
        gpu_create_point.new_arg(1, Sizeof.cl_mem, Memory.vertex_table.gpu.pointer());
        gpu_create_point.def_arg(2, Sizeof.cl_int);
        gpu_create_point.def_arg(3, Sizeof.cl_float4);
        gpu_create_point.def_arg(4, Sizeof.cl_int2);
        Kernel.create_point.set_kernel(gpu_create_point);

        var gpu_create_edge = new GPUKernel(command_queue, _k.get(Kernel.create_edge), 3);
        gpu_create_edge.new_arg(0, Sizeof.cl_mem, Memory.edges.gpu.pointer());
        gpu_create_edge.def_arg(1, Sizeof.cl_int);
        gpu_create_edge.def_arg(2, Sizeof.cl_float4);
        Kernel.create_edge.set_kernel(gpu_create_edge);

        var create_armature = new GPUKernel(command_queue, _k.get(Kernel.create_armature), 5);
        create_armature.new_arg(0, Sizeof.cl_mem, Memory.armatures.gpu.pointer());
        create_armature.new_arg(1, Sizeof.cl_mem, Memory.armature_flags.gpu.pointer());
        create_armature.def_arg(2, Sizeof.cl_int);
        create_armature.def_arg(3, Sizeof.cl_float4);
        create_armature.def_arg(4, Sizeof.cl_int);
        Kernel.create_armature.set_kernel(create_armature);

        var create_vertex_ref = new GPUKernel(command_queue, _k.get(Kernel.create_vertex_reference), 3);
        create_vertex_ref.new_arg(0, Sizeof.cl_mem, Memory.vertex_references.gpu.pointer());
        create_vertex_ref.def_arg(1, Sizeof.cl_int);
        create_vertex_ref.def_arg(2, Sizeof.cl_float2);
        Kernel.create_vertex_reference.set_kernel(create_vertex_ref);

        var create_bone_ref = new GPUKernel(command_queue, _k.get(Kernel.create_bone_reference), 3);
        create_bone_ref.new_arg(0, Sizeof.cl_mem, Memory.bone_references.gpu.pointer());
        create_bone_ref.def_arg(1, Sizeof.cl_int);
        create_bone_ref.def_arg(2, Sizeof.cl_float16);
        Kernel.create_bone_reference.set_kernel(create_bone_ref);

        var create_bone = new GPUKernel(command_queue, _k.get(Kernel.create_bone), 5);
        create_bone.new_arg(0, Sizeof.cl_mem, Memory.bone_instances.gpu.pointer());
        create_bone.new_arg(1, Sizeof.cl_mem, Memory.bone_index.gpu.pointer());
        create_bone.def_arg(2, Sizeof.cl_int);
        create_bone.def_arg(3, Sizeof.cl_float16);
        create_bone.def_arg(4, Sizeof.cl_int);
        Kernel.create_bone.set_kernel(create_bone);

        var create_hull = new GPUKernel(command_queue, _k.get(Kernel.create_hull), 9);
        create_hull.new_arg(0, Sizeof.cl_mem, Memory.hulls.gpu.pointer());
        create_hull.new_arg(1, Sizeof.cl_mem, Memory.hull_rotation.gpu.pointer());
        create_hull.new_arg(2, Sizeof.cl_mem, Memory.hull_element_table.gpu.pointer());
        create_hull.new_arg(3, Sizeof.cl_mem, Memory.hull_flags.gpu.pointer());
        create_hull.def_arg(4, Sizeof.cl_int);
        create_hull.def_arg(5, Sizeof.cl_float4);
        create_hull.def_arg(6, Sizeof.cl_float2);
        create_hull.def_arg(7, Sizeof.cl_int4);
        create_hull.def_arg(8, Sizeof.cl_int2);
        Kernel.create_hull.set_kernel(create_hull);

        var update_accel = new GPUKernel(command_queue, _k.get(Kernel.update_accel), 3);
        update_accel.new_arg(0, Sizeof.cl_mem, Memory.armature_accel.gpu.pointer());
        update_accel.def_arg(1, Sizeof.cl_int);
        update_accel.def_arg(2, Sizeof.cl_float2);
        Kernel.update_accel.set_kernel(update_accel);

        var read_position = new GPUKernel(command_queue, _k.get(Kernel.read_position), 3);
        read_position.new_arg(0, Sizeof.cl_mem, Memory.armatures.gpu.pointer());
        read_position.def_arg(1, Sizeof.cl_float2);
        read_position.def_arg(2, Sizeof.cl_int);
        Kernel.read_position.set_kernel(read_position);

        var animate_hulls = new GPUKernel(command_queue, _k.get(Kernel.animate_hulls), 8);
        animate_hulls.new_arg(0, Sizeof.cl_mem, Memory.points.gpu.pointer());
        animate_hulls.new_arg(1, Sizeof.cl_mem, Memory.hulls.gpu.pointer());
        animate_hulls.new_arg(2, Sizeof.cl_mem, Memory.hull_flags.gpu.pointer());
        animate_hulls.new_arg(3, Sizeof.cl_mem, Memory.vertex_table.gpu.pointer());
        animate_hulls.new_arg(4, Sizeof.cl_mem, Memory.armatures.gpu.pointer());
        animate_hulls.new_arg(5, Sizeof.cl_mem, Memory.armature_flags.gpu.pointer());
        animate_hulls.new_arg(6, Sizeof.cl_mem, Memory.vertex_references.gpu.pointer());
        animate_hulls.new_arg(7, Sizeof.cl_mem, Memory.bone_instances.gpu.pointer());
        Kernel.animate_hulls.set_kernel(animate_hulls);

        var integrate = new GPUKernel(command_queue, _k.get(Kernel.integrate), 12);
        integrate.new_arg(0, Sizeof.cl_mem, Memory.hulls.gpu.pointer());
        integrate.new_arg(1, Sizeof.cl_mem, Memory.armatures.gpu.pointer());
        integrate.new_arg(2, Sizeof.cl_mem, Memory.armature_flags.gpu.pointer());
        integrate.new_arg(3, Sizeof.cl_mem, Memory.hull_element_table.gpu.pointer());
        integrate.new_arg(4, Sizeof.cl_mem, Memory.armature_accel.gpu.pointer());
        integrate.new_arg(5, Sizeof.cl_mem, Memory.hull_rotation.gpu.pointer());
        integrate.new_arg(6, Sizeof.cl_mem, Memory.points.gpu.pointer());
        integrate.new_arg(7, Sizeof.cl_mem, Memory.aabb.gpu.pointer());
        integrate.new_arg(8, Sizeof.cl_mem, Memory.aabb_index.gpu.pointer());
        integrate.new_arg(9, Sizeof.cl_mem, Memory.aabb_key_table.gpu.pointer());
        integrate.new_arg(10, Sizeof.cl_mem, Memory.hull_flags.gpu.pointer());
        integrate.def_arg(11, Sizeof.cl_mem);
        Kernel.integrate.set_kernel(integrate);

        var generate_keys = new GPUKernel(command_queue, _k.get(Kernel.generate_keys), 7);
        generate_keys.new_arg(0, Sizeof.cl_mem, Memory.aabb_index.gpu.pointer());
        generate_keys.new_arg(1, Sizeof.cl_mem, Memory.aabb_key_table.gpu.pointer());
        generate_keys.def_arg(2, Sizeof.cl_mem);
        generate_keys.def_arg(3, Sizeof.cl_mem);
        generate_keys.def_arg(4, Sizeof.cl_int);
        generate_keys.def_arg(5, Sizeof.cl_int);
        generate_keys.def_arg(6, Sizeof.cl_int);
        Kernel.generate_keys.set_kernel(generate_keys);

        var build_key_map = new GPUKernel(command_queue, _k.get(Kernel.build_key_map), 7);
        build_key_map.new_arg(0, Sizeof.cl_mem, Memory.aabb_index.gpu.pointer());
        build_key_map.new_arg(1, Sizeof.cl_mem, Memory.aabb_key_table.gpu.pointer());
        build_key_map.def_arg(2, Sizeof.cl_mem);
        build_key_map.def_arg(3, Sizeof.cl_mem);
        build_key_map.def_arg(4, Sizeof.cl_mem);
        build_key_map.def_arg(5, Sizeof.cl_int);
        build_key_map.def_arg(6, Sizeof.cl_int);
        Kernel.build_key_map.set_kernel(build_key_map);

        var locate_in_bounds = new GPUKernel(command_queue, _k.get(Kernel.locate_in_bounds), 3);
        locate_in_bounds.new_arg(0, Sizeof.cl_mem, Memory.aabb_key_table.gpu.pointer());
        locate_in_bounds.def_arg(1, Sizeof.cl_mem);
        locate_in_bounds.def_arg(2, Sizeof.cl_mem);
        Kernel.locate_in_bounds.set_kernel(locate_in_bounds);

        var count_candidates = new GPUKernel(command_queue, _k.get(Kernel.count_candidates), 7);
        count_candidates.new_arg(0, Sizeof.cl_mem, Memory.aabb_key_table.gpu.pointer());
        count_candidates.def_arg(1, Sizeof.cl_mem);
        count_candidates.def_arg(2, Sizeof.cl_mem);
        count_candidates.def_arg(3, Sizeof.cl_mem);
        count_candidates.def_arg(4, Sizeof.cl_mem);
        count_candidates.def_arg(5, Sizeof.cl_int);
        count_candidates.def_arg(6, Sizeof.cl_int);
        Kernel.count_candidates.set_kernel(count_candidates);

        var aabb_collide = new GPUKernel(command_queue, _k.get(Kernel.aabb_collide), 14);
        aabb_collide.new_arg(0, Sizeof.cl_mem, Memory.aabb.gpu.pointer());
        aabb_collide.new_arg(1, Sizeof.cl_mem, Memory.aabb_key_table.gpu.pointer());
        aabb_collide.new_arg(2, Sizeof.cl_mem, Memory.hull_flags.gpu.pointer());
        aabb_collide.def_arg(3, Sizeof.cl_mem);
        aabb_collide.def_arg(4, Sizeof.cl_mem);
        aabb_collide.def_arg(5, Sizeof.cl_mem);
        aabb_collide.def_arg(6, Sizeof.cl_mem);
        aabb_collide.def_arg(7, Sizeof.cl_mem);
        aabb_collide.def_arg(8, Sizeof.cl_mem);
        aabb_collide.def_arg(9, Sizeof.cl_mem);
        aabb_collide.def_arg(10, Sizeof.cl_mem);
        aabb_collide.def_arg(11, Sizeof.cl_mem);
        aabb_collide.def_arg(12, Sizeof.cl_int);
        aabb_collide.def_arg(13, Sizeof.cl_int);
        Kernel.aabb_collide.set_kernel(aabb_collide);

        var finalize_candidates = new GPUKernel(command_queue, _k.get(Kernel.finalize_candidates), 6);
        finalize_candidates.def_arg(0, Sizeof.cl_mem);
        finalize_candidates.def_arg(1, Sizeof.cl_mem);
        finalize_candidates.def_arg(2, Sizeof.cl_mem);
        finalize_candidates.def_arg(3, Sizeof.cl_mem);
        finalize_candidates.def_arg(4, Sizeof.cl_mem);
        finalize_candidates.def_arg(5, Sizeof.cl_mem);
        Kernel.finalize_candidates.set_kernel(finalize_candidates);




        var resolve_constraints = new GPUKernel(command_queue, _k.get(Kernel.resolve_constraints), 5);
        resolve_constraints.new_arg(0, Sizeof.cl_mem, Memory.hull_element_table.gpu.pointer());
        resolve_constraints.new_arg(1, Sizeof.cl_mem, Memory.aabb_key_table.gpu.pointer());
        resolve_constraints.new_arg(2, Sizeof.cl_mem, Memory.points.gpu.pointer());
        resolve_constraints.new_arg(3, Sizeof.cl_mem, Memory.edges.gpu.pointer());
        resolve_constraints.def_arg(4, Sizeof.cl_int);
        Kernel.resolve_constraints.set_kernel(resolve_constraints);

        var scan_int_single_block = new GPUKernel(command_queue, _k.get(Kernel.scan_int_single_block), 3);
        scan_int_single_block.def_arg(0, Sizeof.cl_mem);
        scan_int_single_block.def_arg(1, -1);
        scan_int_single_block.def_arg(2, Sizeof.cl_int);
        Kernel.scan_int_single_block.set_kernel(scan_int_single_block);

        var scan_int_multi_block = new GPUKernel(command_queue, _k.get(Kernel.scan_int_multi_block), 4);
        scan_int_multi_block.def_arg(0, Sizeof.cl_mem);
        scan_int_multi_block.def_arg(1, -1);
        scan_int_multi_block.def_arg(2, Sizeof.cl_mem);
        scan_int_multi_block.def_arg(3, Sizeof.cl_int);
        Kernel.scan_int_multi_block.set_kernel(scan_int_multi_block);

        var complete_int_multi_block = new GPUKernel(command_queue, _k.get(Kernel.complete_int_multi_block), 4);
        complete_int_multi_block.def_arg(0, Sizeof.cl_mem);
        complete_int_multi_block.def_arg(1, -1);
        complete_int_multi_block.def_arg(2, Sizeof.cl_mem);
        complete_int_multi_block.def_arg(3, Sizeof.cl_int);
        Kernel.complete_int_multi_block.set_kernel(complete_int_multi_block);

        var scan_int_single_block_out = new GPUKernel(command_queue, _k.get(Kernel.scan_int_single_block_out), 4);
        scan_int_single_block_out.def_arg(0, Sizeof.cl_mem);
        scan_int_single_block_out.def_arg(1, Sizeof.cl_mem);
        scan_int_single_block_out.def_arg(2, -1);
        scan_int_single_block_out.def_arg(3, Sizeof.cl_int);
        Kernel.scan_int_single_block_out.set_kernel(scan_int_single_block_out);

        var scan_int_multi_block_out = new GPUKernel(command_queue, _k.get(Kernel.scan_int_multi_block_out), 5);
        scan_int_multi_block_out.def_arg(0, Sizeof.cl_mem);
        scan_int_multi_block_out.def_arg(1, Sizeof.cl_mem);
        scan_int_multi_block_out.def_arg(2, -1);
        scan_int_multi_block_out.def_arg(3, Sizeof.cl_mem);
        scan_int_multi_block_out.def_arg(4, Sizeof.cl_int);
        Kernel.scan_int_multi_block_out.set_kernel(scan_int_multi_block_out);

        var complete_int_multi_block_out = new GPUKernel(command_queue, _k.get(Kernel.complete_int_multi_block_out), 4);
        complete_int_multi_block_out.def_arg(0, Sizeof.cl_mem);
        complete_int_multi_block_out.def_arg(1, -1);
        complete_int_multi_block_out.def_arg(2, Sizeof.cl_mem);
        complete_int_multi_block_out.def_arg(3, Sizeof.cl_int);
        Kernel.complete_int_multi_block_out.set_kernel(complete_int_multi_block_out);

        var scan_candidates_single_block_out = new GPUKernel(command_queue, _k.get(Kernel.scan_candidates_single_block_out), 5);
        scan_candidates_single_block_out.def_arg(0, Sizeof.cl_mem);
        scan_candidates_single_block_out.def_arg(1, Sizeof.cl_mem);
        scan_candidates_single_block_out.def_arg(2, Sizeof.cl_mem);
        scan_candidates_single_block_out.def_arg(3, -1);
        scan_candidates_single_block_out.def_arg(4, Sizeof.cl_int);
        Kernel.scan_candidates_single_block_out.set_kernel(scan_candidates_single_block_out);

        var scan_candidates_multi_block_out = new GPUKernel(command_queue, _k.get(Kernel.scan_candidates_multi_block_out), 5);
        scan_candidates_multi_block_out.def_arg(0, Sizeof.cl_mem);
        scan_candidates_multi_block_out.def_arg(1, Sizeof.cl_mem);
        scan_candidates_multi_block_out.def_arg(2, -1);
        scan_candidates_multi_block_out.def_arg(3, Sizeof.cl_mem);
        scan_candidates_multi_block_out.def_arg(4, Sizeof.cl_int);
        Kernel.scan_candidates_multi_block_out.set_kernel(scan_candidates_multi_block_out);

        var complete_candidates_multi_block_out = new GPUKernel(command_queue, _k.get(Kernel.complete_candidates_multi_block_out), 6);
        complete_candidates_multi_block_out.def_arg(0, Sizeof.cl_mem);
        complete_candidates_multi_block_out.def_arg(1, Sizeof.cl_mem);
        complete_candidates_multi_block_out.def_arg(2, Sizeof.cl_mem);
        complete_candidates_multi_block_out.def_arg(3, -1);
        complete_candidates_multi_block_out.def_arg(4, Sizeof.cl_mem);
        complete_candidates_multi_block_out.def_arg(5, Sizeof.cl_int);
        Kernel.complete_candidates_multi_block_out.set_kernel(complete_candidates_multi_block_out);

        var scan_bounds_single_block = new GPUKernel(command_queue, _k.get(Kernel.scan_bounds_single_block), 4);
        scan_bounds_single_block.def_arg(0, Sizeof.cl_mem);
        scan_bounds_single_block.def_arg(1, Sizeof.cl_mem);
        scan_bounds_single_block.def_arg(2, -1);
        scan_bounds_single_block.def_arg(3, Sizeof.cl_int);
        Kernel.scan_bounds_single_block.set_kernel(scan_bounds_single_block);

        var scan_bounds_multi_block = new GPUKernel(command_queue, _k.get(Kernel.scan_bounds_multi_block), 4);
        scan_bounds_multi_block.def_arg(0, Sizeof.cl_mem);
        scan_bounds_multi_block.def_arg(1, -1);
        scan_bounds_multi_block.def_arg(2, Sizeof.cl_mem);
        scan_bounds_multi_block.def_arg(3, Sizeof.cl_int);
        Kernel.scan_bounds_multi_block.set_kernel(scan_bounds_multi_block);

        var complete_bounds_multi_block = new GPUKernel(command_queue, _k.get(Kernel.complete_bounds_multi_block), 5);
        complete_bounds_multi_block.def_arg(0, Sizeof.cl_mem);
        complete_bounds_multi_block.def_arg(1, Sizeof.cl_mem);
        complete_bounds_multi_block.def_arg(2, -1);
        complete_bounds_multi_block.def_arg(3, Sizeof.cl_mem);
        complete_bounds_multi_block.def_arg(4, Sizeof.cl_int);
        Kernel.complete_bounds_multi_block.set_kernel(complete_bounds_multi_block);
    }

    //#endregion

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
     * @param vbo_id        id of the shared GL buffer object
     * @param bounds_offset offset into the bounds array to start the transfer
     * @param batch_size    number of bounds objects to transfer in this batch
     */
    public static void GL_bounds(int vbo_id, int bounds_offset, int batch_size)
    {
        var gpu_kernel = Kernel.prepare_bounds.gpu;

        var vbo_mem = shared_mem.get(vbo_id);

        gpu_kernel.share_mem(vbo_mem);
        gpu_kernel.set_arg(1, Pointer.to(vbo_mem));
        gpu_kernel.set_arg(2, Pointer.to(arg_long(bounds_offset)));
        gpu_kernel.call(arg_long(batch_size));
    }

    /**
     * Transfers a subset of all bones from CL memory into GL memory, converting the bones
     * into a vertex structure that can be rendered as a point decal.
     *
     * @param vbo_id      id of the shared GL buffer object
     * @param bone_offset offset into the bones array to start the transfer
     * @param batch_size  number of bone objects to transfer in this batch
     */
    public static void GL_bones(int vbo_id, int bone_offset, int batch_size)
    {
        var gpu_kernel = Kernel.prepare_bones.gpu;

        var vbo_mem = shared_mem.get(vbo_id);

        gpu_kernel.share_mem(vbo_mem);
        gpu_kernel.set_arg(6, Pointer.to(vbo_mem));
        gpu_kernel.set_arg(7, Pointer.to(arg_long(bone_offset)));
        gpu_kernel.call(arg_long(batch_size));
    }

    /**
     * Transfers a subset of all edges from CL memory into GL memory, converting the edges
     * into a vertex structure that can be rendered as a line.
     *
     * @param vbo_id      id of the shared GL buffer object
     * @param edge_offset offset into the edges array to start the transfer
     * @param batch_size  number of edge objects to transfer in this batch
     */
    public static void GL_edges(int vbo_id, int edge_offset, int batch_size)
    {
        var gpu_kernel = Kernel.prepare_edges.gpu;

        var vbo_mem = shared_mem.get(vbo_id);

        gpu_kernel.share_mem(vbo_mem);
        gpu_kernel.set_arg(2, Pointer.to(vbo_mem));
        gpu_kernel.set_arg(3, Pointer.to(arg_long(edge_offset)));
        gpu_kernel.call(arg_long(batch_size));
    }

    /**
     * Transfers a subset of all hull transforms from CL memory into GL memory. Hulls
     * are generally not rendered directly using this data, but it is used to transform
     * model reference data from memory into the position of the mesh that the hull
     * represents within the simulation.
     *
     * @param index_buffer_id id of the shared GL buffer object
     * @param transforms_id   id of the shared GL buffer object
     * @param batch_size      number of hull objects to transfer in this batch
     */
    public static void GL_transforms(int index_buffer_id, int transforms_id, int batch_size)
    {
        var gpu_kernel = Kernel.prepare_transforms.gpu;

        var vbo_index_buffer = shared_mem.get(index_buffer_id);
        var vbo_transforms = shared_mem.get(transforms_id);

        gpu_kernel.share_mem(vbo_index_buffer);
        gpu_kernel.share_mem(vbo_transforms);
        gpu_kernel.set_arg(2, Pointer.to(vbo_index_buffer));
        gpu_kernel.set_arg(3, Pointer.to(vbo_transforms));
        gpu_kernel.call(arg_long(batch_size));
    }

    //#endregion

    //#region CPU Create/Read/Update/Delete Functions

    public static void create_point(int point_index,
                                    float pos_x,
                                    float pos_y,
                                    float prv_x,
                                    float prv_y,
                                    int vert_index,
                                    int bone_index)
    {
        var gpu_kernel = Kernel.create_point.gpu;
        gpu_kernel.set_arg(2, Pointer.to(arg_int(point_index)));
        gpu_kernel.set_arg(3, Pointer.to(arg_float4(pos_x, pos_y, prv_x, prv_y)));
        gpu_kernel.set_arg(4, Pointer.to(arg_int2(vert_index, bone_index)));
        gpu_kernel.call(global_single_size);
    }

    public static void create_edge(int edge_index, float p1, float p2, float l, int flags)
    {
        var gpu_kernel = Kernel.create_edge.gpu;
        gpu_kernel.set_arg(1, Pointer.to(arg_int(edge_index)));
        gpu_kernel.set_arg(2, Pointer.to(arg_float4(p1, p2, l, flags)));
        gpu_kernel.call(global_single_size);
    }

    public static void create_armature(int armature_index, float x, float y, int flags)
    {
        var gpu_kernel = Kernel.create_armature.gpu;
        gpu_kernel.set_arg(2, Pointer.to(arg_int(armature_index)));
        gpu_kernel.set_arg(3, Pointer.to(arg_float4(x, y, x, y)));
        gpu_kernel.set_arg(4, Pointer.to(arg_int(flags)));
        gpu_kernel.call(global_single_size);
    }

    public static void create_vertex_reference(int vert_ref_index, float x, float y)
    {
        var gpu_kernel = Kernel.create_vertex_reference.gpu;
        gpu_kernel.set_arg(1, Pointer.to(arg_int(vert_ref_index)));
        gpu_kernel.set_arg(2, Pointer.to(arg_float2(x, y)));
        gpu_kernel.call(global_single_size);
    }

    public static void create_bone_reference(int bone_ref_index, float[] matrix)
    {
        var gpu_kernel = Kernel.create_bone_reference.gpu;
        gpu_kernel.set_arg(1, Pointer.to(arg_int(bone_ref_index)));
        gpu_kernel.set_arg(2, Pointer.to(matrix));
        gpu_kernel.call(global_single_size);
    }

    public static void create_bone(int bone_index, int bone_ref_index, float[] matrix)
    {
        var gpu_kernel = Kernel.create_bone.gpu;
        gpu_kernel.set_arg(2, Pointer.to(arg_int(bone_index)));
        gpu_kernel.set_arg(3, Pointer.to(matrix));
        gpu_kernel.set_arg(4, Pointer.to(arg_int(bone_ref_index)));
        gpu_kernel.call(global_single_size);
    }

    public static void create_hull(int hull_index, float[] hull, float[] rotation, int[] table, int[] flags)
    {
        var gpu_kernel = Kernel.create_hull.gpu;
        gpu_kernel.set_arg(4, Pointer.to(arg_int(hull_index)));
        gpu_kernel.set_arg(5, Pointer.to(hull));
        gpu_kernel.set_arg(6, Pointer.to(rotation));
        gpu_kernel.set_arg(7, Pointer.to(table));
        gpu_kernel.set_arg(8, Pointer.to(flags));
        gpu_kernel.call(global_single_size);
    }

    public static void update_accel(int armature_index, float acc_x, float acc_y)
    {
        var gpu_kernel = Kernel.update_accel.gpu;
        gpu_kernel.set_arg(1, Pointer.to(arg_int(armature_index)));
        gpu_kernel.set_arg(2, Pointer.to(arg_float2(acc_x, acc_y)));
        gpu_kernel.call(global_single_size);
    }

    // todo: implement armature rotations and update this
    public static void rotate_hull(int hull_index, float angle)
    {
        var pnt_index = Pointer.to(arg_int(hull_index));
        var pnt_angle = Pointer.to(arg_float(angle));

        clSetKernelArg(_k.get(Kernel.rotate_hull), 0, Sizeof.cl_mem, Memory.hulls.gpu.pointer());
        clSetKernelArg(_k.get(Kernel.rotate_hull), 1, Sizeof.cl_mem, Memory.hull_element_table.gpu.pointer());
        clSetKernelArg(_k.get(Kernel.rotate_hull), 2, Sizeof.cl_mem, Memory.points.gpu.pointer());
        clSetKernelArg(_k.get(Kernel.rotate_hull), 3, Sizeof.cl_int, pnt_index);
        clSetKernelArg(_k.get(Kernel.rotate_hull), 4, Sizeof.cl_float, pnt_angle);

        k_call(command_queue, _k.get(Kernel.rotate_hull), global_single_size);
    }

    public static float[] read_position(int armature_index)
    {
        if (physics_buffer == null)
        {
            return null;
        }

        var gpu_kernel = Kernel.read_position.gpu;

        int[] index = arg_int(armature_index);

        var result_data = cl_new_buffer(Sizeof.cl_float2);
        cl_zero_buffer(result_data, Sizeof.cl_float2);

        gpu_kernel.set_arg(1, Pointer.to(result_data));
        gpu_kernel.set_arg(2, Pointer.to(index));
        gpu_kernel.call(global_single_size);

        float[] result = arg_float2(0, 0);
        cl_read_buffer(result_data, Sizeof.cl_float2, Pointer.to(result));
        clReleaseMemObject(result_data);

        return result;
    }

    //#endregion

    //#region Physics Simulation

    public static void animate_hulls()
    {
        Kernel.animate_hulls.gpu.call(arg_long(Main.Memory.point_count()));
    }

    public static void integrate(float delta_time, UniformGrid uniform_grid)
    {
        var gpu_kernel = Kernel.integrate.gpu;

        float[] args =
            {
                delta_time,
                uniform_grid.getX_spacing(),
                uniform_grid.getY_spacing(),
                uniform_grid.getX_origin(),
                uniform_grid.getY_origin(),
                uniform_grid.getWidth(),
                uniform_grid.getHeight(),
                (float) uniform_grid.getX_subdivisions(),
                (float) uniform_grid.getY_subdivisions(),
                physics_buffer.get_gravity_x(),
                physics_buffer.get_gravity_y(),
                physics_buffer.get_friction()
            };

        var srcArgs = Pointer.to(args);

        long size = Sizeof.cl_float * args.length;
        var argMem = cl_new_read_only_buffer(size, srcArgs);

        gpu_kernel.set_arg(11, Pointer.to(argMem));
        gpu_kernel.call(arg_long(Main.Memory.hull_count()));

        clReleaseMemObject(argMem);
    }

    public static void calculate_bank_offsets(UniformGrid uniform_grid)
    {
        int bank_size = scan_key_bounds(Memory.aabb_key_table.gpu.memory(), Main.Memory.hull_count());
        uniform_grid.resizeBank(bank_size);
    }

    public static void generate_keys(UniformGrid uniform_grid)
    {
        if (uniform_grid.getKey_bank_size() < 1)
        {
            return;
        }

        var gpu_kernel = Kernel.generate_keys.gpu;

        long bank_buf_size = (long) Sizeof.cl_int * uniform_grid.getKey_bank_size();
        long counts_buf_size = (long) Sizeof.cl_int * uniform_grid.getDirectoryLength();

        var bank_data = cl_new_buffer(bank_buf_size);
        var counts_data = cl_new_buffer(counts_buf_size);
        cl_zero_buffer(counts_data, counts_buf_size);

        physics_buffer.key_counts = new GPUMemory(counts_data);
        physics_buffer.key_bank = new GPUMemory(bank_data);

        gpu_kernel.set_arg(2, Pointer.to(bank_data));
        gpu_kernel.set_arg(3, Pointer.to(counts_data));
        gpu_kernel.set_arg(4, Pointer.to(arg_int(uniform_grid.getX_subdivisions())));
        gpu_kernel.set_arg(5, Pointer.to(arg_int(uniform_grid.getKey_bank_size())));
        gpu_kernel.set_arg(6, Pointer.to(arg_int(uniform_grid.getDirectoryLength())));
        gpu_kernel.call(arg_long(Main.Memory.hull_count()));
    }

    public static void calculate_map_offsets(UniformGrid uniform_grid)
    {
        int n = uniform_grid.getDirectoryLength();
        long data_buf_size = (long) Sizeof.cl_int * n;
        var o_data = cl_new_buffer(data_buf_size);
        physics_buffer.key_offsets = new GPUMemory(o_data);
        scan_int_out(physics_buffer.key_counts.memory(), o_data, n);
    }

    public static void build_key_map(UniformGrid uniform_grid)
    {
        var gpu_kernel = Kernel.build_key_map.gpu;

        long map_buf_size = (long) Sizeof.cl_int * uniform_grid.getKey_map_size();
        long counts_buf_size = (long) Sizeof.cl_int * uniform_grid.getDirectoryLength();

        var map_data = cl_new_buffer(map_buf_size);
        var counts_data = cl_new_buffer(counts_buf_size);

        // the counts buffer needs to start off filled with all zeroes
        cl_zero_buffer(counts_data, counts_buf_size);

        physics_buffer.key_map = new GPUMemory(map_data);

        gpu_kernel.set_arg(2, Pointer.to(map_data));
        gpu_kernel.set_arg(3, physics_buffer.key_offsets.pointer());
        gpu_kernel.set_arg(4, Pointer.to(counts_data));
        gpu_kernel.set_arg(5, Pointer.to(arg_int(uniform_grid.getX_subdivisions())));
        gpu_kernel.set_arg(6, Pointer.to(arg_int(uniform_grid.getDirectoryLength())));
        gpu_kernel.call(arg_long(Main.Memory.hull_count()));

        clReleaseMemObject(counts_data);
    }

    public static void locate_in_bounds(UniformGrid uniform_grid)
    {
        var gpu_kernel = Kernel.locate_in_bounds.gpu;

        int hull_count = Main.Memory.hull_count();

        int x_subdivisions = uniform_grid.getX_subdivisions();
        physics_buffer.x_sub_divisions = Pointer.to(arg_int(x_subdivisions));
        physics_buffer.key_count_length = Pointer.to(arg_int(uniform_grid.getDirectoryLength()));

        long inbound_buf_size = (long) Sizeof.cl_int * hull_count;
        var inbound_data = cl_new_buffer(inbound_buf_size);

        physics_buffer.in_bounds = new GPUMemory(inbound_data);

        int[] size = arg_int(0);
        var dst_size = Pointer.to(size);
        var size_data = cl_new_int_arg_buffer(dst_size);

        gpu_kernel.set_arg(1, physics_buffer.in_bounds.pointer());
        gpu_kernel.set_arg(2, Pointer.to(size_data));
        gpu_kernel.call(arg_long(hull_count));

        cl_read_buffer(size_data, Sizeof.cl_int, dst_size);

        clReleaseMemObject(size_data);

        physics_buffer.set_candidate_buffer_count(size[0]);
    }

    public static void count_candidates()
    {
        var gpu_kernel = Kernel.count_candidates.gpu;

        long candidate_buf_size = (long) Sizeof.cl_int2 * physics_buffer.get_candidate_buffer_count();
        var candidate_data = cl_new_buffer(candidate_buf_size);
        physics_buffer.candidate_counts = new GPUMemory(candidate_data);

        gpu_kernel.set_arg(1, physics_buffer.in_bounds.pointer());
        gpu_kernel.set_arg(2, physics_buffer.key_bank.pointer());
        gpu_kernel.set_arg(3, physics_buffer.key_counts.pointer());
        gpu_kernel.set_arg(4, physics_buffer.candidate_counts.pointer());
        gpu_kernel.set_arg(5, physics_buffer.x_sub_divisions);
        gpu_kernel.set_arg(6, physics_buffer.key_count_length);
        gpu_kernel.call(arg_long(physics_buffer.get_candidate_buffer_count()));
    }

    public static void count_matches()
    {
        int buffer_count = physics_buffer.get_candidate_buffer_count();
        long offset_buf_size = (long) Sizeof.cl_int * buffer_count;
        var offset_data = cl_new_buffer(offset_buf_size);
        physics_buffer.candidate_offsets = new GPUMemory(offset_data);
        int match_count = scan_key_candidates(physics_buffer.candidate_counts.memory(), offset_data, buffer_count);
        physics_buffer.set_candidate_match_count(match_count);
    }

    public static void aabb_collide()
    {
        var gpu_kernel = Kernel.aabb_collide.gpu;

        long matches_buf_size = (long) Sizeof.cl_int * physics_buffer.get_candidate_match_count();
        var matches_data = cl_new_buffer(matches_buf_size);
        physics_buffer.matches = new GPUMemory(matches_data);

        long used_buf_size = (long) Sizeof.cl_int * physics_buffer.get_candidate_buffer_count();
        var used_data = cl_new_buffer(used_buf_size);
        physics_buffer.matches_used = new GPUMemory(used_data);

        // this buffer will contain the total number of candidates that were found
        int[] count = arg_int(0);
        var dst_count = Pointer.to(count);
        var count_data = cl_new_int_arg_buffer(dst_count);

        gpu_kernel.set_arg(3, physics_buffer.candidate_counts.pointer());
        gpu_kernel.set_arg(4, physics_buffer.candidate_offsets.pointer());
        gpu_kernel.set_arg(5, physics_buffer.key_map.pointer());
        gpu_kernel.set_arg(6, physics_buffer.key_bank.pointer());
        gpu_kernel.set_arg(7, physics_buffer.key_counts.pointer());
        gpu_kernel.set_arg(8, physics_buffer.key_offsets.pointer());
        gpu_kernel.set_arg(9, physics_buffer.matches.pointer());
        gpu_kernel.set_arg(10, physics_buffer.matches_used.pointer());
        gpu_kernel.set_arg(11, Pointer.to(count_data));
        gpu_kernel.set_arg(12, physics_buffer.x_sub_divisions);
        gpu_kernel.set_arg(13, physics_buffer.key_count_length);
        gpu_kernel.call(arg_long(physics_buffer.get_candidate_buffer_count()));

        cl_read_buffer(count_data, Sizeof.cl_int, dst_count);

        clReleaseMemObject(count_data);

        physics_buffer.set_candidate_count(count[0]);
    }

    public static void finalize_candidates()
    {
        if (physics_buffer.get_candidate_count() > 0)
        {
            var gpu_kernel = Kernel.finalize_candidates.gpu;

            // create an empty buffer that the kernel will use to store finalized candidates
            long final_buf_size = (long) Sizeof.cl_int2 * physics_buffer.get_candidate_count();
            var finals_data = cl_new_buffer(final_buf_size);

            // the kernel will use this value as an internal atomic counter, always initialize to zero
            int[] counter = new int[]{0};
            var dst_counter = Pointer.to(counter);
            var counter_data = cl_new_int_arg_buffer(dst_counter);

            physics_buffer.set_final_size(final_buf_size);
            physics_buffer.candidates = new GPUMemory(finals_data);

            gpu_kernel.set_arg(0, physics_buffer.candidate_counts.pointer());
            gpu_kernel.set_arg(1, physics_buffer.candidate_offsets.pointer());
            gpu_kernel.set_arg(2, physics_buffer.matches.pointer());
            gpu_kernel.set_arg(3, physics_buffer.matches_used.pointer());
            gpu_kernel.set_arg(4, Pointer.to(counter_data));
            gpu_kernel.set_arg(5, physics_buffer.candidates.pointer());
            gpu_kernel.call(arg_long(physics_buffer.get_candidate_buffer_count()));

            clReleaseMemObject(counter_data);
        }
    }

    public static void sat_collide()
    {
        var gpu_kernel = Kernel.sat_collide.gpu;

        int candidates_size = (int) physics_buffer.get_final_size() / Sizeof.cl_int;

        // candidates are pairs of integer indices, so the global size is half the count
        long[] global_work_size = new long[]{candidates_size / 2};

        // atomic counter
        int[] size = arg_int(0);
        var dst_size = Pointer.to(size);
        var size_data = cl_new_int_arg_buffer(dst_size);

        long max_point_count = physics_buffer.get_final_size()
            * 2  // there are two bodies per collision pair
            * 2; // assume worst case is 2 points per body

        // sizes for the reaction buffers
        long reaction_buf_size = (long) Sizeof.cl_float2 * max_point_count;
        long index_buf_size = (long) Sizeof.cl_int * max_point_count;

        var reaction_data = cl_new_buffer(reaction_buf_size);
        var index_data = cl_new_buffer(index_buf_size);

        physics_buffer.reactions = new GPUMemory(reaction_data);
        physics_buffer.reaction_index = new GPUMemory(index_data);

        gpu_kernel.set_arg(0, physics_buffer.candidates.pointer());
        gpu_kernel.set_arg(7, physics_buffer.reactions.pointer());
        gpu_kernel.set_arg(8, physics_buffer.reaction_index.pointer());
        gpu_kernel.set_arg(10, Pointer.to(size_data));

        gpu_kernel.call(global_work_size);

        cl_read_buffer(size_data, Sizeof.cl_int, dst_size);

        clReleaseMemObject(size_data);

        physics_buffer.set_reaction_count(size[0]);

//        if (physics_buffer.get_reaction_count() > 0)
//        {
//            System.out.println("DEBUG: reaction count=" + physics_buffer.get_reaction_count());
//        }
    }

    public static void scan_reactions()
    {
        scan_int_out(Memory.point_reactions.gpu.memory(), Memory.point_offsets.gpu.memory(), Main.Memory.point_count());
        Memory.point_reactions.clear();
    }

    public static void sort_reactions()
    {
        var gpu_kernel = Kernel.sort_reactions.gpu;
        gpu_kernel.set_arg(0, physics_buffer.reactions.pointer());
        gpu_kernel.set_arg(1, physics_buffer.reaction_index.pointer());
        gpu_kernel.call(arg_long(physics_buffer.get_reaction_count()));
    }

    public static void apply_reactions()
    {
        var gpu_kernel = Kernel.apply_reactions.gpu;
        gpu_kernel.set_arg(0, physics_buffer.reactions.pointer());
        gpu_kernel.call(arg_long(Main.Memory.point_count()));
    }

    public static void resolve_constraints(int edge_steps)
    {
        var gpu_kernel = Kernel.resolve_constraints.gpu;

        boolean last_step;
        for (int i = 0; i < edge_steps; i++)
        {
            last_step = i == edge_steps - 1;
            int n = last_step
                ? 1
                : 0;
            gpu_kernel.set_arg(4, Pointer.to(arg_int(n)));
            gpu_kernel.call(arg_long(Main.Memory.hull_count()));
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
            return scan_bounds_single_block(d_data2, n);
        }
        else
        {
            return scan_bounds_multi_block(d_data2, n, k);
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
        var gpu_kernel = Kernel.scan_int_single_block.gpu;

        long local_buffer_size = Sizeof.cl_int * max_scan_block_size;

        gpu_kernel.set_arg(0, Pointer.to(d_data));
        gpu_kernel.new_arg(1, local_buffer_size, null);
        gpu_kernel.set_arg(2, Pointer.to(arg_int(n)));
        gpu_kernel.call(local_work_default, local_work_default);
    }

    private static void scan_multi_block_int(cl_mem d_data, int n, int k)
    {
        var gpu_kernel_1 = Kernel.scan_int_multi_block.gpu;
        var gpu_kernel_2 = Kernel.complete_int_multi_block.gpu;

        long local_buffer_size = Sizeof.cl_int * max_scan_block_size;
        long gx = k * max_scan_block_size;
        long[] global_work_size = arg_long(gx);
        int part_size = k * 2;
        long part_buf_size = ((long) Sizeof.cl_int * ((long) part_size));

        var part_data = cl_new_buffer(part_buf_size);
        var src_data = Pointer.to(d_data);
        var src_part = Pointer.to(part_data);
        var src_n = Pointer.to(new int[]{n});

        gpu_kernel_1.set_arg(0, src_data);
        gpu_kernel_1.new_arg(1, local_buffer_size);
        gpu_kernel_1.set_arg(2, src_part);
        gpu_kernel_1.set_arg(3, src_n);
        gpu_kernel_1.call(global_work_size, local_work_default);

        scan_int(part_data, part_size);

        gpu_kernel_2.set_arg(0, src_data);
        gpu_kernel_2.new_arg(1, local_buffer_size);
        gpu_kernel_2.set_arg(2, src_part);
        gpu_kernel_2.set_arg(3, src_n);
        gpu_kernel_2.call(global_work_size, local_work_default);

        clReleaseMemObject(part_data);
    }

    private static void scan_single_block_int_out(cl_mem d_data, cl_mem o_data, int n)
    {
        var gpu_kernel = Kernel.scan_int_single_block_out.gpu;

        long local_buffer_size = Sizeof.cl_int * max_scan_block_size;

        gpu_kernel.set_arg(0, Pointer.to(d_data));
        gpu_kernel.set_arg(1, Pointer.to(o_data));
        gpu_kernel.new_arg(2, local_buffer_size);
        gpu_kernel.set_arg(3, Pointer.to(arg_int(n)));
        gpu_kernel.call(local_work_default, local_work_default);
    }

    private static void scan_multi_block_int_out(cl_mem input_data, cl_mem o_data, int n, int k)
    {
        var gpu_kernel_1 = Kernel.scan_int_multi_block_out.gpu;
        var gpu_kernel_2 = Kernel.complete_int_multi_block_out.gpu;

        long local_buffer_size = Sizeof.cl_int * max_scan_block_size;
        long gx = k * max_scan_block_size;
        long[] global_work_size = arg_long(gx);
        int part_size = k * 2;
        long part_buf_size = ((long) Sizeof.cl_int * ((long) part_size));
        var part_data = cl_new_buffer(part_buf_size);
        var src_data = Pointer.to(input_data);
        var src_part = Pointer.to(part_data);
        var dst_data = Pointer.to(o_data);
        var src_n = Pointer.to(new int[]{n});

        gpu_kernel_1.set_arg(0, src_data);
        gpu_kernel_1.set_arg(1, dst_data);
        gpu_kernel_1.new_arg(2, local_buffer_size);
        gpu_kernel_1.set_arg(3, src_part);
        gpu_kernel_1.set_arg(4, src_n);
        gpu_kernel_1.call(global_work_size, local_work_default);

        scan_int(part_data, part_size);

        gpu_kernel_2.set_arg(0, dst_data);
        gpu_kernel_2.new_arg(1, local_buffer_size);
        gpu_kernel_2.set_arg(2, src_part);
        gpu_kernel_2.set_arg(3, src_n);
        gpu_kernel_2.call(global_work_size, local_work_default);

        clReleaseMemObject(part_data);
    }

    private static int scan_single_block_candidates_out(cl_mem d_data, cl_mem o_data, int n)
    {
        var gpu_kernel = Kernel.scan_candidates_single_block_out.gpu;

        long local_buffer_size = Sizeof.cl_int * max_scan_block_size;
        var src_data = Pointer.to(d_data);
        var dst_data = Pointer.to(o_data);

        int[] sz = new int[]{0};
        var dst_size = Pointer.to(sz);
        var size_data = cl_new_buffer(Sizeof.cl_int);
        var src_size = Pointer.to(size_data);

        gpu_kernel.set_arg(0, src_data);
        gpu_kernel.set_arg(1, dst_data);
        gpu_kernel.set_arg(2, src_size);
        gpu_kernel.new_arg(3, local_buffer_size);
        gpu_kernel.set_arg(4, Pointer.to(arg_int(n)));
        gpu_kernel.call(local_work_default, local_work_default);

        cl_read_buffer(size_data, Sizeof.cl_int, dst_size);

        clReleaseMemObject(size_data);

        return sz[0];
    }

    private static int scan_multi_block_candidates_out(cl_mem d_data, cl_mem o_data, int n, int k)
    {
        var gpu_kernel_1 = Kernel.scan_candidates_multi_block_out.gpu;
        var gpu_kernel_2 = Kernel.complete_candidates_multi_block_out.gpu;

        long local_buffer_size = Sizeof.cl_int * max_scan_block_size;
        long gx = k * max_scan_block_size;
        long[] global_work_size = arg_long(gx);
        int part_size = k * 2;
        long part_buf_size = ((long) Sizeof.cl_int * ((long) part_size));
        var p_data = cl_new_buffer(part_buf_size);
        var src_data = Pointer.to(d_data);
        var src_part = Pointer.to(p_data);
        var dst_data = Pointer.to(o_data);
        var src_n = Pointer.to(new int[]{n});

        gpu_kernel_1.set_arg(0, src_data);
        gpu_kernel_1.set_arg(1, dst_data);
        gpu_kernel_1.new_arg(2, local_buffer_size);
        gpu_kernel_1.set_arg(3, src_part);
        gpu_kernel_1.set_arg(4, src_n);
        gpu_kernel_1.call(global_work_size, local_work_default);

        scan_int(p_data, part_size);

        int[] sz = new int[]{0};
        var dst_size = Pointer.to(sz);
        var size_data = cl_new_buffer(Sizeof.cl_int);
        var src_size = Pointer.to(size_data);

        gpu_kernel_2.set_arg(0, src_data);
        gpu_kernel_2.set_arg(1, dst_data);
        gpu_kernel_2.set_arg(2, src_size);
        gpu_kernel_2.new_arg(3, local_buffer_size);
        gpu_kernel_2.set_arg(4, src_part);
        gpu_kernel_2.set_arg(5, src_n);
        gpu_kernel_2.call(global_work_size, local_work_default);

        cl_read_buffer(size_data, Sizeof.cl_int, dst_size);

        clReleaseMemObject(p_data);
        clReleaseMemObject(size_data);

        return sz[0];
    }

    private static int scan_bounds_single_block(cl_mem input_data, int n)
    {
        var gpu_kernel = Kernel.scan_bounds_single_block.gpu;

        long local_buffer_size = Sizeof.cl_int * max_scan_block_size;

        int[] sz = new int[]{0};
        var size_data = cl_new_buffer(Sizeof.cl_int);

        gpu_kernel.set_arg(0, Pointer.to(input_data));
        gpu_kernel.set_arg(1, Pointer.to(size_data));
        gpu_kernel.new_arg(2, local_buffer_size);
        gpu_kernel.set_arg(3, Pointer.to(arg_int(n)));
        gpu_kernel.call(local_work_default, local_work_default);

        cl_read_buffer(size_data, Sizeof.cl_int, Pointer.to(sz));

        clReleaseMemObject(size_data);

        return sz[0];
    }

    private static int scan_bounds_multi_block(cl_mem input_data, int n, int k)
    {
        var gpu_kernel_1 = Kernel.scan_bounds_multi_block.gpu;
        var gpu_kernel_2 = Kernel.complete_bounds_multi_block.gpu;

        long local_buffer_size = Sizeof.cl_int * max_scan_block_size;
        long gx = k * max_scan_block_size;
        long[] global_work_size = arg_long(gx);
        int part_size = k * 2;
        long part_buf_size = ((long) Sizeof.cl_int * ((long) part_size));
        var p_data = cl_new_buffer(part_buf_size);
        var src_data = Pointer.to(input_data);
        var src_part = Pointer.to(p_data);
        var src_n = Pointer.to(new int[]{n});

        gpu_kernel_1.set_arg(0, src_data);
        gpu_kernel_1.new_arg(1, local_buffer_size);
        gpu_kernel_1.set_arg(2, src_part);
        gpu_kernel_1.set_arg(3, src_n);
        gpu_kernel_1.call(global_work_size, local_work_default);

        scan_int(p_data, part_size);

        int[] sz = new int[1];
        var size_data = cl_new_buffer(Sizeof.cl_int);

        gpu_kernel_2.set_arg(0, src_data);
        gpu_kernel_2.set_arg(1, Pointer.to(size_data));
        gpu_kernel_2.new_arg(2, local_buffer_size);
        gpu_kernel_2.set_arg(3, src_part);
        gpu_kernel_2.set_arg(4, src_n);
        gpu_kernel_2.call(global_work_size, local_work_default);

        cl_read_buffer(size_data, Sizeof.cl_int, Pointer.to(sz));

        clReleaseMemObject(size_data);
        clReleaseMemObject(p_data);

        return sz[0];
    }

    //#endregion
}
