package com.controllerface.bvge.cl;

import com.controllerface.bvge.Main;
import com.controllerface.bvge.cl.kernels.*;
import com.controllerface.bvge.cl.programs.*;
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

    /**
     * A convenience object, used when clearing out buffers to fill them with zeroes
     */
    private static final Pointer ZERO_PATTERN = Pointer.to(new int[]{0});

    /**
     * Memory that is shared between Open CL and Open GL contexts
     */
    private static final HashMap<Integer, cl_mem> shared_mem = new LinkedHashMap<>();
    //#endregion

    //#region Workgroup Variables

    /**
     * These values are re-calculated at startup to match the user's hardware. The max work group is the
     * largest group of calculations that can be done in a single "warp" or "wave" of GPU processing.
     * Related to this, we store a max scan block, which is used for variants of the prefix scan kernels.
     * The local work default is simply the max group size formatted as a single element argument array,
     * making it simpler to use for Open Cl calls which expect that format.
     */
    private static int max_work_group_size = 0;
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

    //#region Program Objects

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
        prepare_bounds(new PrepareBounds()),
        prepare_edges(new PrepareEdges()),
        prepare_transforms(new PrepareTransforms()),
        resolve_constraints(new ResolveConstraints()),
        root_hull_filter(new RootHullFilter()),
        sat_collide(new SatCollide()),
        scan_candidates(new ScanCandidates()),
        scan_deletes(new ScanDeletes()),
        scan_int_array(new ScanIntArray()),
        scan_int4_array(new ScanInt4Array()),
        scan_int_array_out(new ScanIntArrayOut()),
        scan_key_bank(new ScanKeyBank()),

        ;

        private final GPUProgram gpu;

        Program(GPUProgram program)
        {
            this.gpu = program;
        }
    }

    //#endregion

    //#region Kernel Objects

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
        compact_armatures,
        compact_bones,
        compact_edges,
        compact_hulls,
        compact_points,
        complete_bounds_multi_block,
        complete_candidates_multi_block_out,
        complete_deletes_multi_block_out,
        complete_int4_multi_block,
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
        locate_out_of_bounds,
        move_armatures,
        prepare_bones,
        prepare_bounds,
        prepare_edges,
        prepare_transforms,
        read_position,
        resolve_constraints,
        root_hull_count,
        root_hull_filter,
        rotate_hull,
        sat_collide,
        scan_bounds_multi_block,
        scan_bounds_single_block,
        scan_candidates_multi_block_out,
        scan_candidates_single_block_out,
        scan_deletes_multi_block_out,
        scan_deletes_single_block_out,
        scan_int4_multi_block,
        scan_int4_single_block,
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

    //#region Memory Objects

    /**
     * Memory buffers that store data used within the various kernel functions. Each buffer
     * has a different layout, but will align to an Open CL supported primitive type, such as
     * int, float or some vectorized type like, int2 or float4.
     */
    private enum Memory
    {
        /*
        Reference objects:
        - Objects in memory at runtime may reference these objects
        - Data stored in references should be considered immutable once written
         */

        /**
         * Vertex information for loaded models. Values are float2 with the following mappings:
         * -
         * x: x position
         * y: y position
         * -
         */
        vertex_references(Sizeof.cl_float2),

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


        /*
        Points
         */

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
         * Reaction counts for points on tracked physics hulls. Values are int with the following mapping:
         * -
         * value: reaction count
         * -
         */
        point_reactions(Sizeof.cl_int),

        /**
         * Reaction offsets for points on tracked physics hulls. Values are int with the following mapping:
         * -
         * value: reaction buffer offset
         * -
         */
        point_offsets(Sizeof.cl_int),

        /**
         * Non-real force modeled for stability of colliding particles. Values are float with the following mapping:
         * -
         * value: antigravity magnitude
         * -
         */
        point_anti_gravity(Sizeof.cl_float),

        /**
         * Indexing table for points of tracked physics hulls. Values are int2 with the following mappings:
         * -
         * x: reference vertex index
         * y: bone index (todo: also used as a proxy for hull ID, based on alignment, but they should be separate)
         * -
         */
        vertex_table(Sizeof.cl_int2),


        /*
        Edges
         */

        /**
         * Edges of tracked physics hulls. Values are float4 with the following mappings:
         * -
         * x: point 1 index
         * y: point 2 index
         * z: distance constraint
         * w: edge flags (bit-field)
         * -
         * note: x, y, and w values are cast to int during use
         */
        edges(Sizeof.cl_float4),


        /*
        Hulls
         */

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
         * x: hull flags (bit-field)
         * y: armature id
         * -
         */
        hull_flags(Sizeof.cl_int2),

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


        /*
        Bones
         */

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


        /*
        Armatures
         */

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
         * Flags for an armature. Values are int with the following mapping:
         * -
         * x: root hull index
         * y: model id
         * z: armature flags (bit-field)
         * w: -unused-
         * -
         */
        armature_flags(Sizeof.cl_int4),

        /**
         * Acceleration value of an armature. Values are float2 with the following mappings:
         * -
         * x: current x acceleration
         * y: current y acceleration
         * -
         */
        armature_accel(Sizeof.cl_float2),

        /**
         * Indexing table for tracked armatures. Values are int2 with the following mappings:
         * -
         * x: start hull index
         * y: end hull index
         * -
         */
        armature_hull_table(Sizeof.cl_int2),


        bone_shift(Sizeof.cl_int),

        point_shift(Sizeof.cl_int),

        edge_shift(Sizeof.cl_int),

        hull_shift(Sizeof.cl_int),
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

    //#region Utility Methods

    private static void cl_read_buffer(cl_mem src, long size, Pointer dst)
    {
        clEnqueueReadBuffer(command_queue,
            src,
            CL_TRUE,
            0,
            size,
            dst,
            0,
            null,
            null);
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
        clEnqueueFillBuffer(command_queue,
            buffer,
            ZERO_PATTERN,
            1,
            0,
            buffer_size,
            0,
            null,
            null);
    }

    private static int work_group_count(int n)
    {
        return (int) Math.ceil((float) n / (float) max_scan_block_size);
    }

    /**
     * Typically, kernels that operate on core objects are called with the maximum count and no group
     * size, allowing the OpenCL implementation to slice up all tasks into workgroups and queue them
     * as needed. However, in some cases it is necessary to ensure that, at most, only one workgroup
     * executes at a time. For example, buffer compaction, which must be computed in ascending order,
     * with a guarantee that items that are of a higher index value are always processed after ones
     * with lower values. This method serves the later use case. The provided kernel is called in a
     * loop, with each call containing a local work size equal to the global size, forcing all work
     * into a single work group. The loop uses a global offset to ensure that, on each iteration, the
     * next group is processed.
     *
     * @param kernel the GPU kernel to linearize
     * @param object_count the number of total kernel threads that will run
     */
    private static void linearize_kernel(GPUKernel kernel, int object_count)
    {
        int offset = 0;
        for (int remaining = object_count; remaining > 0; remaining -= max_work_group_size)
        {
            int count = Math.min(max_work_group_size, remaining);
            var sz = count == max_work_group_size
                ? local_work_default
                : arg_long(count);
            kernel.call(sz, sz, arg_long(offset));
            offset += count;
        }
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
        Memory.point_anti_gravity.init(max_points);
        Memory.edges.init(max_points);
        Memory.vertex_table.init(max_points);
        Memory.vertex_references.init(max_points);
        Memory.bone_references.init(max_points);
        Memory.bone_instances.init(max_points);
        Memory.bone_index.init(max_points);
        Memory.armatures.init(max_points);
        Memory.armature_flags.init(max_points);
        Memory.armature_hull_table.init(max_hulls);

        Memory.bone_shift.init(max_points);
        Memory.point_shift.init(max_points);
        Memory.edge_shift.init(max_points);
        Memory.hull_shift.init(max_hulls);


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
            + Memory.point_anti_gravity.length
            + Memory.edges.length
            + Memory.vertex_table.length
            + Memory.vertex_references.length
            + Memory.bone_references.length
            + Memory.bone_instances.length
            + Memory.bone_index.length
            + Memory.armatures.length
            + Memory.armature_flags.length
            + Memory.armature_hull_table.length
            + Memory.bone_shift.length
            + Memory.point_shift.length
            + Memory.edge_shift.length
            + Memory.hull_shift.length;

        System.out.println("------------- BUFFERS -------------");
        System.out.println("points            : " + Memory.points.length);
        System.out.println("edges             : " + Memory.edges.length);
        System.out.println("hulls             : " + Memory.hulls.length);
        System.out.println("acceleration      : " + Memory.armature_accel.length);
        System.out.println("rotation          : " + Memory.hull_rotation.length);
        System.out.println("element table     : " + Memory.hull_element_table.length);
        System.out.println("hull flags        : " + Memory.hull_flags.length);
        System.out.println("point reactions   : " + Memory.point_reactions.length);
        System.out.println("point offsets     : " + Memory.point_offsets.length);
        System.out.println("point anti-grav   : " + Memory.point_anti_gravity.length);
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
        System.out.println("hull table        : " + Memory.armature_hull_table.length);
        System.out.println("bone shift        : " + Memory.armature_hull_table.length);
        System.out.println("point shift       : " + Memory.armature_hull_table.length);
        System.out.println("edge shift        : " + Memory.armature_hull_table.length);
        System.out.println("hull shift        : " + Memory.armature_hull_table.length);
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
        // Open GL interop

        var prep_bounds_k = new PrepareBounds_k(command_queue, Program.prepare_bounds.gpu);
        prep_bounds_k.set_aabb(Memory.aabb.gpu.pointer());
        Kernel.prepare_bounds.set_kernel(prep_bounds_k);

        var prep_transforms_k = new PrepareTransforms_k(command_queue, Program.prepare_transforms.gpu);
        prep_transforms_k.set_hulls(Memory.hulls.gpu.pointer());
        prep_transforms_k.set_rotations(Memory.hull_rotation.gpu.pointer());
        Kernel.prepare_transforms.set_kernel(prep_transforms_k);

        var root_hull_count_k = new RootHullCount_k(command_queue, Program.root_hull_filter.gpu);
        root_hull_count_k.set_armature_flags(Memory.armature_flags.gpu.pointer());
        Kernel.root_hull_count.set_kernel(root_hull_count_k);

        var root_hull_filter_k = new RootHullFilter_k(command_queue, Program.root_hull_filter.gpu);
        root_hull_filter_k.set_armature_flags(Memory.armature_flags.gpu.pointer());
        Kernel.root_hull_filter.set_kernel(root_hull_filter_k);

        var prep_edges_k = new PrepareEdges_k(command_queue, Program.prepare_edges.gpu);
        prep_edges_k.set_points(Memory.points.gpu.pointer());
        prep_edges_k.set_edges(Memory.edges.gpu.pointer());
        Kernel.prepare_edges.set_kernel(prep_edges_k);

        var prep_bones_k = new PrepareBones_k(command_queue, Program.prepare_bones.gpu);
        prep_bones_k.set_bone_instances(Memory.bone_instances.gpu.pointer());
        prep_bones_k.set_bone_references(Memory.bone_references.gpu.pointer());
        prep_bones_k.set_bone_index(Memory.bone_index.gpu.pointer());
        prep_bones_k.set_hulls(Memory.hulls.gpu.pointer());
        prep_bones_k.set_armatures(Memory.armatures.gpu.pointer());
        prep_bones_k.set_hull_flags(Memory.hull_flags.gpu.pointer());
        Kernel.prepare_bones.set_kernel(prep_bones_k);


        // narrow collision

        var sat_collide_k = new SatCollide_k(command_queue, Program.sat_collide.gpu);
        sat_collide_k.set_hulls(Memory.hulls.gpu.pointer());
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
        apply_reactions_k.set_point_anti_grav(Memory.point_anti_gravity.gpu.pointer());
        apply_reactions_k.set_point_reactions(Memory.point_reactions.gpu.pointer());
        apply_reactions_k.set_point_offsets(Memory.point_offsets.gpu.pointer());
        Kernel.apply_reactions.set_kernel(apply_reactions_k);

        var move_armatures_k = new MoveArmatures_k(command_queue, Program.sat_collide.gpu);
        move_armatures_k.set_hulls(Memory.hulls.gpu.pointer());
        move_armatures_k.set_armatures(Memory.armatures.gpu.pointer());
        move_armatures_k.set_hull_tables(Memory.armature_hull_table.gpu.pointer());
        move_armatures_k.set_element_tables(Memory.hull_element_table.gpu.pointer());
        move_armatures_k.set_hull_flags(Memory.hull_flags.gpu.pointer());
        move_armatures_k.set_points(Memory.points.gpu.pointer());
        Kernel.move_armatures.set_kernel(move_armatures_k);


        // crud

        var create_point_k = new CreatePoint_k(command_queue, Program.gpu_crud.gpu);
        create_point_k.set_points(Memory.points.gpu.pointer());
        create_point_k.set_vertex_table(Memory.vertex_table.gpu.pointer());
        Kernel.create_point.set_kernel(create_point_k);

        var create_edge_k = new CreateEdge_k(command_queue, Program.gpu_crud.gpu);
        create_edge_k.set_edges(Memory.edges.gpu.pointer());
        Kernel.create_edge.set_kernel(create_edge_k);

        var create_vertex_ref_k = new CreateVertexRef_k(command_queue, Program.gpu_crud.gpu);
        create_vertex_ref_k.set_vertex_refs(Memory.vertex_references.gpu.pointer());
        Kernel.create_vertex_reference.set_kernel(create_vertex_ref_k);

        var create_bone_ref_k = new CreateBoneRef_k(command_queue, Program.gpu_crud.gpu);
        create_bone_ref_k.set_bone_refs(Memory.bone_references.gpu.pointer());
        Kernel.create_bone_reference.set_kernel(create_bone_ref_k);

        var create_armature_k = new CreateArmature_k(command_queue, Program.gpu_crud.gpu);
        create_armature_k.set_armatures(Memory.armatures.gpu.pointer());
        create_armature_k.set_armature_flags(Memory.armature_flags.gpu.pointer());
        create_armature_k.set_hull_table(Memory.armature_hull_table.gpu.pointer());
        Kernel.create_armature.set_kernel(create_armature_k);

        var create_bone_k = new CreateBone_k(command_queue, Program.gpu_crud.gpu);
        create_bone_k.set_bone_instances(Memory.bone_instances.gpu.pointer());
        create_bone_k.set_bone_index(Memory.bone_index.gpu.pointer());
        Kernel.create_bone.set_kernel(create_bone_k);

        var create_hull_k = new CreateHull_k(command_queue, Program.gpu_crud.gpu);
        create_hull_k.set_hulls(Memory.hulls.gpu.pointer());
        create_hull_k.set_hull_rotations(Memory.hull_rotation.gpu.pointer());
        create_hull_k.set_hull_flags(Memory.hull_flags.gpu.pointer());
        create_hull_k.set_element_table(Memory.hull_element_table.gpu.pointer());
        Kernel.create_hull.set_kernel(create_hull_k);

        var read_position_k = new ReadPosition_k(command_queue, Program.gpu_crud.gpu);
        read_position_k.set_armatures(Memory.armatures.gpu.pointer());
        Kernel.read_position.set_kernel(read_position_k);

        var update_accel_k = new UpdateAccel_k(command_queue, Program.gpu_crud.gpu);
        update_accel_k.set_accel(Memory.armature_accel.gpu.pointer());
        Kernel.update_accel.set_kernel(update_accel_k);


        // movement

        var animate_hulls_k = new AnimateHulls_k(command_queue, Program.animate_hulls.gpu);
        animate_hulls_k.set_points(Memory.points.gpu.pointer());
        animate_hulls_k.set_hulls(Memory.hulls.gpu.pointer());
        animate_hulls_k.set_hull_flags(Memory.hull_flags.gpu.pointer());
        animate_hulls_k.set_vertex_table(Memory.vertex_table.gpu.pointer());
        animate_hulls_k.set_armatures(Memory.armatures.gpu.pointer());
        animate_hulls_k.set_vertex_refs(Memory.vertex_references.gpu.pointer());
        animate_hulls_k.set_bone_instances(Memory.bone_instances.gpu.pointer());
        Kernel.animate_hulls.set_kernel(animate_hulls_k);

        var integrate_k = new Integrate_k(command_queue, Program.integrate.gpu);
        integrate_k.set_hulls(Memory.hulls.gpu.pointer());
        integrate_k.set_armatures(Memory.armatures.gpu.pointer());
        integrate_k.set_armature_flags(Memory.armature_flags.gpu.pointer());
        integrate_k.set_hull_element_table(Memory.hull_element_table.gpu.pointer());
        integrate_k.set_armature_accel(Memory.armature_accel.gpu.pointer());
        integrate_k.set_hull_rotation(Memory.hull_rotation.gpu.pointer());
        integrate_k.set_points(Memory.points.gpu.pointer());
        integrate_k.set_aabb(Memory.aabb.gpu.pointer());
        integrate_k.set_aabb_index(Memory.aabb_index.gpu.pointer());
        integrate_k.set_aabb_key_table(Memory.aabb_key_table.gpu.pointer());
        integrate_k.set_hull_flags(Memory.hull_flags.gpu.pointer());
        integrate_k.set_point_anti_gravity(Memory.point_anti_gravity.gpu.pointer());
        Kernel.integrate.set_kernel(integrate_k);


        // broad collision

        var generate_keys_k = new GenerateKeys_k(command_queue, Program.generate_keys.gpu);
        generate_keys_k.set_aabb_index(Memory.aabb_index.gpu.pointer());
        generate_keys_k.set_aabb_key_table(Memory.aabb_key_table.gpu.pointer());
        Kernel.generate_keys.set_kernel(generate_keys_k);

        var build_key_map_k = new BuildKeyMap_k(command_queue, Program.build_key_map.gpu);
        build_key_map_k.set_aabb_index(Memory.aabb_index.gpu.pointer());
        build_key_map_k.set_aabb_key_table(Memory.aabb_key_table.gpu.pointer());
        Kernel.build_key_map.set_kernel(build_key_map_k);

        var locate_in_bounds_k = new LocateInBounds_k(command_queue, Program.locate_in_bounds.gpu);
        locate_in_bounds_k.set_aabb_key_table(Memory.aabb_key_table.gpu.pointer());
        Kernel.locate_in_bounds.set_kernel(locate_in_bounds_k);

        var count_candidates_k = new  CountCandidates_k(command_queue, Program.locate_in_bounds.gpu);
        count_candidates_k.set_aabb_key_table(Memory.aabb_key_table.gpu.pointer());
        Kernel.count_candidates.set_kernel(count_candidates_k);

        var aabb_collide_k = new AABBCollide_k(command_queue, Program.aabb_collide.gpu);
        aabb_collide_k.set_aabb(Memory.aabb.gpu.pointer());
        aabb_collide_k.set_aabb_key_table(Memory.aabb_key_table.gpu.pointer());
        aabb_collide_k.set_hull_flags(Memory.hull_flags.gpu.pointer());
        Kernel.aabb_collide.set_kernel(aabb_collide_k);

        var finalize_candidates_k = new FinalizeCandidates_k(command_queue, Program.locate_in_bounds.gpu);
        Kernel.finalize_candidates.set_kernel(finalize_candidates_k);


        // constraint solver

        var resolve_constraints_k = new ResolveConstraints_k(command_queue, Program.resolve_constraints.gpu);
        resolve_constraints_k.set_hull_element_table(Memory.hull_element_table.gpu.pointer());
        resolve_constraints_k.set_aabb_key_table(Memory.aabb_key_table.gpu.pointer());
        resolve_constraints_k.set_points(Memory.points.gpu.pointer());
        resolve_constraints_k.set_edges(Memory.edges.gpu.pointer());
        Kernel.resolve_constraints.set_kernel(resolve_constraints_k);


        // integer exclusive scan in-place

        var scan_int_single_block_k = new ScanIntSingleBlock_k(command_queue, Program.scan_int_array.gpu);
        Kernel.scan_int_single_block.set_kernel(scan_int_single_block_k);

        var scan_int_multi_block_k = new ScanIntMultiBlock_k(command_queue, Program.scan_int_array.gpu);
        Kernel.scan_int_multi_block.set_kernel(scan_int_multi_block_k);

        var complete_int_multi_block_k = new CompleteIntMultiBlock_k(command_queue, Program.scan_int_array.gpu);
        Kernel.complete_int_multi_block.set_kernel(complete_int_multi_block_k);


        // vectorized integer exclusive scan in-place

        var scan_int4_single_block_k = new ScanInt4SingleBlock_k(command_queue, Program.scan_int4_array.gpu);
        Kernel.scan_int4_single_block.set_kernel(scan_int4_single_block_k);

        var scan_int4_multi_block_k = new ScanInt4MultiBlock_k(command_queue, Program.scan_int4_array.gpu);
        Kernel.scan_int4_multi_block.set_kernel(scan_int4_multi_block_k);

        var complete_int4_multi_block_k = new CompleteInt4MultiBlock_k(command_queue, Program.scan_int4_array.gpu);
        Kernel.complete_int4_multi_block.set_kernel(complete_int4_multi_block_k);


        // integer exclusive scan to output buffer

        var scan_int_single_block_out_k = new ScanIntSingleBlockOut_k(command_queue, Program.scan_int_array_out.gpu);
        Kernel.scan_int_single_block_out.set_kernel(scan_int_single_block_out_k);

        var scan_int_multi_block_out_k = new ScanIntMultiBlockOut_k(command_queue, Program.scan_int_array_out.gpu);
        Kernel.scan_int_multi_block_out.set_kernel(scan_int_multi_block_out_k);

        var complete_int_multi_block_out_k = new CompleteIntMultiBlockOut_k(command_queue, Program.scan_int_array_out.gpu);
        Kernel.complete_int_multi_block_out.set_kernel(complete_int_multi_block_out_k);


        // collision candidate scan to output buffer

        var scan_candidates_single_block_out_k = new ScanCandidatesSingleBlockOut_k(command_queue, Program.scan_candidates.gpu);
        Kernel.scan_candidates_single_block_out.set_kernel(scan_candidates_single_block_out_k);

        var scan_candidates_multi_block_out_k = new ScanCandidatesMultiBlockOut_k(command_queue, Program.scan_candidates.gpu);
        Kernel.scan_candidates_multi_block_out.set_kernel(scan_candidates_multi_block_out_k);

        var complete_candidates_multi_block_out_k = new CompleteCandidatesMultiBlockOut_k(command_queue, Program.scan_candidates.gpu);
        Kernel.complete_candidates_multi_block_out.set_kernel(complete_candidates_multi_block_out_k);


        // in-place uniform grid key bounds scan

        var scan_bounds_single_block_k = new ScanBoundsSingleBlock_k(command_queue, Program.scan_key_bank.gpu);
        Kernel.scan_bounds_single_block.set_kernel(scan_bounds_single_block_k);

        var scan_bounds_multi_block_k = new ScanBoundsMultiBlock_k(command_queue, Program.scan_key_bank.gpu);
        Kernel.scan_bounds_multi_block.set_kernel(scan_bounds_multi_block_k);

        var complete_bounds_multi_block_k = new CompleteBoundsMultiBlock_k(command_queue, Program.scan_key_bank.gpu);
        Kernel.complete_bounds_multi_block.set_kernel(complete_bounds_multi_block_k);


        // scan for deleted objects

        var locate_out_of_bounds_k = new LocateOutOfBounds_k(command_queue, Program.scan_deletes.gpu);
        locate_out_of_bounds_k.set_hull_tables(Memory.armature_hull_table.gpu.pointer());
        locate_out_of_bounds_k.set_hull_flags(Memory.hull_flags.gpu.pointer());
        locate_out_of_bounds_k.set_armature_flags(Memory.armature_flags.gpu.pointer());
        Kernel.locate_out_of_bounds.set_kernel(locate_out_of_bounds_k);

        var scan_deletes_single_block_out_k = new ScanDeletesSingleBlockOut_k(command_queue, Program.scan_deletes.gpu);
        scan_deletes_single_block_out_k.set_armature_flags(Memory.armature_flags.gpu.pointer());
        scan_deletes_single_block_out_k.set_hull_tables(Memory.armature_hull_table.gpu.pointer());
        scan_deletes_single_block_out_k.set_element_tables(Memory.hull_element_table.gpu.pointer());
        Kernel.scan_deletes_single_block_out.set_kernel(scan_deletes_single_block_out_k);

        var scan_deletes_multi_block_out_k = new ScanDeletesMultiBlockOut_k(command_queue, Program.scan_deletes.gpu);
        scan_deletes_multi_block_out_k.set_armature_flags(Memory.armature_flags.gpu.pointer());
        scan_deletes_multi_block_out_k.set_hull_tables(Memory.armature_hull_table.gpu.pointer());
        scan_deletes_multi_block_out_k.set_element_tables(Memory.hull_element_table.gpu.pointer());
        Kernel.scan_deletes_multi_block_out.set_kernel(scan_deletes_multi_block_out_k);

        var complete_deletes_multi_block_out_k = new CompleteDeletesMultiBlockOut_k(command_queue, Program.scan_deletes.gpu);
        complete_deletes_multi_block_out_k.set_armature_flags(Memory.armature_flags.gpu.pointer());
        complete_deletes_multi_block_out_k.set_hull_tables(Memory.armature_hull_table.gpu.pointer());
        complete_deletes_multi_block_out_k.set_element_tables(Memory.hull_element_table.gpu.pointer());
        Kernel.complete_deletes_multi_block_out.set_kernel(complete_deletes_multi_block_out_k);

        var compact_armatures_k = new CompactArmatures_k(command_queue, Program.scan_deletes.gpu);
        compact_armatures_k.set_armatures(Memory.armatures.gpu.pointer());
        compact_armatures_k.set_armature_accel(Memory.armature_accel.gpu.pointer());
        compact_armatures_k.set_armature_flags(Memory.armature_flags.gpu.pointer());
        compact_armatures_k.set_hull_tables(Memory.armature_hull_table.gpu.pointer());
        compact_armatures_k.set_hulls(Memory.hulls.gpu.pointer());
        compact_armatures_k.set_hull_flags(Memory.hull_flags.gpu.pointer());
        compact_armatures_k.set_element_tables(Memory.hull_element_table.gpu.pointer());
        compact_armatures_k.set_points(Memory.points.gpu.pointer());
        compact_armatures_k.set_vertex_tables(Memory.vertex_table.gpu.pointer());
        compact_armatures_k.set_edges(Memory.edges.gpu.pointer());
        compact_armatures_k.set_bone_shift(Memory.bone_shift.gpu.pointer());
        compact_armatures_k.set_point_shift(Memory.point_shift.gpu.pointer());
        compact_armatures_k.set_edge_shift(Memory.edge_shift.gpu.pointer());
        compact_armatures_k.set_hull_shift(Memory.hull_shift.gpu.pointer());
        Kernel.compact_armatures.set_kernel(compact_armatures_k);

        var compact_hulls_k = new CompactHulls_k(command_queue, Program.scan_deletes.gpu);
        compact_hulls_k.set_hull_shift(Memory.hull_shift.gpu.pointer());
        compact_hulls_k.set_hulls(Memory.hulls.gpu.pointer());
        compact_hulls_k.set_hull_rotations(Memory.hull_rotation.gpu.pointer());
        compact_hulls_k.set_hull_flags(Memory.hull_flags.gpu.pointer());
        compact_hulls_k.set_element_tables(Memory.hull_element_table.gpu.pointer());
        compact_hulls_k.set_bounds(Memory.aabb.gpu.pointer());
        compact_hulls_k.set_bounds_index(Memory.aabb_index.gpu.pointer());
        compact_hulls_k.set_bounds_bank(Memory.aabb_key_table.gpu.pointer());
        Kernel.compact_hulls.set_kernel(compact_hulls_k);

        var compact_edges_k = new CompactEdges_k(command_queue, Program.scan_deletes.gpu);
        compact_edges_k.set_edge_shift(Memory.edge_shift.gpu.pointer());
        compact_edges_k.set_edges(Memory.edges.gpu.pointer());
        Kernel.compact_edges.set_kernel(compact_edges_k);

        var compact_points_k = new CompactPoints_k(command_queue, Program.scan_deletes.gpu);
        compact_points_k.set_point_shift(Memory.point_shift.gpu.pointer());
        compact_points_k.set_points(Memory.points.gpu.pointer());
        compact_points_k.set_anti_gravity(Memory.point_anti_gravity.gpu.pointer());
        compact_points_k.set_vertex_tables(Memory.vertex_table.gpu.pointer());
        Kernel.compact_points.set_kernel(compact_points_k);

        var compact_bones_k = new CompactBones_k(command_queue, Program.scan_deletes.gpu);
        compact_bones_k.set_bone_shift(Memory.bone_shift.gpu.pointer());
        compact_bones_k.set_bone_instances(Memory.bone_instances.gpu.pointer());
        compact_bones_k.set_bone_indices(Memory.bone_index.gpu.pointer());
        Kernel.compact_bones.set_kernel(compact_bones_k);
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
     * @param vbo_id2     id of the shared GL buffer object
     * @param edge_offset offset into the edges array to start the transfer
     * @param batch_size  number of edge objects to transfer in this batch
     */
    public static void GL_edges(int vbo_id, int vbo_id2, int edge_offset, int batch_size)
    {
        var gpu_kernel = Kernel.prepare_edges.gpu;

        var vbo_mem = shared_mem.get(vbo_id);
        var vbo_mem2 = shared_mem.get(vbo_id2);

        gpu_kernel.share_mem(vbo_mem);
        gpu_kernel.share_mem(vbo_mem2);

        gpu_kernel.set_arg(2, Pointer.to(vbo_mem));
        gpu_kernel.set_arg(3, Pointer.to(vbo_mem2));
        gpu_kernel.set_arg(4, Pointer.to(arg_long(edge_offset)));
        gpu_kernel.call(arg_long(batch_size));
    }

    /**
     * Performs a filter query on all physics hulls, returning an index buffer and count of items.
     * The returned object will contain the indices of all hulls that match the model with the given ID.
     *
     * @param model_id ID of model to filter on
     * @return a HullIndexData object with the query result
     */
    public static HullIndexData GL_hull_filter(int model_id)
    {
        var gpu_kernel = Kernel.root_hull_count.gpu;
        var gpu_kernel_2 = Kernel.root_hull_filter.gpu;

        // the kernel will use this value as an internal atomic counter, always initialize to zero
        int[] counter = new int[]{0};
        var dst_counter = Pointer.to(counter);
        var counter_data = cl_new_int_arg_buffer(dst_counter);

        gpu_kernel.set_arg(1, Pointer.to(counter_data));
        gpu_kernel.set_arg(2, Pointer.to(arg_int(model_id)));

        gpu_kernel.call(arg_long(Main.Memory.next_armature()));
        cl_read_buffer(counter_data, Sizeof.cl_int, dst_counter);

        clReleaseMemObject(counter_data);

        int final_count = counter[0];
        if (final_count == 0)
        {
            return new HullIndexData(null, final_count);
        }

        long final_buffer_size = (long) Sizeof.cl_int * final_count;
        var hulls_out = cl_new_buffer(final_buffer_size);

        // the kernel will use this value as an internal atomic counter, always initialize to zero
        int[] hulls_counter = new int[]{0};
        var dst_hulls_counter = Pointer.to(hulls_counter);
        var hulls_counter_data = cl_new_int_arg_buffer(dst_hulls_counter);

        gpu_kernel_2.set_arg(1,Pointer.to(hulls_out));
        gpu_kernel_2.set_arg(2,Pointer.to(hulls_counter_data));
        gpu_kernel_2.set_arg(3,Pointer.to(arg_int(model_id)));

        gpu_kernel_2.call(arg_long(Main.Memory.next_armature()));

        clReleaseMemObject(hulls_counter_data);

        return new HullIndexData(hulls_out, final_count);
    }

    /**
     * Transfers a subset of all hull transforms from CL memory into GL memory. Hulls
     * are generally not rendered directly using this data, but it is used to transform
     * model reference data from memory into the position of the mesh that the hull
     * represents within the simulation.
     *
     * @param vbo_id        id of the shared GL buffer object
     * @param hulls_out     array of hulls filtered to be circles only
     * @param offset        where we are starting in the indices array
     * @param batch_size         number of hull objects to transfer in this batch
     */
    public static void GL_circles(int vbo_id, cl_mem hulls_out, int offset, int batch_size)
    {
        var gpu_kernel = Kernel.prepare_transforms.gpu;

        var vbo_index_buffer = shared_mem.get(vbo_id);
        gpu_kernel.share_mem(vbo_index_buffer);

        gpu_kernel.set_arg(2, Pointer.to(hulls_out));
        gpu_kernel.set_arg(3, Pointer.to(vbo_index_buffer));
        gpu_kernel.set_arg(4, Pointer.to(arg_int(offset)));
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
    public static void GL_transforms(int index_buffer_id, int transforms_id, int batch_size, int offset)
    {
        var gpu_kernel = Kernel.prepare_transforms.gpu;

        var vbo_index_buffer = shared_mem.get(index_buffer_id);
        var vbo_transforms = shared_mem.get(transforms_id);

        gpu_kernel.share_mem(vbo_index_buffer);
        gpu_kernel.share_mem(vbo_transforms);
        gpu_kernel.set_arg(2, Pointer.to(vbo_index_buffer));
        gpu_kernel.set_arg(3, Pointer.to(vbo_transforms));
        gpu_kernel.set_arg(4, Pointer.to(arg_int(offset)));
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

    public static void create_armature(int armature_index, float x, float y, int[] table, int[] flags)
    {
        var gpu_kernel = Kernel.create_armature.gpu;
        gpu_kernel.set_arg(3, Pointer.to(arg_int(armature_index)));
        gpu_kernel.set_arg(4, Pointer.to(arg_float4(x, y, x, y)));
        gpu_kernel.set_arg(5, Pointer.to(flags));
        gpu_kernel.set_arg(6, Pointer.to(table));
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
//        var pnt_index = Pointer.to(arg_int(hull_index));
//        var pnt_angle = Pointer.to(arg_float(angle));
//
//        clSetKernelArg(_k.get(Kernel.rotate_hull), 0, Sizeof.cl_mem, Memory.hulls.gpu.pointer());
//        clSetKernelArg(_k.get(Kernel.rotate_hull), 1, Sizeof.cl_mem, Memory.hull_element_table.gpu.pointer());
//        clSetKernelArg(_k.get(Kernel.rotate_hull), 2, Sizeof.cl_mem, Memory.points.gpu.pointer());
//        clSetKernelArg(_k.get(Kernel.rotate_hull), 3, Sizeof.cl_int, pnt_index);
//        clSetKernelArg(_k.get(Kernel.rotate_hull), 4, Sizeof.cl_float, pnt_angle);
//
//        k_call(command_queue, _k.get(Kernel.rotate_hull), global_single_size);
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
        Kernel.animate_hulls.gpu.call(arg_long(Main.Memory.next_point()));
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
                physics_buffer.get_damping()
            };

        var srcArgs = Pointer.to(args);

        long size = Sizeof.cl_float * args.length;
        var argMem = cl_new_read_only_buffer(size, srcArgs);

        gpu_kernel.set_arg(12, Pointer.to(argMem));
        gpu_kernel.call(arg_long(Main.Memory.next_hull()));

        clReleaseMemObject(argMem);
    }

    public static void calculate_bank_offsets(UniformGrid uniform_grid)
    {
        int bank_size = scan_key_bounds(Memory.aabb_key_table.gpu.memory(), Main.Memory.next_hull());
        uniform_grid.resizeBank(bank_size);
    }

    public static void generate_keys(UniformGrid uniform_grid)
    {
        if (uniform_grid.get_key_bank_size() < 1)
        {
            return;
        }

        var gpu_kernel = Kernel.generate_keys.gpu;

        long bank_buf_size = (long) Sizeof.cl_int * uniform_grid.get_key_bank_size();
        long counts_buf_size = (long) Sizeof.cl_int * uniform_grid.get_directory_length();

        var bank_data = cl_new_buffer(bank_buf_size);
        var counts_data = cl_new_buffer(counts_buf_size);
        cl_zero_buffer(counts_data, counts_buf_size);

        physics_buffer.key_counts = new GPUMemory(counts_data);
        physics_buffer.key_bank = new GPUMemory(bank_data);

        gpu_kernel.set_arg(2, Pointer.to(bank_data));
        gpu_kernel.set_arg(3, Pointer.to(counts_data));
        gpu_kernel.set_arg(4, Pointer.to(arg_int(uniform_grid.getX_subdivisions())));
        gpu_kernel.set_arg(5, Pointer.to(arg_int(uniform_grid.get_key_bank_size())));
        gpu_kernel.set_arg(6, Pointer.to(arg_int(uniform_grid.get_directory_length())));
        gpu_kernel.call(arg_long(Main.Memory.next_hull()));
    }

    public static void calculate_map_offsets(UniformGrid uniform_grid)
    {
        int n = uniform_grid.get_directory_length();
        long data_buf_size = (long) Sizeof.cl_int * n;
        var o_data = cl_new_buffer(data_buf_size);
        physics_buffer.key_offsets = new GPUMemory(o_data);
        scan_int_out(physics_buffer.key_counts.memory(), o_data, n);
    }

    public static void build_key_map(UniformGrid uniform_grid)
    {
        var gpu_kernel = Kernel.build_key_map.gpu;

        long map_buf_size = (long) Sizeof.cl_int * uniform_grid.getKey_map_size();
        long counts_buf_size = (long) Sizeof.cl_int * uniform_grid.get_directory_length();

        var map_data = cl_new_buffer(map_buf_size);
        var counts_data = cl_new_buffer(counts_buf_size);

        // the counts buffer needs to start off filled with all zeroes
        cl_zero_buffer(counts_data, counts_buf_size);

        physics_buffer.key_map = new GPUMemory(map_data);

        gpu_kernel.set_arg(2, Pointer.to(map_data));
        gpu_kernel.set_arg(3, physics_buffer.key_offsets.pointer());
        gpu_kernel.set_arg(4, Pointer.to(counts_data));
        gpu_kernel.set_arg(5, Pointer.to(arg_int(uniform_grid.getX_subdivisions())));
        gpu_kernel.set_arg(6, Pointer.to(arg_int(uniform_grid.get_directory_length())));
        gpu_kernel.call(arg_long(Main.Memory.next_hull()));

        clReleaseMemObject(counts_data);
    }

    public static void locate_in_bounds(UniformGrid uniform_grid)
    {
        var gpu_kernel = Kernel.locate_in_bounds.gpu;

        int hull_count = Main.Memory.next_hull();

        int x_subdivisions = uniform_grid.getX_subdivisions();
        physics_buffer.x_sub_divisions = Pointer.to(arg_int(x_subdivisions));
        physics_buffer.key_count_length = Pointer.to(arg_int(uniform_grid.get_directory_length()));

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

    public static void locate_out_of_bounds()
    {
        var gpu_kernel = Kernel.locate_out_of_bounds.gpu;
        int armature_count = Main.Memory.next_armature();

        int[] counter = new int[]{0};
        var dst_counter = Pointer.to(counter);
        var counter_data = cl_new_int_arg_buffer(dst_counter);
        var p = Pointer.to(counter_data);
        gpu_kernel.set_arg(3, p);
        gpu_kernel.call(arg_long(armature_count));

        clReleaseMemObject(counter_data);
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

    public static void delete_and_compact()
    {
        int armature_count = Main.Memory.next_armature();
        long output_buf_size = (long) Sizeof.cl_int * armature_count;
        long output_buf_size2 = (long) Sizeof.cl_int4 * armature_count;

        var output_buf_data = cl_new_buffer(output_buf_size);
        var output_buf_data2 = cl_new_buffer(output_buf_size2);

        var del_buffer_1 = new GPUMemory(output_buf_data);
        var del_buffer_2 = new GPUMemory(output_buf_data2);

        int[] m = scan_deletes(del_buffer_1.memory(), del_buffer_2.memory(), armature_count);

        if (m[0] == 0)
        {
            del_buffer_1.release();
            del_buffer_2.release();
            return;
        }

        // shift buffers are cleared before compacting
        Memory.hull_shift.clear();
        Memory.edge_shift.clear();
        Memory.point_shift.clear();
        Memory.bone_shift.clear();

        // as armatures are compacted, the shift buffers for the other components are updated
        var armature_kernel = Kernel.compact_armatures.gpu;
        armature_kernel.set_arg(0, del_buffer_1.pointer());
        armature_kernel.set_arg(1, del_buffer_2.pointer());

        linearize_kernel(armature_kernel, armature_count);
        linearize_kernel(Kernel.compact_bones.gpu, Main.Memory.next_bone());
        linearize_kernel(Kernel.compact_points.gpu, Main.Memory.next_point());
        linearize_kernel(Kernel.compact_edges.gpu, Main.Memory.next_edge());
        linearize_kernel(Kernel.compact_hulls.gpu, Main.Memory.next_hull());

        Main.Memory.compact_buffers(m[0], m[1], m[2], m[3], m[4]);

        del_buffer_1.release();
        del_buffer_2.release();
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
        var reaction_data_out = cl_new_buffer(reaction_buf_size);
        var index_data = cl_new_buffer(index_buf_size);

        physics_buffer.reactions_in = new GPUMemory(reaction_data);
        physics_buffer.reactions_out = new GPUMemory(reaction_data_out);
        physics_buffer.reaction_index = new GPUMemory(index_data);

        gpu_kernel.set_arg(0, physics_buffer.candidates.pointer());
        gpu_kernel.set_arg(6, physics_buffer.reactions_in.pointer());
        gpu_kernel.set_arg(7, physics_buffer.reaction_index.pointer());
        gpu_kernel.set_arg(9, Pointer.to(size_data));

        gpu_kernel.call(global_work_size);

        cl_read_buffer(size_data, Sizeof.cl_int, dst_size);

        clReleaseMemObject(size_data);

        physics_buffer.set_reaction_count(size[0]);
    }

    public static void scan_reactions()
    {
        scan_int_out(Memory.point_reactions.gpu.memory(), Memory.point_offsets.gpu.memory(), Main.Memory.next_point());

        // it is important to zero out the reactions buffer after the scan. It will be reused during sorting
        Memory.point_reactions.clear();
    }

    public static void sort_reactions()
    {
        var gpu_kernel = Kernel.sort_reactions.gpu;
        gpu_kernel.set_arg(0, physics_buffer.reactions_in.pointer());
        gpu_kernel.set_arg(1, physics_buffer.reactions_out.pointer());
        gpu_kernel.set_arg(2, physics_buffer.reaction_index.pointer());
        gpu_kernel.call(arg_long(physics_buffer.get_reaction_count()));
    }

    public static void apply_reactions()
    {
        var gpu_kernel = Kernel.apply_reactions.gpu;
        gpu_kernel.set_arg(0, physics_buffer.reactions_out.pointer());
        gpu_kernel.call(arg_long(Main.Memory.next_point()));
    }

    public static void move_armatures()
    {
        Kernel.move_armatures.gpu.call(arg_long(Main.Memory.next_armature()));
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
            gpu_kernel.call(arg_long(Main.Memory.next_hull()));
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

    private static void scan_int4(cl_mem d_data, int n)
    {
        int k = work_group_count(n);
        if (k == 1)
        {
            scan_single_block_int4(d_data, n);
        }
        else
        {
            scan_multi_block_int4(d_data, n, k);
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

    private static int[] scan_deletes(cl_mem o1_data, cl_mem o2_data, int n)
    {
        int k = work_group_count(n);
        if (k == 1)
        {
            return scan_single_block_deletes_out(o1_data, o2_data, n);
        }
        else
        {
            return scan_multi_block_deletes_out(o1_data, o2_data, n, k);
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


    private static void scan_single_block_int4(cl_mem d_data, int n)
    {
        var gpu_kernel = Kernel.scan_int4_single_block.gpu;

        long local_buffer_size = Sizeof.cl_int4 * max_scan_block_size;

        gpu_kernel.set_arg(0, Pointer.to(d_data));
        gpu_kernel.new_arg(1, local_buffer_size, null);
        gpu_kernel.set_arg(2, Pointer.to(arg_int(n)));
        gpu_kernel.call(local_work_default, local_work_default);
    }

    private static void scan_multi_block_int4(cl_mem d_data, int n, int k)
    {
        var gpu_kernel_1 = Kernel.scan_int4_multi_block.gpu;
        var gpu_kernel_2 = Kernel.complete_int4_multi_block.gpu;

        long local_buffer_size = Sizeof.cl_int4 * max_scan_block_size;
        long gx = k * max_scan_block_size;
        long[] global_work_size = arg_long(gx);
        int part_size = k * 2;
        long part_buf_size = ((long) Sizeof.cl_int4 * ((long) part_size));

        var part_data = cl_new_buffer(part_buf_size);
        var src_data = Pointer.to(d_data);
        var src_part = Pointer.to(part_data);
        var src_n = Pointer.to(new int[]{n});

        gpu_kernel_1.set_arg(0, src_data);
        gpu_kernel_1.new_arg(1, local_buffer_size);
        gpu_kernel_1.set_arg(2, src_part);
        gpu_kernel_1.set_arg(3, src_n);
        gpu_kernel_1.call(global_work_size, local_work_default);

        scan_int4(part_data, part_size);

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

    private static int[] scan_single_block_deletes_out(cl_mem o1_data, cl_mem o2_data, int n)
    {
        var gpu_kernel = Kernel.scan_deletes_single_block_out.gpu;
        long local_buffer_size = Sizeof.cl_int * max_scan_block_size;
        long local_buffer_size2 = Sizeof.cl_int4 * max_scan_block_size;

        int[] sz = new int[]{ 0, 0, 0, 0, 0 };
        var dst_size = Pointer.to(sz);
        var size_data = cl_new_buffer(Sizeof.cl_int + Sizeof.cl_int4);
        var src_size = Pointer.to(size_data);

        var dst_data = Pointer.to(o1_data);
        var dst_data2 = Pointer.to(o2_data);

        gpu_kernel.set_arg(3, dst_data);
        gpu_kernel.set_arg(4, dst_data2);
        gpu_kernel.set_arg(5, src_size);
        gpu_kernel.new_arg(6, local_buffer_size);
        gpu_kernel.new_arg(7, local_buffer_size2);
        gpu_kernel.set_arg(8, Pointer.to(arg_int(n)));
        gpu_kernel.call(local_work_default, local_work_default);

        cl_read_buffer(size_data, Sizeof.cl_int + Sizeof.cl_int4, dst_size);

        clReleaseMemObject(size_data);

        return sz;
    }

    private static int[] scan_multi_block_deletes_out(cl_mem o1_data, cl_mem o2_data, int n, int k)
    {
        var gpu_kernel_1 = Kernel.scan_deletes_multi_block_out.gpu;
        var gpu_kernel_2 = Kernel.complete_deletes_multi_block_out.gpu;

        long local_buffer_size = Sizeof.cl_int * max_scan_block_size;
        long local_buffer_size2 = Sizeof.cl_int4 * max_scan_block_size;

        long gx = k * max_scan_block_size;
        long[] global_work_size = arg_long(gx);
        int part_size = k * 2;

        long part_buf_size = ((long) Sizeof.cl_int * ((long) part_size));
        long part_buf_size2 = ((long) Sizeof.cl_int4 * ((long) part_size));

        var p_data = cl_new_buffer(part_buf_size);
        var p_data2 = cl_new_buffer(part_buf_size2);

        var dst_data = Pointer.to(o1_data);
        var src_part = Pointer.to(p_data);
        var dst_data2 = Pointer.to(o2_data);
        var src_part2 = Pointer.to(p_data2);
        var src_n = Pointer.to(new int[]{n});

        gpu_kernel_1.set_arg(3, dst_data);
        gpu_kernel_1.set_arg(4, dst_data2);
        gpu_kernel_1.new_arg(5, local_buffer_size);
        gpu_kernel_1.new_arg(6, local_buffer_size2);
        gpu_kernel_1.set_arg(7, src_part);
        gpu_kernel_1.set_arg(8, src_part2);
        gpu_kernel_1.set_arg(9, src_n);
        gpu_kernel_1.call(global_work_size, local_work_default);

        // note the partial buffers are scanned and updated in-place
        scan_int(p_data, part_size);
        scan_int4(p_data2, part_size);

        int[] sz = new int[]{ 0, 0, 0, 0, 0 };
        var dst_size = Pointer.to(sz);
        var size_data = cl_new_buffer(Sizeof.cl_int + Sizeof.cl_int4);
        var src_size = Pointer.to(size_data);

        gpu_kernel_2.set_arg(3, dst_data);
        gpu_kernel_2.set_arg(4, dst_data2);
        gpu_kernel_2.set_arg(5, src_size);
        gpu_kernel_2.new_arg(6, local_buffer_size);
        gpu_kernel_2.new_arg(7, local_buffer_size2);
        gpu_kernel_2.set_arg(8, src_part);
        gpu_kernel_2.set_arg(9, src_part2);
        gpu_kernel_2.set_arg(10, src_n);
        gpu_kernel_2.call(global_work_size, local_work_default);

        cl_read_buffer(size_data, Sizeof.cl_int * 5, dst_size);

        clReleaseMemObject(p_data);
        clReleaseMemObject(p_data2);
        clReleaseMemObject(size_data);

        return sz;
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

    //#region Misc. Public API

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

        max_work_group_size = (int) getSize(device, CL_DEVICE_MAX_WORK_GROUP_SIZE);
        max_scan_block_size = (long) max_work_group_size * 2;
        local_work_default = arg_long(max_work_group_size);

        // initialize gpu programs
        for (var program : Program.values())
        {
            program.gpu.init();
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

        for (Program program : Program.values())
        {
            Optional.ofNullable(program.gpu)
                .ifPresent(GPUProgram::destroy);
        }

        shared_mem.values().forEach(CL::clReleaseMemObject);

        clReleaseCommandQueue(command_queue);
        clReleaseContext(context);
    }

    //#endregion
}
