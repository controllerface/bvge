package com.controllerface.bvge.cl;

import com.controllerface.bvge.cl.kernels.*;
import com.controllerface.bvge.cl.programs.*;
import com.controllerface.bvge.gpu.GPUCoreMemory;
import org.lwjgl.BufferUtils;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.IntBuffer;
import java.util.List;

import static com.controllerface.bvge.cl.CLUtils.*;
import static org.lwjgl.opencl.CL12.*;
import static org.lwjgl.opencl.CL12GL.clCreateFromGLBuffer;
import static org.lwjgl.opencl.KHRGLSharing.CL_GL_CONTEXT_KHR;
import static org.lwjgl.opencl.KHRGLSharing.CL_WGL_HDC_KHR;
import static org.lwjgl.opengl.WGL.wglGetCurrentContext;
import static org.lwjgl.opengl.WGL.wglGetCurrentDC;

/**
 * Core class used for executing General Purpose GPU (GPGPU) functions
 */
public class GPGPU
{
    //#region Constants

    private static final long FLAGS_WRITE_GPU = CL_MEM_READ_WRITE;
    private static final long FLAGS_WRITE_CPU_COPY = CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR;
    private static final long FLAGS_READ_CPU_COPY = CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR;

    /**
     * A convenience object, used when clearing out buffers to fill them with zeroes
     */
    private static final ByteBuffer ZERO_PATTERN_BUFFER = BufferUtils.createByteBuffer(1)
        .put(0, (byte) 0);

    //#endregion

    //#region Workgroup Variables

    /*
      These values are re-calculated at startup to match the user's hardware.
     */

    /**
     * The largest group of calculations that can be done in a single "warp" or "wave" of GPU processing.
     */
    public static long max_work_group_size = 0;

    /**
     * Used for the prefix scan kernels and their variants.
     */
    public static long max_scan_block_size = 0;

    /**
     * The max group size formatted as a single element array, making it simpler to use for Open Cl calls.
     */
    public static long[] local_work_default = arg_long(0);

    /**
     * This convenience array defines a work group size of 1, used primarily for setting up data buffers at
     * startup. Kernels of this size should be used sparingly, favor making bulk calls. However, there are
     * specific use cases where it makes sense to perform a singular operation on GPU memory.
     */
    public static final long[] global_single_size = arg_long(1);

    //#endregion

    //#region Class Variables

    /**
     * The Open CL command queue that this class uses to issue GPU commands.
     */
    public static long command_queue_ptr;

    /**
     * The Open CL context associated with this class.
     */
    private static long context_ptr;

    /**
     * An array of devices that support being used with Open CL. In practice, this should
     * only ever have single element, and that device should be the main GPU in the system.
     */
    private static long device_id_ptr;

    /**
     * There are several kernels that use an atomic counter, so rather than re-allocate a new
     * buffer for every call, this buffer is reused in all kernels that need a counter.
     */
    private static long atomic_counter_ptr;

    public static GPUCoreMemory core_memory;

    //#endregion

    //#region Program Objects

    private enum Program
    {
        root_hull_filter(new RootHullFilter()),
        scan_int2_array(new ScanInt2Array()),
        scan_int4_array(new ScanInt4Array()),
        scan_int_array(new ScanIntArray()),
        scan_int_array_out(new ScanIntArrayOut()),

        ;

        public final GPUProgram gpu;

        Program(GPUProgram program)
        {
            this.gpu = program;
        }
    }

    //#endregion

    //#region Kernel Objects

    private static GPUKernel scan_int_single_block_k;
    private static GPUKernel scan_int_multi_block_k;
    private static GPUKernel complete_int_multi_block_k;
    private static GPUKernel scan_int2_single_block_k;
    private static GPUKernel scan_int2_multi_block_k;
    private static GPUKernel complete_int2_multi_block_k;
    private static GPUKernel scan_int4_single_block_k;
    private static GPUKernel scan_int4_multi_block_k;
    private static GPUKernel complete_int4_multi_block_k;
    private static GPUKernel scan_int_single_block_out_k;
    private static GPUKernel scan_int_multi_block_out_k;
    private static GPUKernel complete_int_multi_block_out_k;
    private static GPUKernel root_hull_count_k;
    private static GPUKernel root_hull_filter_k;

    //#endregion

    //#region Buffer Objects

    public enum Buffer
    {
        /*
        Reference objects:
        - Objects in memory at runtime may reference these objects
        - Data stored in references should be considered immutable once written
         */

        /**
         * x: x position
         * y: y position
         */
        vertex_references(CLSize.cl_float2),

        /**
         * x: bone 1 weight
         * y: bone 2 weight
         * z: bone 3 weight
         * w: bone 4 weight
         */
        vertex_weights(CLSize.cl_float4),

        /**
         * x: u coordinate
         * y: v coordinate
         */
        texture_uvs(CLSize.cl_float2),

        /**
         * x: start UV index
         * y: end UV index
         */
        uv_tables(CLSize.cl_int2),

        /**
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
        model_transforms(CLSize.cl_float16),

        /**
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
        bone_references(CLSize.cl_float16),

        /**
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
        bone_bind_poses(CLSize.cl_float16),

        /**
         * value: reference index of the parent bone bind pose
         */
        bone_bind_parents(CLSize.cl_int),

        /**
         * x: start vertex index
         * y: end vertex index
         * z: start face index
         * w: end face index
         */
        mesh_references(CLSize.cl_int4),

        /**
         * x: vertex 1 index
         * y: vertex 2 index
         * z: vertex 3 index
         * w: parent reference mesh ID
         */
        mesh_faces(CLSize.cl_int4),

        /**
         * x: vector x / quaternion x
         * y: vector y / quaternion y
         * z: vector z / quaternion z
         * w: vector w / quaternion w
         */
        key_frames(CLSize.cl_float4),

        /**
         * value: key frame timestamp
         */
        frame_times(CLSize.cl_double),

        /**
         * x: position channel start index
         * y: position channel end index
         */
        bone_pos_channel_tables(CLSize.cl_int2),

        /**
         * x: rotation channel start index
         * y: rotation channel end index
         */
        bone_rot_channel_tables(CLSize.cl_int2),

        /**
         * x: scaling channel start index
         * y: scaling channel end index
         */
        bone_scl_channel_tables(CLSize.cl_int2),

        /**
         * x: bone channel start index
         * y: bone channel end index
         */
        bone_channel_tables(CLSize.cl_int2),

        /**
         * x: animation duration
         * y: ticks per second (FPS)
         */
        animation_timings(CLSize.cl_double2),

        /**
         * value: animation timing index
         */
        animation_timing_indices(CLSize.cl_int),

        /*
        Points
         */

        /**
         * x: current x position
         * y: current y position
         * z: previous x position
         * w: previous y position
         */
        points(CLSize.cl_float4),

        /**
         * value: reaction count
         */
        point_reactions(CLSize.cl_int), // todo: isolate to physics simulation

        /**
         * value: reaction buffer offset
         */
        point_offsets(CLSize.cl_int), // todo: isolate to physics simulation

        /**
         * value: antigravity magnitude
         */
        point_anti_gravity(CLSize.cl_float),

        /**
         * x: reference vertex index
         * y: hull index
         * z: vertex flags (bit field)
         * w: (unused)
         */
        point_vertex_tables(CLSize.cl_int4),

        /**
         * x: bone 1 instance id
         * y: bone 2 instance id
         * z: bone 3 instance id
         * w: bone 4 instance id
         */
        point_bone_tables(CLSize.cl_int4),

        /*
        Edges
         */

        /**
         * x: point 1 index
         * y: point 2 index
         * z: distance constraint
         * w: edge flags (bit-field)
         * note: x, y, and w values are cast to int during use
         */
        edges(CLSize.cl_float4),

        /*
        Hulls
         */

        /**
         * x: current x position
         * y: current y position
         * z: scale x
         * w: scale y
         */
        hulls(CLSize.cl_float4),

        /**
         * value: reference mesh id
         */
        hull_mesh_ids(CLSize.cl_int),

        /**
         * x: initial reference angle
         * y: current rotation
         */
        hull_rotation(CLSize.cl_float2),

        /**
         * x: start point index
         * y: end point index
         * z: start edge index
         * w: end edge index
         */
        hull_element_tables(CLSize.cl_int4),

        /**
         * x: hull flags (bit-field)
         * y: armature id
         * z: start bone
         * w: end bone
         */
        hull_flags(CLSize.cl_int4),

        /**
         * x: corner x position
         * y: corner y position
         * z: width
         * w: height
         */
        aabb(CLSize.cl_float4),

        /**
         * x: minimum x key index
         * y: maximum x key index
         * z: minimum y key index
         * w: maximum y key index
         */
        aabb_index(CLSize.cl_int4),

        /**
         * x: key bank offset
         * y: key bank size
         */
        aabb_key_table(CLSize.cl_int2),

        /*
        Bones
         */

        /**
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
        bone_instances(CLSize.cl_float16),

        /**
         * x: bone inverse bind pose index (mesh-space)
         * y: bone bind pose index (model space)
         */
        bone_index_tables(CLSize.cl_int2),


        /**
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
        armatures_bones(CLSize.cl_float16),

        /**
         * x: bind pose reference id
         * y: armature bone parent id
         */
        bone_bind_tables(CLSize.cl_int2),

        /*
        Armatures
         */

        /**
         * x: current x position
         * y: current y position
         * z: previous x position
         * w: previous y position
         */
        armatures(CLSize.cl_float4),

        /**
         * x: root hull index
         * y: model id
         * z: armature flags (bit-field)
         * w: model transform index
         */
        armature_flags(CLSize.cl_int4),

        /**
         * x: current x acceleration
         * y: current y acceleration
         */
        armature_accel(CLSize.cl_float2),

        /**
         * value: mass of the armature
         */
        armature_mass(CLSize.cl_float),

        /**
         * value: the currently selected animation index
         */
        armature_animation_indices(CLSize.cl_int),

        /**
         * value: the last rendered timestamp
         */
        armature_animation_elapsed(CLSize.cl_double),

        /**
         * x: start hull index
         * y: end hull index
         * z: start bone anim index
         * w: end bone anim index
         */
        armature_hull_table(CLSize.cl_int4),

        /*
        Buffer Compaction
         */

        /**
         * During the armature deletion process, these buffers are written to, and store the number of
         * positions that the corresponding values must shift left within their own buffers when the
         * buffer compaction step is reached. Each index is aligned with the corresponding data type
         * that will be shifted. I.e. every bone in the bone buffer has a corresponding entry in the
         * bone shift buffer. Points, edges, and hulls work the same way.
         */
        bone_shift(CLSize.cl_int), // todo: isolate to core memory class
        point_shift(CLSize.cl_int), // todo: isolate to core memory class
        edge_shift(CLSize.cl_int), // todo: isolate to core memory class
        hull_shift(CLSize.cl_int), // todo: isolate to core memory class
        bone_bind_shift(CLSize.cl_int), // todo: isolate to core memory class

        ;

        public GPUMemory memory;
        final int size;
        int length;

        Buffer(int valueSize)
        {
            size = valueSize;
        }

        public void init(int buffer_length)
        {
            this.length = buffer_length * size;
            var mem = cl_new_buffer(this.length);
            this.memory = new GPUMemory(mem);
            clear();
        }

        public void clear()
        {
            cl_zero_buffer(this.memory.pointer(), this.length);
        }
    }

    //#endregion

    //#region Init Methods

    private static long init_device()
    {
        // TODO: may need some updates for cases where there's more than one possible device

        // The platform, device type and device number
        // that will be used
        long deviceType = CL_DEVICE_TYPE_GPU;

        // Obtain the number of platforms
        int[] numPlatformsArray = new int[1];
        clGetPlatformIDs(null, numPlatformsArray);
        int numPlatforms = numPlatformsArray[0];

        // Obtain a platform ID
        var platform_buffer = BufferUtils.createPointerBuffer(numPlatforms);
        clGetPlatformIDs(platform_buffer, (IntBuffer) null);
        var platform = platform_buffer.get();

        // Obtain the number of devices for the platform
        int[] numDevicesArray = new int[1];
        clGetDeviceIDs(platform, deviceType, null, numDevicesArray);
        int numDevices = numDevicesArray[0];

        // Obtain a device ID
        var device_buffer = BufferUtils.createPointerBuffer(numDevices);
        clGetDeviceIDs(platform, deviceType, device_buffer, (IntBuffer) null);

        long device = device_buffer.get();

        var dc = wglGetCurrentDC();
        var ctx = wglGetCurrentContext();

        // todo: the above code is windows specific add linux code path,
        //  should look something like this:
        // var ctx = glXGetCurrentContext();
        // var dc = glXGetCurrentDrawable(); OR glfwGetX11Display();
        // contextProperties.addProperty(CL_GLX_DISPLAY_KHR, dc);

        // Create a context for the selected device
        var ctxProps = BufferUtils.createPointerBuffer(7);
        ctxProps.put(CL_CONTEXT_PLATFORM)
            .put(platform)
            .put(CL_GL_CONTEXT_KHR)
            .put(ctx)
            .put(CL_WGL_HDC_KHR)
            .put(dc)
            .put(0L)
            .flip();

        context_ptr = clCreateContext(ctxProps,
            device, null, 0L, null);

        // Create a command-queue for the selected device
        command_queue_ptr = clCreateCommandQueue(context_ptr,
            device, 0, (IntBuffer) null);

        return device;
    }

    private static void init_memory(int max_hulls, int max_points)
    {
        atomic_counter_ptr = cl_new_pinned_int();
        // todo: there should be more granularity than just max hulls and points. There should be
        //  limits on armatures and other data types.

        Buffer.armature_accel.init(max_hulls);
        Buffer.armature_mass.init(max_hulls);
        Buffer.armature_animation_indices.init(max_hulls);
        Buffer.armature_animation_elapsed.init(max_hulls);
        Buffer.hull_rotation.init(max_hulls);
        Buffer.hull_element_tables.init(max_hulls);
        Buffer.hull_flags.init(max_hulls);
        Buffer.aabb_index.init(max_hulls);
        Buffer.aabb_key_table.init(max_hulls);
        Buffer.hulls.init(max_hulls);
        Buffer.hull_mesh_ids.init(max_hulls);
        Buffer.mesh_references.init(max_hulls);
        Buffer.mesh_faces.init(max_hulls);
        Buffer.aabb.init(max_hulls);
        Buffer.points.init(max_points);
        Buffer.point_reactions.init(max_points);
        Buffer.point_offsets.init(max_points);
        Buffer.point_anti_gravity.init(max_points);
        Buffer.edges.init(max_points);
        Buffer.point_vertex_tables.init(max_points);
        Buffer.point_bone_tables.init(max_points);
        Buffer.vertex_references.init(max_points);
        Buffer.vertex_weights.init(max_points);
        Buffer.texture_uvs.init(max_points);
        Buffer.uv_tables.init(max_points);
        Buffer.bone_bind_poses.init(max_hulls);
        Buffer.bone_bind_parents.init(max_hulls);
        Buffer.bone_references.init(max_points);
        Buffer.bone_instances.init(max_points);
        Buffer.bone_index_tables.init(max_points);
        Buffer.bone_bind_tables.init(max_points);
        Buffer.model_transforms.init(max_points);
        Buffer.armatures.init(max_points);
        Buffer.armature_flags.init(max_points);
        Buffer.armatures_bones.init(max_points);
        Buffer.armature_hull_table.init(max_hulls);
        Buffer.key_frames.init(max_points);
        Buffer.frame_times.init(max_points);
        Buffer.bone_pos_channel_tables.init(max_points);
        Buffer.bone_rot_channel_tables.init(max_points);
        Buffer.bone_scl_channel_tables.init(max_points);
        Buffer.bone_channel_tables.init(max_points);
        Buffer.animation_timings.init(max_points);
        Buffer.animation_timing_indices.init(max_points);
        Buffer.bone_shift.init(max_points);
        Buffer.point_shift.init(max_points);
        Buffer.edge_shift.init(max_points);
        Buffer.hull_shift.init(max_hulls);
        Buffer.bone_bind_shift.init(max_hulls);

        core_memory = new GPUCoreMemory();

        int total = Buffer.hulls.length
            + Buffer.hull_mesh_ids.length
            + Buffer.armature_accel.length
            + Buffer.armature_mass.length
            + Buffer.armature_animation_indices.length
            + Buffer.armature_animation_elapsed.length
            + Buffer.hull_rotation.length
            + Buffer.hull_element_tables.length
            + Buffer.hull_flags.length
            + Buffer.mesh_references.length
            + Buffer.mesh_faces.length
            + Buffer.aabb.length
            + Buffer.aabb_index.length
            + Buffer.aabb_key_table.length
            + Buffer.points.length
            + Buffer.point_reactions.length
            + Buffer.point_offsets.length
            + Buffer.point_anti_gravity.length
            + Buffer.edges.length
            + Buffer.point_vertex_tables.length
            + Buffer.point_bone_tables.length
            + Buffer.vertex_references.length
            + Buffer.vertex_weights.length
            + Buffer.texture_uvs.length
            + Buffer.uv_tables.length
            + Buffer.bone_bind_poses.length
            + Buffer.bone_bind_parents.length
            + Buffer.bone_references.length
            + Buffer.bone_instances.length
            + Buffer.bone_index_tables.length
            + Buffer.bone_bind_tables.length
            + Buffer.model_transforms.length
            + Buffer.armatures.length
            + Buffer.armature_flags.length
            + Buffer.armatures_bones.length
            + Buffer.armature_hull_table.length
            + Buffer.key_frames.length
            + Buffer.frame_times.length
            + Buffer.bone_pos_channel_tables.length
            + Buffer.bone_rot_channel_tables.length
            + Buffer.bone_scl_channel_tables.length
            + Buffer.bone_channel_tables.length
            + Buffer.animation_timings.length
            + Buffer.animation_timing_indices.length
            + Buffer.bone_shift.length
            + Buffer.point_shift.length
            + Buffer.edge_shift.length
            + Buffer.hull_shift.length
            + Buffer.bone_bind_shift.length;

        System.out.println("---------------------------- BUFFERS ----------------------------");
        System.out.println("points               : " + Buffer.points.length);
        System.out.println("edges                : " + Buffer.edges.length);
        System.out.println("hulls                : " + Buffer.hulls.length);
        System.out.println("hull mesh ids        : " + Buffer.hull_mesh_ids.length);
        System.out.println("acceleration         : " + Buffer.armature_accel.length);
        System.out.println("mass                 : " + Buffer.armature_mass.length);
        System.out.println("armature anim index  : " + Buffer.armature_animation_indices.length);
        System.out.println("armature anim times  : " + Buffer.armature_animation_elapsed.length);
        System.out.println("rotation             : " + Buffer.hull_rotation.length);
        System.out.println("element table        : " + Buffer.hull_element_tables.length);
        System.out.println("hull flags           : " + Buffer.hull_flags.length);
        System.out.println("mesh references      : " + Buffer.mesh_references.length);
        System.out.println("mesh faces           : " + Buffer.mesh_faces.length);
        System.out.println("point reactions      : " + Buffer.point_reactions.length);
        System.out.println("point offsets        : " + Buffer.point_offsets.length);
        System.out.println("point anti-grav      : " + Buffer.point_anti_gravity.length);
        System.out.println("bounding box         : " + Buffer.aabb.length);
        System.out.println("spatial index        : " + Buffer.aabb_index.length);
        System.out.println("spatial key bank     : " + Buffer.aabb_key_table.length);
        System.out.println("point vertex tables  : " + Buffer.point_vertex_tables.length);
        System.out.println("point bone tables    : " + Buffer.point_bone_tables.length);
        System.out.println("vertex references    : " + Buffer.vertex_references.length);
        System.out.println("vertex weights       : " + Buffer.vertex_weights.length);
        System.out.println("texture uvs          : " + Buffer.texture_uvs.length);
        System.out.println("uv maps              : " + Buffer.uv_tables.length);
        System.out.println("bone bind poses      : " + Buffer.bone_bind_poses.length);
        System.out.println("bone bind parents    : " + Buffer.bone_bind_parents.length);
        System.out.println("bone references      : " + Buffer.bone_references.length);
        System.out.println("bone instances       : " + Buffer.bone_instances.length);
        System.out.println("bone index           : " + Buffer.bone_index_tables.length);
        System.out.println("bone bind indices    : " + Buffer.bone_bind_tables.length);
        System.out.println("model transforms     : " + Buffer.model_transforms.length);
        System.out.println("armatures            : " + Buffer.armatures.length);
        System.out.println("armature flags       : " + Buffer.armature_flags.length);
        System.out.println("armature bones       : " + Buffer.armatures_bones.length);
        System.out.println("hull tables          : " + Buffer.armature_hull_table.length);
        System.out.println("keyframes            : " + Buffer.key_frames.length);
        System.out.println("frame times          : " + Buffer.frame_times.length);
        System.out.println("position channels    : " + Buffer.bone_pos_channel_tables.length);
        System.out.println("rotation channels    : " + Buffer.bone_rot_channel_tables.length);
        System.out.println("scaling channels     : " + Buffer.bone_scl_channel_tables.length);
        System.out.println("bone channels        : " + Buffer.bone_channel_tables.length);
        System.out.println("animation timings    : " + Buffer.animation_timings.length);
        System.out.println("animation indices    : " + Buffer.animation_timing_indices.length);
        System.out.println("bone shift           : " + Buffer.bone_shift.length);
        System.out.println("point shift          : " + Buffer.point_shift.length);
        System.out.println("edge shift           : " + Buffer.edge_shift.length);
        System.out.println("hull shift           : " + Buffer.hull_shift.length);
        System.out.println("bone bind shift      : " + Buffer.bone_bind_shift.length);
        System.out.println("=====================================");
        System.out.println(" Total (Bytes)       : " + total);
        System.out.println("                  KB : " + ((float) total / 1024f));
        System.out.println("                  MB : " + ((float) total / 1024f / 1024f));
        System.out.println("                  GB : " + ((float) total / 1024f / 1024f / 1024f));
        System.out.println("---------------------------------------------------------------\n");
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
        // integer exclusive scan in-place

        long scan_int_array_single_ptr = Program.scan_int_array.gpu.kernel_ptr(Kernel.scan_int_single_block);
        long scan_int_array_multi_ptr = Program.scan_int_array.gpu.kernel_ptr(Kernel.scan_int_multi_block);
        long scan_int_array_comp_ptr = Program.scan_int_array.gpu.kernel_ptr(Kernel.complete_int_multi_block);
        scan_int_single_block_k = new ScanIntSingleBlock_k(command_queue_ptr, scan_int_array_single_ptr);
        scan_int_multi_block_k = new ScanIntMultiBlock_k(command_queue_ptr, scan_int_array_multi_ptr);
        complete_int_multi_block_k = new CompleteIntMultiBlock_k(command_queue_ptr, scan_int_array_comp_ptr);

        // 2D vector integer exclusive scan in-place

        long scan_int2_array_single_ptr = Program.scan_int2_array.gpu.kernel_ptr(Kernel.scan_int2_single_block);
        long scan_int2_array_multi_ptr = Program.scan_int2_array.gpu.kernel_ptr(Kernel.scan_int2_multi_block);
        long scan_int2_array_comp_ptr = Program.scan_int2_array.gpu.kernel_ptr(Kernel.complete_int2_multi_block);
        scan_int2_single_block_k = new ScanInt2SingleBlock_k(command_queue_ptr, scan_int2_array_single_ptr);
        scan_int2_multi_block_k = new ScanInt2MultiBlock_k(command_queue_ptr, scan_int2_array_multi_ptr);
        complete_int2_multi_block_k = new CompleteInt2MultiBlock_k(command_queue_ptr, scan_int2_array_comp_ptr);

        // 4D vector integer exclusive scan in-place

        long scan_int4_array_single_ptr = Program.scan_int4_array.gpu.kernel_ptr(Kernel.scan_int4_single_block);
        long scan_int4_array_multi_ptr = Program.scan_int4_array.gpu.kernel_ptr(Kernel.scan_int4_multi_block);
        long scan_int4_array_comp_ptr = Program.scan_int4_array.gpu.kernel_ptr(Kernel.complete_int4_multi_block);
        scan_int4_single_block_k = new ScanInt4SingleBlock_k(command_queue_ptr, scan_int4_array_single_ptr);
        scan_int4_multi_block_k = new ScanInt4MultiBlock_k(command_queue_ptr, scan_int4_array_multi_ptr);
        complete_int4_multi_block_k = new CompleteInt4MultiBlock_k(command_queue_ptr, scan_int4_array_comp_ptr);

        // integer exclusive scan to output buffer

        long scan_int_array_out_single_ptr = Program.scan_int_array_out.gpu.kernel_ptr(Kernel.scan_int_single_block_out);
        long scan_int_array_out_multi_ptr = Program.scan_int_array_out.gpu.kernel_ptr(Kernel.scan_int_multi_block_out);
        long scan_int_array_out_comp_ptr = Program.scan_int_array_out.gpu.kernel_ptr(Kernel.complete_int_multi_block_out);
        scan_int_single_block_out_k = new ScanIntSingleBlockOut_k(command_queue_ptr, scan_int_array_out_single_ptr);
        scan_int_multi_block_out_k = new ScanIntMultiBlockOut_k(command_queue_ptr, scan_int_array_out_multi_ptr);
        complete_int_multi_block_out_k = new CompleteIntMultiBlockOut_k(command_queue_ptr, scan_int_array_out_comp_ptr);

        // Open GL interop

        long root_hull_filter_ptr = Program.root_hull_filter.gpu.kernel_ptr(Kernel.root_hull_filter);
        root_hull_filter_k = new RootHullFilter_k(command_queue_ptr, root_hull_filter_ptr)
            .mem_arg(RootHullFilter_k.Args.armature_flags, Buffer.armature_flags.memory);

        long root_hull_count_ptr = Program.root_hull_filter.gpu.kernel_ptr(Kernel.root_hull_count);
        root_hull_count_k = new RootHullCount_k(command_queue_ptr, root_hull_count_ptr)
            .mem_arg(RootHullCount_k.Args.armature_flags, Buffer.armature_flags.memory);
    }

    //#endregion

    //#region Utility Methods

    public static void cl_read_buffer(long src_ptr, int[] dst)
    {
        clEnqueueReadBuffer(command_queue_ptr,
            src_ptr,
            true,
            0,
            dst,
            null,
            null);
    }

    public static long cl_new_buffer(long size)
    {
        return clCreateBuffer(context_ptr, FLAGS_WRITE_GPU, size, null);
    }

    public static long cl_new_int_arg_buffer(int[] src)
    {
        return clCreateBuffer(context_ptr, FLAGS_WRITE_CPU_COPY, src, null);
    }

    public static long cl_new_cpu_copy_buffer(float[] src)
    {
        return clCreateBuffer(context_ptr, FLAGS_READ_CPU_COPY, src, null);
    }

    public static void cl_zero_buffer(long buffer_ptr, long buffer_size)
    {
        clEnqueueFillBuffer(command_queue_ptr,
            buffer_ptr,
            ZERO_PATTERN_BUFFER,
            0,
            buffer_size,
            null,
            null
            );
    }

    public static long cl_new_pinned_buffer(long size)
    {
        long flags = CL_MEM_HOST_READ_ONLY | CL_MEM_ALLOC_HOST_PTR;
        return clCreateBuffer(context_ptr, flags, size, null);
    }

    public static int[] cl_read_pinned_int_buffer(long pinned_ptr, long size, int count)
    {
        var out = clEnqueueMapBuffer(command_queue_ptr,
            pinned_ptr,
            true,
            CL_MAP_READ,
            0,
            size,
            null,
            null,
            (IntBuffer) null,
            null);

        assert out != null;
        int[] xa = new int[count];
        var ib = out.order(ByteOrder.LITTLE_ENDIAN).asIntBuffer();
        for (int i = 0; i < count; i++)
        {
            xa[i] = ib.get(i);
        }
        clEnqueueUnmapMemObject(command_queue_ptr, pinned_ptr, out, null, null);
        return xa;
    }

    public static float[] cl_read_pinned_float_buffer(long pinned_ptr, long size, int count)
    {
        var out = clEnqueueMapBuffer(command_queue_ptr,
            pinned_ptr,
            true,
            CL_MAP_READ,
            0,
            size,
            null,
            null,
            (IntBuffer) null,
            null);

        assert out != null;
        float[] xa = new float[count];
        var ib = out.order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer();
        for (int i = 0; i < count; i++)
        {
            xa[i] = ib.get(i);
        }
        clEnqueueUnmapMemObject(command_queue_ptr, pinned_ptr, out, null, null);
        return xa;
    }

    public static long cl_new_pinned_int()
    {
        long flags = CL_MEM_HOST_READ_ONLY | CL_MEM_ALLOC_HOST_PTR;
        return clCreateBuffer(context_ptr, flags, CLSize.cl_int, null);
    }

    public static int cl_read_pinned_int(long pinned_ptr)
    {
        var out = clEnqueueMapBuffer(command_queue_ptr,
            pinned_ptr,
            true,
            CL_MAP_READ,
            0,
            CLSize.cl_int,
            null,
            null,
            (IntBuffer) null,
            null);

        assert out != null;
        int result = out.order(ByteOrder.LITTLE_ENDIAN).asIntBuffer().get(0);
        clEnqueueUnmapMemObject(command_queue_ptr, pinned_ptr, out, null, null);
        return result;
    }

    public static int work_group_count(int n)
    {
        return (int) Math.ceil((float) n / (float) max_scan_block_size);
    }

    //#endregion

    //#region GL Interop

    public static long share_memory(int vboID)
    {
        return clCreateFromGLBuffer(context_ptr, FLAGS_WRITE_GPU, vboID, (IntBuffer) null);
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
        cl_zero_buffer(atomic_counter_ptr, CLSize.cl_int);

        root_hull_count_k
            .ptr_arg(RootHullCount_k.Args.counter, atomic_counter_ptr)
            .set_arg(RootHullCount_k.Args.model_id, model_id)
            .call(arg_long(GPGPU.core_memory.next_armature()));

        int final_count = cl_read_pinned_int(atomic_counter_ptr);

        if (final_count == 0)
        {
            return new HullIndexData(-1, final_count);
        }

        long final_buffer_size = (long) CLSize.cl_int * final_count;
        var hulls_out = cl_new_buffer(final_buffer_size);

        cl_zero_buffer(atomic_counter_ptr, CLSize.cl_int);

        root_hull_filter_k
            .ptr_arg(RootHullFilter_k.Args.hulls_out, hulls_out)
            .ptr_arg(RootHullFilter_k.Args.counter, atomic_counter_ptr)
            .set_arg(RootHullFilter_k.Args.model_id, model_id)
            .call(arg_long(GPGPU.core_memory.next_armature()));

        return new HullIndexData(hulls_out, final_count);
    }

    //#endregion

    //#region Exclusive scan variants

    public static void scan_int(long data_ptr, int n)
    {
        int k = work_group_count(n);
        if (k == 1)
        {
            scan_single_block_int(data_ptr, n);
        }
        else
        {
            scan_multi_block_int(data_ptr, n, k);
        }
    }

    public static void scan_int2(long data_ptr, int n)
    {
        int k = work_group_count(n);
        if (k == 1)
        {
            scan_single_block_int2(data_ptr, n);
        }
        else
        {
            scan_multi_block_int2(data_ptr, n, k);
        }
    }

    public static void scan_int4(long data_ptr, int n)
    {
        int k = work_group_count(n);
        if (k == 1)
        {
            scan_single_block_int4(data_ptr, n);
        }
        else
        {
            scan_multi_block_int4(data_ptr, n, k);
        }
    }

    public static void scan_int_out(long data_ptr, long o_data_ptr, int n)
    {
        int k = work_group_count(n);
        if (k == 1)
        {
            scan_single_block_int_out(data_ptr, o_data_ptr, n);
        }
        else
        {
            scan_multi_block_int_out(data_ptr, o_data_ptr, n, k);
        }
    }

    private static void scan_single_block_int(long data_ptr, int n)
    {
        long local_buffer_size = CLSize.cl_int * max_scan_block_size;

        scan_int_single_block_k
            .ptr_arg(ScanIntSingleBlock_k.Args.data, data_ptr)
            .loc_arg(ScanIntSingleBlock_k.Args.buffer, local_buffer_size)
            .set_arg(ScanIntSingleBlock_k.Args.n, n)
            .call(local_work_default, local_work_default);
    }

    private static void scan_multi_block_int(long data_ptr, int n, int k)
    {
        long local_buffer_size = CLSize.cl_int * max_scan_block_size;
        long gx = k * max_scan_block_size;
        long[] global_work_size = arg_long(gx);
        int part_size = k * 2;
        long part_buf_size = ((long) CLSize.cl_int * ((long) part_size));

        var part_data = cl_new_buffer(part_buf_size);

        scan_int_multi_block_k
            .ptr_arg(ScanIntMultiBlock_k.Args.data, data_ptr)
            .loc_arg(ScanIntMultiBlock_k.Args.buffer, local_buffer_size)
            .ptr_arg(ScanIntMultiBlock_k.Args.part, part_data)
            .set_arg(ScanIntMultiBlock_k.Args.n, n)
            .call(global_work_size, local_work_default);

        scan_int(part_data, part_size);

        complete_int_multi_block_k
            .ptr_arg(CompleteIntMultiBlock_k.Args.data, data_ptr)
            .loc_arg(CompleteIntMultiBlock_k.Args.buffer, local_buffer_size)
            .ptr_arg(CompleteIntMultiBlock_k.Args.part, part_data)
            .set_arg(CompleteIntMultiBlock_k.Args.n, n)
            .call(global_work_size, local_work_default);

        cl_release_buffer(part_data);
    }

    private static void scan_single_block_int2(long data_ptr, int n)
    {
        long local_buffer_size = CLSize.cl_int2 * max_scan_block_size;

       scan_int2_single_block_k
            .ptr_arg(ScanInt2SingleBlock_k.Args.data, data_ptr)
            .loc_arg(ScanInt2SingleBlock_k.Args.buffer, local_buffer_size)
            .set_arg(ScanInt2SingleBlock_k.Args.n, n)
            .call(local_work_default, local_work_default);
    }

    private static void scan_multi_block_int2(long data_ptr, int n, int k)
    {
        long local_buffer_size = CLSize.cl_int2 * max_scan_block_size;
        long gx = k * max_scan_block_size;
        long[] global_work_size = arg_long(gx);
        int part_size = k * 2;
        long part_buf_size = ((long) CLSize.cl_int2 * ((long) part_size));

        var part_data = cl_new_buffer(part_buf_size);

        scan_int2_multi_block_k
            .ptr_arg(ScanInt2MultiBlock_k.Args.data, data_ptr)
            .loc_arg(ScanInt2MultiBlock_k.Args.buffer, local_buffer_size)
            .ptr_arg(ScanInt2MultiBlock_k.Args.part, part_data)
            .set_arg(ScanInt2MultiBlock_k.Args.n, n)
            .call(global_work_size, local_work_default);

        scan_int2(part_data, part_size);

        complete_int2_multi_block_k
            .ptr_arg(CompleteInt2MultiBlock_k.Args.data, data_ptr)
            .loc_arg(CompleteInt2MultiBlock_k.Args.buffer, local_buffer_size)
            .ptr_arg(CompleteInt2MultiBlock_k.Args.part, part_data)
            .set_arg(CompleteInt2MultiBlock_k.Args.n, n)
            .call(global_work_size, local_work_default);

        cl_release_buffer(part_data);
    }

    private static void scan_single_block_int4(long data_ptr, int n)
    {
        long local_buffer_size = CLSize.cl_int4 * max_scan_block_size;

        scan_int4_single_block_k
            .ptr_arg(ScanInt4SingleBlock_k.Args.data, data_ptr)
            .loc_arg(ScanInt4SingleBlock_k.Args.buffer, local_buffer_size)
            .set_arg(ScanInt4SingleBlock_k.Args.n, n)
            .call(local_work_default, local_work_default);
    }

    private static void scan_multi_block_int4(long data_ptr, int n, int k)
    {
        long local_buffer_size = CLSize.cl_int4 * max_scan_block_size;
        long gx = k * max_scan_block_size;
        long[] global_work_size = arg_long(gx);
        int part_size = k * 2;
        long part_buf_size = ((long) CLSize.cl_int4 * ((long) part_size));

        var part_data = cl_new_buffer(part_buf_size);

        scan_int4_multi_block_k
            .ptr_arg(ScanInt4MultiBlock_k.Args.data, data_ptr)
            .loc_arg(ScanInt4MultiBlock_k.Args.buffer, local_buffer_size)
            .ptr_arg(ScanInt4MultiBlock_k.Args.part, part_data)
            .set_arg(ScanInt4MultiBlock_k.Args.n, n)
            .call(global_work_size, local_work_default);

        scan_int4(part_data, part_size);

        complete_int4_multi_block_k
            .ptr_arg(CompleteInt4MultiBlock_k.Args.data, data_ptr)
            .loc_arg(CompleteInt4MultiBlock_k.Args.buffer, local_buffer_size)
            .ptr_arg(CompleteInt4MultiBlock_k.Args.part, part_data)
            .set_arg(CompleteInt4MultiBlock_k.Args.n, n)
            .call(global_work_size, local_work_default);

        cl_release_buffer(part_data);
    }

    private static void scan_single_block_int_out(long data_ptr, long o_data_ptr, int n)
    {
        long local_buffer_size = CLSize.cl_int * max_scan_block_size;

        scan_int_single_block_out_k
            .ptr_arg(ScanIntSingleBlockOut_k.Args.input, data_ptr)
            .ptr_arg(ScanIntSingleBlockOut_k.Args.output, o_data_ptr)
            .loc_arg(ScanIntSingleBlockOut_k.Args.buffer, local_buffer_size)
            .set_arg(ScanIntSingleBlockOut_k.Args.n, n)
            .call(local_work_default, local_work_default);
    }

    private static void scan_multi_block_int_out(long data_ptr, long o_data_ptr, int n, int k)
    {
        long local_buffer_size = CLSize.cl_int * max_scan_block_size;
        long gx = k * max_scan_block_size;
        long[] global_work_size = arg_long(gx);
        int part_size = k * 2;
        long part_buf_size = ((long) CLSize.cl_int * ((long) part_size));
        var part_data = cl_new_buffer(part_buf_size);

        scan_int_multi_block_out_k
            .ptr_arg(ScanIntMultiBlockOut_k.Args.input, data_ptr)
            .ptr_arg(ScanIntMultiBlockOut_k.Args.output, o_data_ptr)
            .loc_arg(ScanIntMultiBlockOut_k.Args.buffer, local_buffer_size)
            .ptr_arg(ScanIntMultiBlockOut_k.Args.part, part_data)
            .set_arg(ScanIntMultiBlockOut_k.Args.n, n)
            .call(global_work_size, local_work_default);

        scan_int(part_data, part_size);

        complete_int_multi_block_out_k
            .ptr_arg(CompleteIntMultiBlockOut_k.Args.output, o_data_ptr)
            .loc_arg(CompleteIntMultiBlockOut_k.Args.buffer, local_buffer_size)
            .ptr_arg(CompleteIntMultiBlockOut_k.Args.part, part_data)
            .set_arg(CompleteIntMultiBlockOut_k.Args.n, n)
            .call(global_work_size, local_work_default);

        cl_release_buffer(part_data);
    }

    //#endregion

    //#region Misc. Public API

    public static long build_gpu_program(List<String> src_strings)
    {
        String[] src = src_strings.toArray(new String[]{});
        return CLUtils.cl_p(context_ptr, device_id_ptr, src);
    }

    public static long new_mutable_buffer(int[] src)
    {
        return clCreateBuffer(context_ptr, FLAGS_READ_CPU_COPY, src, null);
    }

    public static long new_empty_buffer(long size)
    {
        var new_buffer_ptr = cl_new_buffer(size);
        cl_zero_buffer(new_buffer_ptr, size);
        return new_buffer_ptr;
    }

    public static void cl_release_buffer(long mem_ptr)
    {
        clReleaseMemObject(mem_ptr);
    }

    public static void init(int max_hulls, int max_points)
    {
        device_id_ptr = init_device();

        System.out.println("-------- OPEN CL DEVICE -----------");
        System.out.println(getString(device_id_ptr, CL_DEVICE_VENDOR));
        System.out.println(getString(device_id_ptr, CL_DEVICE_NAME));
        System.out.println(getString(device_id_ptr, CL_DRIVER_VERSION));
        System.out.println("-----------------------------------\n");

        // At runtime, local buffers are used to perform prefix scan operations.
        // It is vital that the max scan block size does not exceed the maximum
        // local buffer size of the GPU. In order to ensure this doesn't happen,
        // the following logic halves the effective max workgroup size, if needed
        // to ensure that at runtime, the amount of local buffer storage requested
        // does not meet or exceed the local memory size.
        /*
         * The maximum size of a local buffer that can be used as a __local prefixed, GPU allocated
         * buffer within a kernel. Note that in practice, local memory buffers should be _less_ than
         * this value. Even though it is given a maximum, tests have shown that trying to allocate
         * exactly this amount can fail, likely due to some small amount of the local buffer being
         * used by the hardware either for individual arguments, or some other internal data.
         */
        long max_local_buffer_size = getSize(device_id_ptr, CL_DEVICE_LOCAL_MEM_SIZE);
        long current_max_group_size = getSize(device_id_ptr, CL_DEVICE_MAX_WORK_GROUP_SIZE);
        long current_max_block_size = current_max_group_size * 2;

        long int2_max = CLSize.cl_int2 * current_max_block_size;
        long int4_max = CLSize.cl_int4 * current_max_block_size;
        long size_cap = int2_max + int4_max;

        while (size_cap >= max_local_buffer_size)
        {
            current_max_group_size /= 2;
            current_max_block_size = current_max_group_size * 2;
            int2_max = CLSize.cl_int2 * current_max_block_size;
            int4_max = CLSize.cl_int4 * current_max_block_size;
            size_cap = int2_max + int4_max;
        }

        assert current_max_group_size > 0 : "Invalid Group Size";

        max_work_group_size = current_max_group_size;
        max_scan_block_size = current_max_block_size;
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
        core_memory.destroy();

        for (Buffer buffer : Buffer.values())
        {
            if (buffer.memory != null) buffer.memory.release();
        }

        for (Program program : Program.values())
        {
            if (program.gpu != null) program.gpu.destroy();
        }

        clReleaseCommandQueue(command_queue_ptr);
        clReleaseContext(context_ptr);
    }

    //#endregion
}
