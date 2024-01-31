package com.controllerface.bvge.cl;

import com.controllerface.bvge.cl.kernels.*;
import com.controllerface.bvge.cl.programs.*;
import com.controllerface.bvge.ecs.systems.physics.PhysicsBuffer;
import com.controllerface.bvge.ecs.systems.physics.UniformGrid;
import org.jocl.*;

import java.nio.ByteOrder;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;

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
     * Memory that is shared between Open CL and Open GL contexts.
     */
    private static final HashMap<Integer, cl_mem> shared_mem = new LinkedHashMap<>();
    //#endregion

    //#region Workgroup Variables

    /*
      These values are re-calculated at startup to match the user's hardware.
     */

    /**
     * The largest group of calculations that can be done in a single "warp" or "wave" of GPU processing.
     */
    private static int max_work_group_size = 0;

    /**
     * Used for the prefix scan kernels and their variants.
     */
    private static long max_scan_block_size = 0;

    /**
     * The max group size formatted as a single element array, making it simpler to use for Open Cl calls.
     */
    private static long[] local_work_default = arg_long(0);

    /**
     * This convenience array defines a work group size of 1, used primarily for setting up data buffers at
     * startup. Kernels of this size should be used sparingly, favor making bulk calls. However, there are
     * specific use cases where it makes sense to perform a singular operation on GPU memory.
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

    /**
     * There are several kernels that use an atomic counter, so rather than re-allocate a new
     * buffer for every call, this buffer is reused in all kernels that need a counter.
     */
    private static cl_mem counter_buffer = null;

    /**
     * This pointer to the counter buffer is used to pass the buffer to certain API calls.
     */
    private static Pointer counter_pointer = null;

    //#endregion

    //#region Program Objects

    public enum Program
    {
        aabb_collide(new AabbCollide()),
        animate_hulls(new AnimateHulls()),
        build_key_map(new BuildKeyMap()),
        generate_keys(new GenerateKeys()),
        gpu_crud(new GpuCrud()),
        integrate(new Integrate()),
        locate_in_bounds(new LocateInBounds()),
        mesh_query(new MeshQuery()),
        prepare_bones(new PrepareBones()),
        prepare_bounds(new PrepareBounds()),
        prepare_edges(new PrepareEdges()),
        prepare_points(new PreparePoints()),
        prepare_transforms(new PrepareTransforms()),
        resolve_constraints(new ResolveConstraints()),
        root_hull_filter(new RootHullFilter()),
        sat_collide(new SatCollide()),
        scan_deletes(new ScanDeletes()),
        scan_int2_array(new ScanInt2Array()),
        scan_int4_array(new ScanInt4Array()),
        scan_int_array(new ScanIntArray()),
        scan_int_array_out(new ScanIntArrayOut()),
        scan_key_bank(new ScanKeyBank()),
        scan_key_candidates(new ScanKeyCandidates()),

        ;

        public final GPUProgram gpu;

        Program(GPUProgram program)
        {
            this.gpu = program;
        }
    }

    //#endregion

    //#region Kernel Objects

    public enum Kernel
    {
        aabb_collide,
        animate_armatures,
        animate_hulls,
        animate_points,
        apply_reactions,
        build_key_map,
        calculate_batch_offsets,
        compact_armatures,
        compact_bones,
        compact_edges,
        compact_hulls,
        compact_points,
        complete_bounds_multi_block,
        complete_candidates_multi_block_out,
        complete_deletes_multi_block_out,
        complete_int2_multi_block,
        complete_int4_multi_block,
        complete_int_multi_block,
        complete_int_multi_block_out,
        count_candidates,
        count_mesh_batches,
        count_mesh_instances,
        create_armature,
        create_armature_bone,
        create_bone,
        create_bone_reference,
        create_bone_bind_pose,
        create_edge,
        create_hull,
        create_mesh_reference,
        create_mesh_face,
        create_point,
        create_texture_uv,
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
        prepare_points,
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
        scan_int2_multi_block,
        scan_int2_single_block,
        scan_int4_multi_block,
        scan_int4_single_block,
        scan_int_multi_block,
        scan_int_multi_block_out,
        scan_int_single_block,
        scan_int_single_block_out,
        sort_reactions,
        transfer_detail_data,
        transfer_render_data,
        update_accel,
        write_mesh_details,

        ;

        GPUKernel<?> kernel;

        public GPUKernel<?> set_kernel(GPUKernel<?> gpu_kernel)
        {
            this.kernel = gpu_kernel;
            return this.kernel;
        }

        public Kernel new_arg(Enum<?> val, long size)
        {
            kernel.new_arg(val.ordinal(), size);
            return this;
        }

        public Kernel new_arg(Enum<?> val, long size, Pointer pointer)
        {
            kernel.new_arg(val.ordinal(), size, pointer);
            return this;
        }


        public Kernel set_arg(Enum<?> val, Pointer pointer)
        {
            kernel.set_arg(val.ordinal(), pointer);
            return this;
        }

        public Kernel share_mem(cl_mem mem)
        {
            kernel.share_mem(mem);
            return this;
        }

        public void call(long[] global_work_size)
        {
            kernel.call(global_work_size);
        }

        public void call(long[] global_work_size, long[] local_work_size)
        {
            kernel.call(global_work_size, local_work_size);
        }

        public void call(long[] global_work_size, long[] local_work_size, long[] global_work_offset)
        {
            kernel.call(global_work_size, local_work_size, global_work_offset);
        }
    }

    //#endregion

    //#region Buffer Objects

    private enum Buffer
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
        vertex_references(Sizeof.cl_float2),

        /**
         * x: bone 1 weight
         * y: bone 2 weight
         * z: bone 3 weight
         * w: bone 4 weight
         */
        vertex_weights(Sizeof.cl_float4),

        /**
         * x: u coordinate
         * y: v coordinate
         */
        texture_uvs(Sizeof.cl_float2),

        /**
         * x: start UV index
         * y: end UV index
         */
        uv_tables(Sizeof.cl_int2),

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
        bone_references(Sizeof.cl_float16),

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
        bone_bind_poses(Sizeof.cl_float16),

        /**
         * value: reference index of the parent bone bind pose
         */
        bone_bind_parents(Sizeof.cl_int),

        /**
         * x: start vertex index
         * y: end vertex index
         * z: start face index
         * w: end face index
         */
        mesh_references(Sizeof.cl_int4),

        /**
         * x: vertex 1 index
         * y: vertex 2 index
         * z: vertex 3 index
         * w: parent reference mesh ID
         */
        mesh_faces(Sizeof.cl_int4),

        /*
        Points
         */

        /**
         * x: current x position
         * y: current y position
         * z: previous x position
         * w: previous y position
         */
        points(Sizeof.cl_float4),

        /**
         * value: reaction count
         */
        point_reactions(Sizeof.cl_int),

        /**
         * value: reaction buffer offset
         */
        point_offsets(Sizeof.cl_int),

        /**
         * value: antigravity magnitude
         */
        point_anti_gravity(Sizeof.cl_float),

        /**
         * x: reference vertex index
         * y: hull index
         * z: vertex flags (bit field)
         * w: (unused)
         */
        point_vertex_tables(Sizeof.cl_int4),

        /**
         * x: bone 1 instance id
         * y: bone 2 instance id
         * z: bone 3 instance id
         * w: bone 4 instance id
         */
        point_bone_tables(Sizeof.cl_int4),

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
        edges(Sizeof.cl_float4),

        /*
        Hulls
         */

        /**
         * x: current x position
         * y: current y position
         * z: scale x
         * w: scale y
         */
        hulls(Sizeof.cl_float4),

        /**
         * value: reference mesh id
         */
        hull_mesh_ids(Sizeof.cl_int),

        /**
         * x: initial reference angle
         * y: current rotation
         */
        hull_rotation(Sizeof.cl_float2),

        /**
         * x: start point index
         * y: end point index
         * z: start edge index
         * w: end edge index
         */
        hull_element_tables(Sizeof.cl_int4),

        /**
         * x: hull flags (bit-field)
         * y: armature id
         * z: start bone
         * w: end bone
         */
        hull_flags(Sizeof.cl_int4),

        /**
         * x: corner x position
         * y: corner y position
         * z: width
         * w: height
         */
        aabb(Sizeof.cl_float4),

        /**
         * x: minimum x key index
         * y: maximum x key index
         * z: minimum y key index
         * w: maximum y key index
         */
        aabb_index(Sizeof.cl_int4),

        /**
         * x: key bank offset
         * y: key bank size
         */
        aabb_key_table(Sizeof.cl_int2),

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
        bone_instances(Sizeof.cl_float16),

        /**
         * x: bone inverse bind pose index (mesh-space)
         * y: bone bind pose index (model space)
         */
        bone_index_tables(Sizeof.cl_int2),


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
        armatures_bones(Sizeof.cl_float16),

        /**
         * x: armature bone parent id
         * y: bind post reference id
         */
        bone_bind_tables(Sizeof.cl_int2),

        /*
        Armatures
         */

        /**
         * x: current x position
         * y: current y position
         * z: previous x position
         * w: previous y position
         */
        armatures(Sizeof.cl_float4),

        /**
         * x: root hull index
         * y: model id
         * z: armature flags (bit-field)
         * w: -unused-
         */
        armature_flags(Sizeof.cl_int4),

        /**
         * x: current x acceleration
         * y: current y acceleration
         */
        armature_accel(Sizeof.cl_float2),

        /**
         * value: mass of the armature
         */
        armature_mass(Sizeof.cl_float),

        /**
         * x: start hull index
         * y: end hull index
         * z: start bone anim index // todo: extend buffer for this
         * w: end bone anim index   // todo: extend buffer for this
         */
        armature_hull_table(Sizeof.cl_int4),

        // todo: add armature bone animation objects

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
        bone_shift(Sizeof.cl_int),
        point_shift(Sizeof.cl_int),
        edge_shift(Sizeof.cl_int),
        hull_shift(Sizeof.cl_int),
        bone_bind_shift(Sizeof.cl_int),

        ;

        GPUMemory memory;
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
            cl_zero_buffer(this.memory.memory(), this.length);
        }
    }

    //#endregion

    //#region Main Memory Access

    /**
     * The "Main Memory" interface
     * -
     * Calling code uses this object as a single access point to the core memory that
     * is involved in the game engine's operation. The memory being used is resident
     * on the GPU, so this class is essentially a direct interface into GPU memory.
     * This class keeps track of the number of "top-level" memory objects, which are
     * generally defined as any "logical object" that is composed of various primitive
     * values.
     * Typically, a method is exposed, that accepts as parameters, the various
     * components that describe an object. These methods in turn delegate the memory
     * access calls to the GPU APIs. In practice, this means that the actual buffers
     * that store the components of created objects may reside in separate memory
     * regions depending on the layout and type of the data being stored.
     * For example, consider an object that has two float components and one integer
     * component, a method to create this object may have a signature like this:
     * -
     *    foo(float x, float y, int i)
     * -
     * The GPU implementation may store all three of these values in one continuous
     * block of memory, or it may store the x and y components together, and the int
     * value in a separate memory segment, or even store all three in completely
     * different sections of memory. As such, this class cannot make too many
     * assumptions about the memory layout of all the various components. However,
     * each top-level object does have one known "memory width" that is used to keep
     * track of the number of objects of that type that are currently stored in
     * memory.
     * For these base objects, they will always be laid out in a continuous manner.
     * Just note that this width generally applies to one or a small number of "base"
     * properties of the object, and any extra properties or meta-data related to the
     * object will use separate, but index-aligned, memory space. The best existing
     * example of this concept are the HULL object type. The "main" value tracked for
     * these objects is represented on the GPU as a float4, with the first two values
     * (x,y) designating the world-space position of the hull center, and the second
     * two values (z,w) defining the width and height scaling of the hull. This base
     * set of values has a width of 4, so the hull index increments by 4 when each new
     * hull is created. Hulls however also have other data, for example an indexing
     * table that defines the start/end indices in the point and edge buffers of the
     * points and edges that are part of the hull. These values are stored in buffers
     * that align with the hull buffer, so that an index into one buffer can be used
     * interchangeably with the buffers that store all the other components of the
     * object, making access to a single "object" possible by indexing into all the
     * aligned arrays with the same index.
     */
    public static class Memory
    {
        public static class Width
        {
            public static final int ARMATURE = 4;
            public static final int BONE     = 16;
            public static final int EDGE     = 4;
            public static final int FACE     = 4;
            public static final int HULL     = 4;
            public static final int MESH     = 4;
            public static final int POINT    = 4;
            public static final int UV       = 2;
            public static final int VERTEX   = 2;
        }

        private static int hull_index          = 0;
        private static int point_index         = 0;
        private static int edge_index          = 0;
        private static int vertex_ref_index    = 0;
        private static int bone_bind_index     = 0;
        private static int bone_ref_index      = 0;
        private static int bone_index          = 0;
        private static int armature_bone_index = 0;
        private static int armature_index      = 0;
        private static int mesh_index          = 0;
        private static int face_index          = 0;
        private static int uv_index            = 0;

        public static int next_uv()
        {
            return uv_index / GPU.Memory.Width.UV;
        }

        public static int next_face()
        {
            return face_index / GPU.Memory.Width.FACE;
        }

        public static int next_mesh()
        {
            return mesh_index / GPU.Memory.Width.MESH;
        }

        public static int next_armature()
        {
            return armature_index / GPU.Memory.Width.ARMATURE;
        }

        public static int next_hull()
        {
            return hull_index / GPU.Memory.Width.HULL;
        }

        public static int next_point()
        {
            return point_index / GPU.Memory.Width.POINT;
        }

        public static int next_edge()
        {
            return edge_index / GPU.Memory.Width.EDGE;
        }

        public static int next_vertex_ref()
        {
            return vertex_ref_index / GPU.Memory.Width.VERTEX;
        }

        public static int next_bone_bind()
        {
            return bone_bind_index / GPU.Memory.Width.BONE;
        }

        public static int next_bone_ref()
        {
            return bone_ref_index / GPU.Memory.Width.BONE;
        }

        public static int next_bone()
        {
            return bone_index / GPU.Memory.Width.BONE;
        }

        public static int next_armature_bone()
        {
            return armature_bone_index / GPU.Memory.Width.BONE;
        }

        public static int new_texture_uv(float u, float v)
        {
            GPU.create_texture_uv(next_uv(), u, v);
            var idx = uv_index;
            uv_index += GPU.Memory.Width.UV;
            return idx / GPU.Memory.Width.UV;
        }

        public static int new_edge(int p1, int p2, float l, int flags)
        {
            GPU.create_edge(next_edge(), p1, p2, l, flags);
            var idx = edge_index;
            edge_index += GPU.Memory.Width.EDGE;
            return idx / GPU.Memory.Width.EDGE;
        }

        public static int new_point(float[] position, int[] vertex_table, int[] bone_ids)
        {
            var init_vert = new float[]{position[0], position[1], position[0], position[1]};
            GPU.create_point(next_point(), init_vert, vertex_table, bone_ids);
            var idx = point_index;
            point_index += GPU.Memory.Width.POINT;
            return idx / GPU.Memory.Width.POINT;
        }

        public static int new_hull(int mesh_id, float[] transform, float[] rotation, int[] table, int[] flags)
        {
            GPU.create_hull(next_hull(), mesh_id, transform, rotation, table, flags);
            var idx = hull_index;
            hull_index += GPU.Memory.Width.HULL;
            return idx / GPU.Memory.Width.HULL;
        }

        public static int new_mesh_reference(int[] mesh_ref_table)
        {
            GPU.create_mesh_reference(next_mesh(), mesh_ref_table);
            var idx = mesh_index;
            mesh_index += GPU.Memory.Width.MESH;
            return idx / GPU.Memory.Width.MESH;
        }

        public static int new_mesh_face(int[] face)
        {
            GPU.create_mesh_face(next_face(), face);
            var idx = face_index;
            face_index += GPU.Memory.Width.FACE;
            return idx / GPU.Memory.Width.FACE;
        }

        public static int new_armature(float x, float y, int[] table, int[] flags, float mass)
        {
            GPU.create_armature(next_armature(), x, y, table, flags, mass);
            var idx = armature_index;
            armature_index += GPU.Memory.Width.ARMATURE;
            return idx / GPU.Memory.Width.ARMATURE;
        }

        public static int new_vertex_reference(float x, float y, float[] weights, int[] uv_table)
        {
            GPU.create_vertex_reference(next_vertex_ref(), x, y, weights, uv_table);
            var idx = vertex_ref_index;
            vertex_ref_index += GPU.Memory.Width.VERTEX;
            return idx / GPU.Memory.Width.VERTEX;
        }

        public static int new_bone_bind_pose(int bind_parent, float[] bone_data)
        {
            GPU.create_bone_bind_pose(next_bone_bind(), bind_parent, bone_data);
            var idx = bone_bind_index;
            bone_bind_index += GPU.Memory.Width.BONE;
            return idx / GPU.Memory.Width.BONE;
        }

        public static int new_bone_reference(float[] bone_data)
        {
            GPU.create_bone_reference(next_bone_ref(), bone_data);
            var idx = bone_ref_index;
            bone_ref_index += GPU.Memory.Width.BONE;
            return idx / GPU.Memory.Width.BONE;
        }

        public static int new_bone(int[] offset_id, float[] bone_data)
        {
            GPU.create_bone(next_bone(), offset_id, bone_data);
            var idx = bone_index;
            bone_index += GPU.Memory.Width.BONE;
            return idx / GPU.Memory.Width.BONE;
        }

        public static int new_armature_bone(int[] bone_bind_table, float[] bone_data)
        {
            GPU.create_armature_bone(next_armature_bone(), bone_bind_table, bone_data);
            var idx = armature_bone_index;
            armature_bone_index += GPU.Memory.Width.BONE;
            return idx / GPU.Memory.Width.BONE;
        }

        public static void compact_buffers(int edge_shift,
                                           int bone_shift,
                                           int point_shift,
                                           int hull_shift,
                                           int armature_shift)
        {
            edge_index     -= (edge_shift * GPU.Memory.Width.EDGE);
            bone_index     -= (bone_shift * GPU.Memory.Width.BONE);
            point_index    -= (point_shift * GPU.Memory.Width.POINT);
            hull_index     -= (hull_shift * GPU.Memory.Width.HULL);
            armature_index -= (armature_shift * GPU.Memory.Width.ARMATURE);
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
        counter_buffer = cl_new_pinned_int();
        counter_pointer = Pointer.to(counter_buffer);
        // todo: there should be more granularity than just max hulls and points. There should be
        //  limits on armatures and other data types.

        Buffer.armature_accel.init(max_hulls);
        Buffer.armature_mass.init(max_hulls);
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
        Buffer.armatures.init(max_points);
        Buffer.armature_flags.init(max_points);
        Buffer.armatures_bones.init(max_points);
        Buffer.armature_hull_table.init(max_hulls);
        Buffer.bone_shift.init(max_points);
        Buffer.point_shift.init(max_points);
        Buffer.edge_shift.init(max_points);
        Buffer.hull_shift.init(max_hulls);
        Buffer.bone_bind_shift.init(max_hulls);

        int total = Buffer.hulls.length
            + Buffer.hull_mesh_ids.length
            + Buffer.armature_accel.length
            + Buffer.armature_mass.length
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
            + Buffer.armatures.length
            + Buffer.armature_flags.length
            + Buffer.armatures_bones.length
            + Buffer.armature_hull_table.length
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
        System.out.println("armatures            : " + Buffer.armatures.length);
        System.out.println("armature flags       : " + Buffer.armature_flags.length);
        System.out.println("armature bones       : " + Buffer.armatures_bones.length);
        System.out.println("hull tables          : " + Buffer.armature_hull_table.length);
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

        Kernel.scan_int_single_block.set_kernel(new ScanIntSingleBlock_k(command_queue));
        Kernel.scan_int_multi_block.set_kernel(new ScanIntMultiBlock_k(command_queue));
        Kernel.complete_int_multi_block.set_kernel(new CompleteIntMultiBlock_k(command_queue));

        // 2D vector integer exclusive scan in-place

        Kernel.scan_int2_single_block.set_kernel(new ScanInt2SingleBlock_k(command_queue));
        Kernel.scan_int2_multi_block.set_kernel(new ScanInt2MultiBlock_k(command_queue));
        Kernel.complete_int2_multi_block.set_kernel(new CompleteInt2MultiBlock_k(command_queue));

        // 4D vector integer exclusive scan in-place

        Kernel.scan_int4_single_block.set_kernel(new ScanInt4SingleBlock_k(command_queue));
        Kernel.scan_int4_multi_block.set_kernel(new ScanInt4MultiBlock_k(command_queue));
        Kernel.complete_int4_multi_block.set_kernel(new CompleteInt4MultiBlock_k(command_queue));

        // integer exclusive scan to output buffer

        Kernel.scan_int_single_block_out.set_kernel(new ScanIntSingleBlockOut_k(command_queue));
        Kernel.scan_int_multi_block_out.set_kernel(new ScanIntMultiBlockOut_k(command_queue));
        Kernel.complete_int_multi_block_out.set_kernel(new CompleteIntMultiBlockOut_k(command_queue));

        // collision candidate scan to output buffer

        Kernel.scan_candidates_single_block_out.set_kernel(new ScanCandidatesSingleBlockOut_k(command_queue));
        Kernel.scan_candidates_multi_block_out.set_kernel(new ScanCandidatesMultiBlockOut_k(command_queue));
        Kernel.complete_candidates_multi_block_out.set_kernel(new CompleteCandidatesMultiBlockOut_k(command_queue));

        // in-place uniform grid key bounds scan

        Kernel.scan_bounds_single_block.set_kernel(new ScanBoundsSingleBlock_k(command_queue));
        Kernel.scan_bounds_multi_block.set_kernel(new ScanBoundsMultiBlock_k(command_queue));
        Kernel.complete_bounds_multi_block.set_kernel(new CompleteBoundsMultiBlock_k(command_queue));

        // constraint solver

        Kernel.resolve_constraints.set_kernel(new ResolveConstraints_k(command_queue))
            .mem_arg(ResolveConstraints_k.Args.element_table, Buffer.hull_element_tables.memory)
            .mem_arg(ResolveConstraints_k.Args.bounds_bank_dat, Buffer.aabb_key_table.memory)
            .mem_arg(ResolveConstraints_k.Args.point, Buffer.points.memory)
            .mem_arg(ResolveConstraints_k.Args.edges, Buffer.edges.memory);

        // Open GL interop

        Kernel.prepare_bounds.set_kernel(new PrepareBounds_k(command_queue))
            .mem_arg(PrepareBounds_k.Args.bounds, Buffer.aabb.memory);

        Kernel.prepare_transforms.set_kernel(new PrepareTransforms_k(command_queue))
            .mem_arg(PrepareTransforms_k.Args.transforms, Buffer.hulls.memory)
            .mem_arg(PrepareTransforms_k.Args.hull_rotations, Buffer.hull_rotation.memory);

        Kernel.root_hull_count.set_kernel(new RootHullCount_k(command_queue))
            .mem_arg(RootHullCount_k.Args.armature_flags, Buffer.armature_flags.memory);

        Kernel.root_hull_filter.set_kernel(new RootHullFilter_k(command_queue))
            .mem_arg(RootHullFilter_k.Args.armature_flags, Buffer.armature_flags.memory);

        Kernel.prepare_points.set_kernel(new PreparePoints_k(command_queue))
            .mem_arg(PreparePoints_k.Args.points, Buffer.points.memory);

        Kernel.prepare_edges.set_kernel(new PrepareEdges_k(command_queue))
            .mem_arg(PrepareEdges_k.Args.points, Buffer.points.memory)
            .mem_arg(PrepareEdges_k.Args.edges, Buffer.edges.memory);

        Kernel.prepare_bones.set_kernel(new PrepareBones_k(command_queue))
            .mem_arg(PrepareBones_k.Args.bones, Buffer.bone_instances.memory)
            .mem_arg(PrepareBones_k.Args.bone_references, Buffer.bone_references.memory)
            .mem_arg(PrepareBones_k.Args.bone_index, Buffer.bone_index_tables.memory)
            .mem_arg(PrepareBones_k.Args.hulls, Buffer.hulls.memory)
            .mem_arg(PrepareBones_k.Args.armatures, Buffer.armatures.memory)
            .mem_arg(PrepareBones_k.Args.hull_flags, Buffer.hull_flags.memory);

        // narrow collision

        Kernel.sat_collide.set_kernel(new SatCollide_k(command_queue))
            .mem_arg(SatCollide_k.Args.hulls, Buffer.hulls.memory)
            .mem_arg(SatCollide_k.Args.element_tables, Buffer.hull_element_tables.memory)
            .mem_arg(SatCollide_k.Args.hull_flags, Buffer.hull_flags.memory)
            .mem_arg(SatCollide_k.Args.vertex_tables, Buffer.point_vertex_tables.memory)
            .mem_arg(SatCollide_k.Args.points, Buffer.points.memory)
            .mem_arg(SatCollide_k.Args.edges, Buffer.edges.memory)
            .mem_arg(SatCollide_k.Args.point_reactions, Buffer.point_reactions.memory)
            .mem_arg(SatCollide_k.Args.masses, Buffer.armature_mass.memory);

        Kernel.sort_reactions.set_kernel(new SortReactions_k(command_queue))
            .mem_arg(SortReactions_k.Args.point_reactions, Buffer.point_reactions.memory)
            .mem_arg(SortReactions_k.Args.point_offsets, Buffer.point_offsets.memory);

        Kernel.apply_reactions.set_kernel(new ApplyReactions_k(command_queue))
            .mem_arg(ApplyReactions_k.Args.points, Buffer.points.memory)
            .mem_arg(ApplyReactions_k.Args.anti_gravity, Buffer.point_anti_gravity.memory)
            .mem_arg(ApplyReactions_k.Args.point_reactions, Buffer.point_reactions.memory)
            .mem_arg(ApplyReactions_k.Args.point_offsets, Buffer.point_offsets.memory);

        Kernel.move_armatures.set_kernel(new MoveArmatures_k(command_queue))
            .mem_arg(MoveArmatures_k.Args.hulls, Buffer.hulls.memory)
            .mem_arg(MoveArmatures_k.Args.armatures, Buffer.armatures.memory)
            .mem_arg(MoveArmatures_k.Args.hull_tables, Buffer.armature_hull_table.memory)
            .mem_arg(MoveArmatures_k.Args.element_tables, Buffer.hull_element_tables.memory)
            .mem_arg(MoveArmatures_k.Args.hull_flags, Buffer.hull_flags.memory)
            .mem_arg(MoveArmatures_k.Args.points, Buffer.points.memory);

        // crud

        Kernel.create_point.set_kernel(new CreatePoint_k(command_queue))
            .mem_arg(CreatePoint_k.Args.points, Buffer.points.memory)
            .mem_arg(CreatePoint_k.Args.vertex_tables, Buffer.point_vertex_tables.memory)
            .mem_arg(CreatePoint_k.Args.bone_tables, Buffer.point_bone_tables.memory);

        Kernel.create_texture_uv.set_kernel(new CreateTextureUV_k(command_queue))
            .mem_arg(CreateTextureUV_k.Args.texture_uvs, Buffer.texture_uvs.memory);

        Kernel.create_edge.set_kernel(new CreateEdge_k(command_queue))
            .mem_arg(CreateEdge_k.Args.edges, Buffer.edges.memory);

        Kernel.create_vertex_reference.set_kernel(new CreateVertexRef_k(command_queue))
            .mem_arg(CreateVertexRef_k.Args.vertex_references, Buffer.vertex_references.memory)
            .mem_arg(CreateVertexRef_k.Args.vertex_weights, Buffer.vertex_weights.memory)
            .mem_arg(CreateVertexRef_k.Args.uv_tables, Buffer.uv_tables.memory);

        Kernel.create_bone_bind_pose.set_kernel(new CreateBoneBindPose_k(command_queue))
            .mem_arg(CreateBoneBindPose_k.Args.bone_bind_poses, Buffer.bone_bind_poses.memory)
            .mem_arg(CreateBoneBindPose_k.Args.bone_bind_parents, Buffer.bone_bind_parents.memory);

        Kernel.create_bone_reference.set_kernel(new CreateBoneRef_k(command_queue))
            .mem_arg(CreateBoneRef_k.Args.bone_references, Buffer.bone_references.memory);

        Kernel.create_armature.set_kernel(new CreateArmature_k(command_queue))
            .mem_arg(CreateArmature_k.Args.armatures, Buffer.armatures.memory)
            .mem_arg(CreateArmature_k.Args.armature_flags, Buffer.armature_flags.memory)
            .mem_arg(CreateArmature_k.Args.hull_tables, Buffer.armature_hull_table.memory)
            .mem_arg(CreateArmature_k.Args.armature_masses, Buffer.armature_mass.memory);

        Kernel.create_bone.set_kernel(new CreateBone_k(command_queue))
            .mem_arg(CreateBone_k.Args.bones, Buffer.bone_instances.memory)
            .mem_arg(CreateBone_k.Args.bone_ref_tables, Buffer.bone_index_tables.memory);

        Kernel.create_armature_bone.set_kernel(new CreateArmatureBone_k(command_queue))
            .mem_arg(CreateArmatureBone_k.Args.armature_bones, Buffer.armatures_bones.memory)
            .mem_arg(CreateArmatureBone_k.Args.bone_bind_tables, Buffer.bone_bind_tables.memory);

        Kernel.create_hull.set_kernel(new CreateHull_k(command_queue))
            .mem_arg(CreateHull_k.Args.hulls, Buffer.hulls.memory)
            .mem_arg(CreateHull_k.Args.hull_rotations, Buffer.hull_rotation.memory)
            .mem_arg(CreateHull_k.Args.element_tables, Buffer.hull_element_tables.memory)
            .mem_arg(CreateHull_k.Args.hull_flags, Buffer.hull_flags.memory)
            .mem_arg(CreateHull_k.Args.hull_mesh_ids, Buffer.hull_mesh_ids.memory);

        Kernel.create_mesh_reference.set_kernel(new CreateMeshReference_k(command_queue))
            .mem_arg(CreateMeshReference_k.Args.mesh_ref_tables, Buffer.mesh_references.memory);

        Kernel.create_mesh_face.set_kernel(new CreateMeshFace_k(command_queue))
            .mem_arg(CreateMeshFace_k.Args.mesh_faces, Buffer.mesh_faces.memory);

        Kernel.read_position.set_kernel(new ReadPosition_k(command_queue))
            .mem_arg(ReadPosition_k.Args.armatures, Buffer.armatures.memory);

        Kernel.update_accel.set_kernel(new UpdateAccel_k(command_queue))
            .mem_arg(UpdateAccel_k.Args.armature_accel, Buffer.armature_accel.memory);

        // object delete support

        Kernel.locate_out_of_bounds.set_kernel(new LocateOutOfBounds_k(command_queue))
            .mem_arg(LocateOutOfBounds_k.Args.hull_tables, Buffer.armature_hull_table.memory)
            .mem_arg(LocateOutOfBounds_k.Args.hull_flags, Buffer.hull_flags.memory)
            .mem_arg(LocateOutOfBounds_k.Args.armature_flags, Buffer.armature_flags.memory);

        Kernel.scan_deletes_single_block_out.set_kernel(new ScanDeletesSingleBlockOut_k(command_queue))
            .mem_arg(ScanDeletesSingleBlockOut_k.Args.armature_flags, Buffer.armature_flags.memory)
            .mem_arg(ScanDeletesSingleBlockOut_k.Args.hull_tables, Buffer.armature_hull_table.memory)
            .mem_arg(ScanDeletesSingleBlockOut_k.Args.element_tables, Buffer.hull_element_tables.memory)
            .mem_arg(ScanDeletesSingleBlockOut_k.Args.hull_flags, Buffer.hull_flags.memory);

        Kernel.scan_deletes_multi_block_out.set_kernel(new ScanDeletesMultiBlockOut_k(command_queue))
            .mem_arg(ScanDeletesMultiBlockOut_k.Args.armature_flags, Buffer.armature_flags.memory)
            .mem_arg(ScanDeletesMultiBlockOut_k.Args.hull_tables, Buffer.armature_hull_table.memory)
            .mem_arg(ScanDeletesMultiBlockOut_k.Args.element_tables, Buffer.hull_element_tables.memory)
            .mem_arg(ScanDeletesMultiBlockOut_k.Args.hull_flags, Buffer.hull_flags.memory);

        Kernel.complete_deletes_multi_block_out.set_kernel(new CompleteDeletesMultiBlockOut_k(command_queue))
            .mem_arg(CompleteDeletesMultiBlockOut_k.Args.armature_flags, Buffer.armature_flags.memory)
            .mem_arg(CompleteDeletesMultiBlockOut_k.Args.hull_tables, Buffer.armature_hull_table.memory)
            .mem_arg(CompleteDeletesMultiBlockOut_k.Args.element_tables, Buffer.hull_element_tables.memory)
            .mem_arg(CompleteDeletesMultiBlockOut_k.Args.hull_flags, Buffer.hull_flags.memory);

        // post-delete buffer compaction

        Kernel.compact_armatures.set_kernel(new CompactArmatures_k(command_queue))
            .mem_arg(CompactArmatures_k.Args.armatures, Buffer.armatures.memory)
            .mem_arg(CompactArmatures_k.Args.armature_accel, Buffer.armature_accel.memory)
            .mem_arg(CompactArmatures_k.Args.armature_flags, Buffer.armature_flags.memory)
            .mem_arg(CompactArmatures_k.Args.hull_tables, Buffer.armature_hull_table.memory)
            .mem_arg(CompactArmatures_k.Args.hulls, Buffer.hulls.memory)
            .mem_arg(CompactArmatures_k.Args.hull_flags, Buffer.hull_flags.memory)
            .mem_arg(CompactArmatures_k.Args.element_tables, Buffer.hull_element_tables.memory)
            .mem_arg(CompactArmatures_k.Args.points, Buffer.points.memory)
            .mem_arg(CompactArmatures_k.Args.vertex_tables, Buffer.point_vertex_tables.memory)
            .mem_arg(CompactArmatures_k.Args.bone_tables, Buffer.point_bone_tables.memory)
            .mem_arg(CompactArmatures_k.Args.edges, Buffer.edges.memory)
            .mem_arg(CompactArmatures_k.Args.bone_shift, Buffer.bone_shift.memory)
            .mem_arg(CompactArmatures_k.Args.point_shift, Buffer.point_shift.memory)
            .mem_arg(CompactArmatures_k.Args.edge_shift, Buffer.edge_shift.memory)
            .mem_arg(CompactArmatures_k.Args.hull_shift, Buffer.hull_shift.memory)
            .mem_arg(CompactArmatures_k.Args.bone_bind_shift, Buffer.bone_bind_shift.memory);

        Kernel.compact_hulls.set_kernel(new CompactHulls_k(command_queue))
            .mem_arg(CompactHulls_k.Args.hull_shift, Buffer.hull_shift.memory)
            .mem_arg(CompactHulls_k.Args.hulls, Buffer.hulls.memory)
            .mem_arg(CompactHulls_k.Args.hull_mesh_ids, Buffer.hull_mesh_ids.memory)
            .mem_arg(CompactHulls_k.Args.hull_rotations, Buffer.hull_rotation.memory)
            .mem_arg(CompactHulls_k.Args.hull_flags, Buffer.hull_flags.memory)
            .mem_arg(CompactHulls_k.Args.element_tables, Buffer.hull_element_tables.memory)
            .mem_arg(CompactHulls_k.Args.bounds, Buffer.aabb.memory)
            .mem_arg(CompactHulls_k.Args.bounds_index_data, Buffer.aabb_index.memory)
            .mem_arg(CompactHulls_k.Args.bounds_bank_data, Buffer.aabb_key_table.memory);

        Kernel.compact_edges.set_kernel(new CompactEdges_k(command_queue))
            .mem_arg(CompactEdges_k.Args.edge_shift, Buffer.edge_shift.memory)
            .mem_arg(CompactEdges_k.Args.edges, Buffer.edges.memory);

        Kernel.compact_points.set_kernel(new CompactPoints_k(command_queue))
            .mem_arg(CompactPoints_k.Args.point_shift, Buffer.point_shift.memory)
            .mem_arg(CompactPoints_k.Args.points, Buffer.points.memory)
            .mem_arg(CompactPoints_k.Args.anti_gravity, Buffer.point_anti_gravity.memory)
            .mem_arg(CompactPoints_k.Args.vertex_tables, Buffer.point_vertex_tables.memory)
            .mem_arg(CompactPoints_k.Args.bone_tables, Buffer.point_bone_tables.memory);

        Kernel.compact_bones.set_kernel(new CompactBones_k(command_queue))
            .mem_arg(CompactBones_k.Args.bone_shift, Buffer.bone_shift.memory)
            .mem_arg(CompactBones_k.Args.bone_instances, Buffer.bone_instances.memory)
            .mem_arg(CompactBones_k.Args.bone_indices, Buffer.bone_index_tables.memory);

        // movement

        Kernel.animate_points.set_kernel(new AnimatePoints_k(command_queue))
            .mem_arg(AnimatePoints_k.Args.points, Buffer.points.memory)
            .mem_arg(AnimatePoints_k.Args.hulls, Buffer.hulls.memory)
            .mem_arg(AnimatePoints_k.Args.hull_flags, Buffer.hull_flags.memory)
            .mem_arg(AnimatePoints_k.Args.vertex_tables, Buffer.point_vertex_tables.memory)
            .mem_arg(AnimatePoints_k.Args.bone_tables, Buffer.point_bone_tables.memory)
            .mem_arg(AnimatePoints_k.Args.vertex_weights, Buffer.vertex_weights.memory)
            .mem_arg(AnimatePoints_k.Args.armatures, Buffer.armatures.memory)
            .mem_arg(AnimatePoints_k.Args.vertex_references, Buffer.vertex_references.memory)
            .mem_arg(AnimatePoints_k.Args.bones, Buffer.bone_instances.memory);

        Kernel.integrate.set_kernel(new Integrate_k(command_queue))
            .mem_arg(Integrate_k.Args.hulls, Buffer.hulls.memory)
            .mem_arg(Integrate_k.Args.armatures, Buffer.armatures.memory)
            .mem_arg(Integrate_k.Args.armature_flags, Buffer.armature_flags.memory)
            .mem_arg(Integrate_k.Args.element_tables, Buffer.hull_element_tables.memory)
            .mem_arg(Integrate_k.Args.armature_accel, Buffer.armature_accel.memory)
            .mem_arg(Integrate_k.Args.hull_rotations, Buffer.hull_rotation.memory)
            .mem_arg(Integrate_k.Args.points, Buffer.points.memory)
            .mem_arg(Integrate_k.Args.bounds, Buffer.aabb.memory)
            .mem_arg(Integrate_k.Args.bounds_index_data, Buffer.aabb_index.memory)
            .mem_arg(Integrate_k.Args.bounds_bank_data, Buffer.aabb_key_table.memory)
            .mem_arg(Integrate_k.Args.hull_flags, Buffer.hull_flags.memory)
            .mem_arg(Integrate_k.Args.anti_gravity, Buffer.point_anti_gravity.memory);

        // broad collision

        Kernel.generate_keys.set_kernel(new GenerateKeys_k(command_queue))
            .mem_arg(GenerateKeys_k.Args.bounds_index_data, Buffer.aabb_index.memory)
            .mem_arg(GenerateKeys_k.Args.bounds_bank_data, Buffer.aabb_key_table.memory);

        Kernel.build_key_map.set_kernel(new BuildKeyMap_k(command_queue))
            .mem_arg(BuildKeyMap_k.Args.bounds_index_data, Buffer.aabb_index.memory)
            .mem_arg(BuildKeyMap_k.Args.bounds_bank_data, Buffer.aabb_key_table.memory);

        Kernel.locate_in_bounds.set_kernel(new LocateInBounds_k(command_queue))
            .mem_arg(LocateInBounds_k.Args.bounds_bank_data, Buffer.aabb_key_table.memory);

        Kernel.count_candidates.set_kernel(new CountCandidates_k(command_queue))
            .mem_arg(CountCandidates_k.Args.bounds_bank_data, Buffer.aabb_key_table.memory);

        Kernel.aabb_collide.set_kernel(new AABBCollide_k(command_queue))
            .mem_arg(AABBCollide_k.Args.bounds, Buffer.aabb.memory)
            .mem_arg(AABBCollide_k.Args.bounds_bank_data, Buffer.aabb_key_table.memory)
            .mem_arg(AABBCollide_k.Args.hull_flags, Buffer.hull_flags.memory);

        Kernel.finalize_candidates.set_kernel(new FinalizeCandidates_k(command_queue));

        // mesh query

        Kernel.transfer_detail_data.set_kernel(new TransferDetailData_k(command_queue));
        Kernel.calculate_batch_offsets.set_kernel(new CalculateBatchOffsets_k(command_queue));
        Kernel.count_mesh_batches.set_kernel(new CountMeshBatches_k(command_queue));

        Kernel.count_mesh_instances.set_kernel(new CountMeshInstances_k(command_queue))
            .mem_arg(CountMeshInstances_k.Args.hull_mesh_ids, Buffer.hull_mesh_ids.memory);

        Kernel.write_mesh_details.set_kernel(new WriteMeshDetails_k(command_queue))
            .mem_arg(WriteMeshDetails_k.Args.hull_mesh_ids, Buffer.hull_mesh_ids.memory)
            .mem_arg(WriteMeshDetails_k.Args.mesh_references, Buffer.mesh_references.memory);

        Kernel.transfer_render_data.set_kernel(new TransferRenderData_k(command_queue))
            .mem_arg(TransferRenderData_k.Args.hull_element_tables, Buffer.hull_element_tables.memory)
            .mem_arg(TransferRenderData_k.Args.hull_mesh_ids, Buffer.hull_mesh_ids.memory)
            .mem_arg(TransferRenderData_k.Args.mesh_references, Buffer.mesh_references.memory)
            .mem_arg(TransferRenderData_k.Args.mesh_faces, Buffer.mesh_faces.memory)
            .mem_arg(TransferRenderData_k.Args.points, Buffer.points.memory)
            .mem_arg(TransferRenderData_k.Args.vertex_tables, Buffer.point_vertex_tables.memory)
            .mem_arg(TransferRenderData_k.Args.uv_tables, Buffer.uv_tables.memory)
            .mem_arg(TransferRenderData_k.Args.texture_uvs, Buffer.texture_uvs.memory);
    }

    //#endregion

    //#region Utility Methods

    public static void cl_read_buffer(cl_mem src, long size, Pointer dst)
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

    private static cl_mem cl_new_cpu_copy_buffer(long size, Pointer src)
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

    private static cl_mem cl_new_pinned_buffer(long size)
    {
        long flags = CL_MEM_HOST_READ_ONLY | CL_MEM_ALLOC_HOST_PTR;
        return clCreateBuffer(context, flags, size, null, null);
    }

    private static int[] cl_read_pinned_int_buffer(cl_mem pinned, long size, int count)
    {
        var out = clEnqueueMapBuffer(command_queue, pinned, true, CL_MAP_READ, 0, size, 0,
            null, null, null);
        assert out != null;
        int[] xa = new int[count];
        var ib = out.order(ByteOrder.LITTLE_ENDIAN).asIntBuffer();
        for (int i = 0; i < count; i++)
        {
            xa[i] = ib.get(i);
        }
        clEnqueueUnmapMemObject(command_queue, pinned, out, 0, null, null);
        return xa;
    }

    private static float[] cl_read_pinned_float_buffer(cl_mem pinned, long size, int count)
    {
        var out = clEnqueueMapBuffer(command_queue, pinned, true, CL_MAP_READ, 0, size, 0,
            null, null, null);
        assert out != null;
        float[] xa = new float[count];
        var ib = out.order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer();
        for (int i = 0; i < count; i++)
        {
            xa[i] = ib.get(i);
        }
        clEnqueueUnmapMemObject(command_queue, pinned, out, 0, null, null);
        return xa;
    }

    public static cl_mem cl_new_pinned_int()
    {
        long flags = CL_MEM_HOST_READ_ONLY | CL_MEM_ALLOC_HOST_PTR;
        return clCreateBuffer(context, flags, Sizeof.cl_int, null, null);
    }

    public static int cl_read_pinned_int(cl_mem pinned)
    {
        var out = clEnqueueMapBuffer(command_queue, pinned, true, CL_MAP_READ, 0, Sizeof.cl_int, 0,
            null, null, null);
        assert out != null;
        int result = out.order(ByteOrder.LITTLE_ENDIAN).asIntBuffer().get(0);
        clEnqueueUnmapMemObject(command_queue, pinned, out, 0, null, null);
        return result;
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
    private static void linearize_kernel(Kernel kernel, int object_count)
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
        var vbo_mem = shared_mem.get(vbo_id);

        Kernel.prepare_bounds
            .share_mem(vbo_mem)
            .set_arg(PrepareBounds_k.Args.vbo, Pointer.to(vbo_mem))
            .set_arg(PrepareBounds_k.Args.offset, Pointer.to(arg_long(bounds_offset)))
            .call(arg_long(batch_size));
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
        var vbo_mem = shared_mem.get(vbo_id);

        Kernel.prepare_bones
            .share_mem(vbo_mem)
            .set_arg(PrepareBones_k.Args.vbo, Pointer.to(vbo_mem))
            .set_arg(PrepareBones_k.Args.offset, Pointer.to(arg_long(bone_offset)))
            .call(arg_long(batch_size));
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
        var vbo_mem1 = shared_mem.get(vbo_id);
        var vbo_mem2 = shared_mem.get(vbo_id2);

        Kernel.prepare_edges
            .share_mem(vbo_mem1)
            .share_mem(vbo_mem2)
            .set_arg(PrepareEdges_k.Args.vertex_vbo, Pointer.to(vbo_mem1))
            .set_arg(PrepareEdges_k.Args.flag_vbo, Pointer.to(vbo_mem2))
            .set_arg(PrepareEdges_k.Args.offset, Pointer.to(arg_int(edge_offset)))
            .call(arg_long(batch_size));
    }

    public static void GL_points(int vbo_id, int point_offset, int batch_size)
    {
        var vbo_mem = shared_mem.get(vbo_id);

        Kernel.prepare_points
            .share_mem(vbo_mem)
            .set_arg(PreparePoints_k.Args.vertex_vbo, Pointer.to(vbo_mem))
            .set_arg(PreparePoints_k.Args.offset, Pointer.to(arg_int(point_offset)))
            .call(arg_long(batch_size));
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
        cl_zero_buffer(counter_buffer, Sizeof.cl_int);

        Kernel.root_hull_count
            .set_arg(RootHullCount_k.Args.counter, counter_pointer)
            .set_arg(RootHullCount_k.Args.model_id, Pointer.to(arg_int(model_id)))
            .call(arg_long(GPU.Memory.next_armature()));

        int final_count = cl_read_pinned_int(counter_buffer);

        if (final_count == 0)
        {
            return new HullIndexData(null, final_count);
        }

        long final_buffer_size = (long) Sizeof.cl_int * final_count;
        var hulls_out = cl_new_buffer(final_buffer_size);

        // the kernel will use this value as an internal atomic counter, always initialize to zero
        var dst_hulls_counter = Pointer.to(arg_int(0));
        var hulls_counter_data = cl_new_int_arg_buffer(dst_hulls_counter);

        Kernel.root_hull_filter
            .set_arg(RootHullFilter_k.Args.hulls_out, Pointer.to(hulls_out))
            .set_arg(RootHullFilter_k.Args.counter, Pointer.to(hulls_counter_data))
            .set_arg(RootHullFilter_k.Args.model_id, Pointer.to(arg_int(model_id)))
            .call(arg_long(GPU.Memory.next_armature()));

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
     * @param batch_size    number of hull objects to transfer in this batch
     */
    public static void GL_circles(int vbo_id, cl_mem hulls_out, int offset, int batch_size)
    {
        var vbo_mem = shared_mem.get(vbo_id);
        Kernel.prepare_transforms
            .share_mem(vbo_mem)
            .set_arg(PrepareTransforms_k.Args.indices, Pointer.to(hulls_out))
            .set_arg(PrepareTransforms_k.Args.transforms_out, Pointer.to(vbo_mem))
            .set_arg(PrepareTransforms_k.Args.offset, Pointer.to(arg_int(offset)))
            .call(arg_long(batch_size));
    }

    /**
     * Transfers a subset of all hull transforms from CL memory into GL memory. Hulls
     * are generally not rendered directly using this data, but it is used to transform
     * model reference data from memory into the position of the mesh that the hull
     * represents within the simulation.
     *
     * @param transforms_id   id of the shared GL buffer object
     * @param hulls_out id of the shared GL buffer object
     * @param batch_size      number of hull objects to transfer in this batch
     */
    public static void GL_transforms(int transforms_id, cl_mem hulls_out, int batch_size, int offset)
    {
        var vbo_transforms = shared_mem.get(transforms_id);

        Kernel.prepare_transforms
            .share_mem(vbo_transforms)
            .set_arg(PrepareTransforms_k.Args.indices, Pointer.to(hulls_out))
            .set_arg(PrepareTransforms_k.Args.transforms_out, Pointer.to(vbo_transforms))
            .set_arg(PrepareTransforms_k.Args.offset, Pointer.to(arg_int(offset)))
            .call(arg_long(batch_size));
    }

    public static void GL_count_mesh_instances(cl_mem query, cl_mem counters, cl_mem total, int count)
    {
        Kernel.count_mesh_instances
            .set_arg(CountMeshInstances_k.Args.counters, Pointer.to(counters))
            .set_arg(CountMeshInstances_k.Args.query, Pointer.to(query))
            .set_arg(CountMeshInstances_k.Args.total, Pointer.to(total))
            .set_arg(CountMeshInstances_k.Args.count, Pointer.to(arg_int(count)))
            .call(arg_long(GPU.Memory.next_hull()));
    }

    public static void GL_scan_mesh_offsets(cl_mem counts, cl_mem offsets, int count)
    {
        scan_int_out(counts, offsets, count);
    }

    public static void GL_write_mesh_details(cl_mem query, cl_mem counters, cl_mem offsets, cl_mem mesh_details, int count)
    {
        Kernel.write_mesh_details
            .set_arg(WriteMeshDetails_k.Args.counters, Pointer.to(counters))
            .set_arg(WriteMeshDetails_k.Args.query, Pointer.to(query))
            .set_arg(WriteMeshDetails_k.Args.offsets, Pointer.to(offsets))
            .set_arg(WriteMeshDetails_k.Args.mesh_details, Pointer.to(mesh_details))
            .set_arg(WriteMeshDetails_k.Args.count, Pointer.to(arg_int(count)))
            .call(arg_long(GPU.Memory.next_hull()));
    }

    public static void GL_count_mesh_batches(cl_mem mesh_details, cl_mem total, int count, int max_per_batch)
    {
        Kernel.count_mesh_batches
            .set_arg(CountMeshBatches_k.Args.mesh_details, Pointer.to(mesh_details))
            .set_arg(CountMeshBatches_k.Args.total, Pointer.to(total))
            .set_arg(CountMeshBatches_k.Args.max_per_batch, Pointer.to(arg_int(max_per_batch)))
            .set_arg(CountMeshBatches_k.Args.count, Pointer.to(arg_int(count)))
            .call(global_single_size);
    }

    public static void GL_calculate_batch_offsets(cl_mem mesh_offsets, cl_mem mesh_details, int count)
    {
        Kernel.calculate_batch_offsets
            .set_arg(CalculateBatchOffsets_k.Args.mesh_offsets, Pointer.to(mesh_offsets))
            .set_arg(CalculateBatchOffsets_k.Args.mesh_details, Pointer.to(mesh_details))
            .set_arg(CalculateBatchOffsets_k.Args.count, Pointer.to(arg_int(count)))
            .call(global_single_size);
    }

    public static void GL_transfer_detail_data(cl_mem mesh_details, cl_mem mesh_transfer, int count, int offset)
    {
        Kernel.transfer_detail_data
            .set_arg(TransferDetailData_k.Args.mesh_details, Pointer.to(mesh_details))
            .set_arg(TransferDetailData_k.Args.mesh_transfer, Pointer.to(mesh_transfer))
            .set_arg(TransferDetailData_k.Args.offset, Pointer.to(arg_int(offset)))
            .call(arg_long(count));

        scan_int2(mesh_transfer, count);
    }

    public static void GL_transfer_render_data(int ebo,
                                               int vbo,
                                               int cbo,
                                               int uvo,
                                               cl_mem mesh_details,
                                               cl_mem mesh_transfer,
                                               int count,
                                               int offset)
    {
        var ebo_mem = shared_mem.get(ebo);
        var vbo_mem = shared_mem.get(vbo);
        var cbo_mem = shared_mem.get(cbo);
        var uvo_mem = shared_mem.get(uvo);

        Kernel.transfer_render_data
            .share_mem(ebo_mem)
            .share_mem(vbo_mem)
            .share_mem(cbo_mem)
            .share_mem(uvo_mem)
            .set_arg(TransferRenderData_k.Args.command_buffer, Pointer.to(cbo_mem))
            .set_arg(TransferRenderData_k.Args.vertex_buffer, Pointer.to(vbo_mem))
            .set_arg(TransferRenderData_k.Args.uv_buffer, Pointer.to(uvo_mem))
            .set_arg(TransferRenderData_k.Args.element_buffer, Pointer.to(ebo_mem))
            .set_arg(TransferRenderData_k.Args.mesh_details, Pointer.to(mesh_details))
            .set_arg(TransferRenderData_k.Args.mesh_transfer, Pointer.to(mesh_transfer))
            .set_arg(TransferRenderData_k.Args.offset, Pointer.to(arg_int(offset)))
            .call(arg_long(count));
    }

    //#endregion

    //#region CPU Create/Read/Update/Delete Functions

    public static void create_point(int point_index, float[] position, int[] vertex_table, int[] bone_indices)
    {
        Kernel.create_point
            .set_arg(CreatePoint_k.Args.target, Pointer.to(arg_int(point_index)))
            .set_arg(CreatePoint_k.Args.new_point, Pointer.to(position))
            .set_arg(CreatePoint_k.Args.new_vertex_table, Pointer.to(vertex_table))
            .set_arg(CreatePoint_k.Args.new_bone_table, Pointer.to(bone_indices))
            .call(global_single_size);
    }

    public static void create_texture_uv(int uv_index, float u, float v)
    {
        Kernel.create_texture_uv
            .set_arg(CreateTextureUV_k.Args.target, Pointer.to(arg_int(uv_index)))
            .set_arg(CreateTextureUV_k.Args.new_texture_uv, Pointer.to(arg_float2(u, v)))
            .call(global_single_size);
    }

    public static void create_edge(int edge_index, float p1, float p2, float l, int flags)
    {
        Kernel.create_edge
            .set_arg(CreateEdge_k.Args.target, Pointer.to(arg_int(edge_index)))
            .set_arg(CreateEdge_k.Args.new_edge, Pointer.to(arg_float4(p1, p2, l, flags)))
            .call(global_single_size);
    }

    public static void create_armature(int armature_index, float x, float y, int[] table, int[] flags, float mass)
    {
        Kernel.create_armature
            .set_arg(CreateArmature_k.Args.target, Pointer.to(arg_int(armature_index)))
            .set_arg(CreateArmature_k.Args.new_armature, Pointer.to(arg_float4(x, y, x, y)))
            .set_arg(CreateArmature_k.Args.new_armature_flags, Pointer.to(flags))
            .set_arg(CreateArmature_k.Args.new_hull_table, Pointer.to(table))
            .set_arg(CreateArmature_k.Args.new_armature_mass, Pointer.to(arg_float(mass)))
            .call(global_single_size);
    }

    public static void create_vertex_reference(int vert_ref_index, float x, float y, float[] weights, int[] uv_table)
    {
        Kernel.create_vertex_reference
            .set_arg(CreateVertexRef_k.Args.target, Pointer.to(arg_int(vert_ref_index)))
            .set_arg(CreateVertexRef_k.Args.new_vertex_reference, Pointer.to(arg_float2(x, y)))
            .set_arg(CreateVertexRef_k.Args.new_vertex_weights, Pointer.to(weights))
            .set_arg(CreateVertexRef_k.Args.new_uv_table, Pointer.to(uv_table))
            .call(global_single_size);
    }

    public static void create_bone_bind_pose(int bone_bind_index, int bone_bond_parent, float[] matrix)
    {
        Kernel.create_bone_bind_pose
            .set_arg(CreateBoneBindPose_k.Args.target, Pointer.to(arg_int(bone_bind_index)))
            .set_arg(CreateBoneBindPose_k.Args.new_bone_bind_pose, Pointer.to(matrix))
            .set_arg(CreateBoneBindPose_k.Args.bone_bind_parent, Pointer.to(arg_int(bone_bond_parent)))
            .call(global_single_size);
    }

    public static void create_bone_reference(int bone_ref_index, float[] matrix)
    {
        Kernel.create_bone_reference
            .set_arg(CreateBoneRef_k.Args.target, Pointer.to(arg_int(bone_ref_index)))
            .set_arg(CreateBoneRef_k.Args.new_bone_reference, Pointer.to(matrix))
            .call(global_single_size);
    }

    public static void create_bone(int bone_index, int[] bone_index_table, float[] matrix)
    {
        Kernel.create_bone
            .set_arg(CreateBone_k.Args.target, Pointer.to(arg_int(bone_index)))
            .set_arg(CreateBone_k.Args.new_bone, Pointer.to(matrix))
            .set_arg(CreateBone_k.Args.new_bone_table, Pointer.to(bone_index_table))
            .call(global_single_size);
    }

    public static void create_armature_bone(int armature_bone_index, int[] bone_bind_table, float[] matrix)
    {
        Kernel.create_armature_bone
            .set_arg(CreateArmatureBone_k.Args.target, Pointer.to(arg_int(armature_bone_index)))
            .set_arg(CreateArmatureBone_k.Args.new_armature_bone, Pointer.to(matrix))
            .set_arg(CreateArmatureBone_k.Args.new_bone_bind_table, Pointer.to(bone_bind_table))
            .call(global_single_size);
    }

    public static void create_hull(int hull_index, int mesh_index, float[] hull, float[] rotation, int[] table, int[] flags)
    {
        Kernel.create_hull
            .set_arg(CreateHull_k.Args.target, Pointer.to(arg_int(hull_index)))
            .set_arg(CreateHull_k.Args.new_hull, Pointer.to(hull))
            .set_arg(CreateHull_k.Args.new_rotation, Pointer.to(rotation))
            .set_arg(CreateHull_k.Args.new_table, Pointer.to(table))
            .set_arg(CreateHull_k.Args.new_flags, Pointer.to(flags))
            .set_arg(CreateHull_k.Args.new_hull_mesh_id, Pointer.to(arg_int(mesh_index)))
            .call(global_single_size);
    }

    public static void create_mesh_reference(int mesh_index, int[] mesh_ref_table)
    {
        Kernel.create_mesh_reference
            .set_arg(CreateMeshReference_k.Args.target, Pointer.to(arg_int(mesh_index)))
            .set_arg(CreateMeshReference_k.Args.new_mesh_ref_table, Pointer.to(mesh_ref_table))
            .call(global_single_size);
    }

    public static void create_mesh_face(int face_index, int[] face)
    {
        Kernel.create_mesh_face
            .set_arg(CreateMeshFace_k.Args.target, Pointer.to(arg_int(face_index)))
            .set_arg(CreateMeshFace_k.Args.new_mesh_face, Pointer.to(face))
            .call(global_single_size);
    }

    public static void update_accel(int armature_index, float acc_x, float acc_y)
    {
        Kernel.update_accel
            .set_arg(UpdateAccel_k.Args.target, Pointer.to(arg_int(armature_index)))
            .set_arg(UpdateAccel_k.Args.new_value, Pointer.to(arg_float2(acc_x, acc_y)))
            .call(global_single_size);
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
        var result_data = cl_new_pinned_buffer(Sizeof.cl_float2);
        cl_zero_buffer(result_data, Sizeof.cl_float2);

        Kernel.read_position
            .set_arg(ReadPosition_k.Args.output, Pointer.to(result_data))
            .set_arg(ReadPosition_k.Args.target, Pointer.to(arg_int(armature_index)))
            .call(global_single_size);

        float[] result = cl_read_pinned_float_buffer(result_data, Sizeof.cl_float2, 2);
        clReleaseMemObject(result_data);
        return result;
    }

    //#endregion

    //#region Physics Simulation

    public static void animate_hulls()
    {
        Kernel.animate_points.call(arg_long(GPU.Memory.next_point()));
    }

    public static void integrate(float delta_time, UniformGrid uniform_grid)
    {
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

        var src_args = Pointer.to(args);

        long size = Sizeof.cl_float * args.length;
        var arg_mem = cl_new_cpu_copy_buffer(size, src_args);

        Kernel.integrate
            .set_arg(Integrate_k.Args.args, Pointer.to(arg_mem))
            .call(arg_long(GPU.Memory.next_hull()));

        clReleaseMemObject(arg_mem);
    }

    public static void calculate_bank_offsets(UniformGrid uniform_grid)
    {
        int bank_size = scan_key_bounds(Buffer.aabb_key_table.memory.memory(), GPU.Memory.next_hull());
        uniform_grid.resizeBank(bank_size);
    }

    public static void generate_keys(UniformGrid uniform_grid)
    {
        if (uniform_grid.get_key_bank_size() < 1)
        {
            return;
        }

        long bank_buf_size = (long) Sizeof.cl_int * uniform_grid.get_key_bank_size();
        long counts_buf_size = (long) Sizeof.cl_int * uniform_grid.get_directory_length();

        var bank_data = cl_new_buffer(bank_buf_size);
        var counts_data = cl_new_buffer(counts_buf_size);
        cl_zero_buffer(counts_data, counts_buf_size);

        physics_buffer.key_counts = new GPUMemory(counts_data);
        physics_buffer.key_bank = new GPUMemory(bank_data);

        Kernel.generate_keys
            .set_arg(GenerateKeys_k.Args.key_bank, Pointer.to(bank_data))
            .set_arg(GenerateKeys_k.Args.key_counts, Pointer.to(counts_data))
            .set_arg(GenerateKeys_k.Args.x_subdivisions, Pointer.to(arg_int(uniform_grid.getX_subdivisions())))
            .set_arg(GenerateKeys_k.Args.key_bank_length, Pointer.to(arg_int(uniform_grid.get_key_bank_size())))
            .set_arg(GenerateKeys_k.Args.key_count_length, Pointer.to(arg_int(uniform_grid.get_directory_length())))
            .call(arg_long(GPU.Memory.next_hull()));
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
        long map_buf_size = (long) Sizeof.cl_int * uniform_grid.getKey_map_size();
        long counts_buf_size = (long) Sizeof.cl_int * uniform_grid.get_directory_length();

        var map_data = cl_new_buffer(map_buf_size);
        var counts_data = cl_new_buffer(counts_buf_size);

        // the counts buffer needs to start off filled with all zeroes
        cl_zero_buffer(counts_data, counts_buf_size);

        physics_buffer.key_map = new GPUMemory(map_data);

        Kernel.build_key_map
            .set_arg(BuildKeyMap_k.Args.key_map, Pointer.to(map_data))
            .set_arg(BuildKeyMap_k.Args.key_offsets, physics_buffer.key_offsets.pointer())
            .set_arg(BuildKeyMap_k.Args.key_counts, Pointer.to(counts_data))
            .set_arg(BuildKeyMap_k.Args.x_subdivisions, Pointer.to(arg_int(uniform_grid.getX_subdivisions())))
            .set_arg(BuildKeyMap_k.Args.key_count_length, Pointer.to(arg_int(uniform_grid.get_directory_length())))
            .call(arg_long(GPU.Memory.next_hull()));

        clReleaseMemObject(counts_data);
    }

    public static void locate_in_bounds(UniformGrid uniform_grid)
    {
        int hull_count = GPU.Memory.next_hull();

        int x_subdivisions = uniform_grid.getX_subdivisions();
        physics_buffer.x_sub_divisions = Pointer.to(arg_int(x_subdivisions));
        physics_buffer.key_count_length = Pointer.to(arg_int(uniform_grid.get_directory_length()));

        long inbound_buf_size = (long) Sizeof.cl_int * hull_count;
        var inbound_data = cl_new_buffer(inbound_buf_size);

        physics_buffer.in_bounds = new GPUMemory(inbound_data);

        cl_zero_buffer(counter_buffer, Sizeof.cl_int);

        Kernel.locate_in_bounds
            .set_arg(LocateInBounds_k.Args.in_bounds, physics_buffer.in_bounds.pointer())
            .set_arg(LocateInBounds_k.Args.counter, counter_pointer)
            .call(arg_long(hull_count));

        int size = cl_read_pinned_int(counter_buffer);

        physics_buffer.set_candidate_buffer_count(size);
    }

    public static void locate_out_of_bounds()
    {
        int armature_count = GPU.Memory.next_armature();

        int[] counter = new int[]{0};
        var dst_counter = Pointer.to(counter);
        var counter_data = cl_new_int_arg_buffer(dst_counter);
        var p = Pointer.to(counter_data);

        Kernel.locate_out_of_bounds
            .set_arg(LocateOutOfBounds_k.Args.counter, p)
            .call(arg_long(armature_count));

        clReleaseMemObject(counter_data);
    }

    public static void calculate_match_candidates()
    {
        long candidate_buf_size = (long) Sizeof.cl_int2 * physics_buffer.get_candidate_buffer_count();
        var candidate_data = cl_new_buffer(candidate_buf_size);
        physics_buffer.candidate_counts = new GPUMemory(candidate_data);

        Kernel.count_candidates
            .set_arg(CountCandidates_k.Args.in_bounds, physics_buffer.in_bounds.pointer())
            .set_arg(CountCandidates_k.Args.key_bank, physics_buffer.key_bank.pointer())
            .set_arg(CountCandidates_k.Args.key_counts, physics_buffer.key_counts.pointer())
            .set_arg(CountCandidates_k.Args.candidates, physics_buffer.candidate_counts.pointer())
            .set_arg(CountCandidates_k.Args.x_subdivisions, physics_buffer.x_sub_divisions)
            .set_arg(CountCandidates_k.Args.key_count_length, physics_buffer.key_count_length)
            .call(arg_long(physics_buffer.get_candidate_buffer_count()));
    }

    public static void calculate_match_offsets()
    {
        int buffer_count = physics_buffer.get_candidate_buffer_count();
        long offset_buf_size = (long) Sizeof.cl_int * buffer_count;
        var offset_data = cl_new_buffer(offset_buf_size);
        physics_buffer.candidate_offsets = new GPUMemory(offset_data);
        int match_count = scan_key_candidates(physics_buffer.candidate_counts.memory(), offset_data, buffer_count);
        physics_buffer.set_candidate_match_count(match_count);
    }

    // todo: extend buffers to account for armature bones
    public static void delete_and_compact()
    {
        int armature_count = GPU.Memory.next_armature();
        long output_buf_size = (long) Sizeof.cl_int * armature_count;
        long output_buf_size2 = (long) Sizeof.cl_int4 * armature_count;

        var output_buf_data = cl_new_buffer(output_buf_size);
        var output_buf_data2 = cl_new_buffer(output_buf_size2);

        var del_buffer_1 = new GPUMemory(output_buf_data);
        var del_buffer_2 = new GPUMemory(output_buf_data2);

        int[] shift_counts = scan_deletes(del_buffer_1.memory(), del_buffer_2.memory(), armature_count);

        if (shift_counts[4] == 0)
        {
            del_buffer_1.release();
            del_buffer_2.release();
            return;
        }

        // shift buffers are cleared before compacting to clean out any data from the last tick
        Buffer.hull_shift.clear();
        Buffer.edge_shift.clear();
        Buffer.point_shift.clear();
        Buffer.bone_shift.clear();

        // as armatures are compacted, the shift buffers for the other components are updated
        Kernel.compact_armatures
            .set_arg(CompactArmatures_k.Args.buffer_in, del_buffer_1.pointer())
            .set_arg(CompactArmatures_k.Args.buffer_in_2, del_buffer_2.pointer());

        linearize_kernel(Kernel.compact_armatures, armature_count);
        linearize_kernel(Kernel.compact_bones, GPU.Memory.next_bone());
        linearize_kernel(Kernel.compact_points, GPU.Memory.next_point());
        linearize_kernel(Kernel.compact_edges, GPU.Memory.next_edge());
        linearize_kernel(Kernel.compact_hulls, GPU.Memory.next_hull());

        GPU.Memory.compact_buffers(shift_counts[0], shift_counts[1], shift_counts[2], shift_counts[3], shift_counts[4]);

        del_buffer_1.release();
        del_buffer_2.release();
    }

    public static void aabb_collide()
    {
        long matches_buf_size = (long) Sizeof.cl_int * physics_buffer.get_candidate_match_count();
        var matches_data = cl_new_buffer(matches_buf_size);
        physics_buffer.matches = new GPUMemory(matches_data);

        long used_buf_size = (long) Sizeof.cl_int * physics_buffer.get_candidate_buffer_count();
        var used_data = cl_new_buffer(used_buf_size);
        physics_buffer.matches_used = new GPUMemory(used_data);

        cl_zero_buffer(counter_buffer, Sizeof.cl_int);

        Kernel.aabb_collide
            .set_arg(AABBCollide_k.Args.candidates, physics_buffer.candidate_counts.pointer())
            .set_arg(AABBCollide_k.Args.match_offsets, physics_buffer.candidate_offsets.pointer())
            .set_arg(AABBCollide_k.Args.key_map, physics_buffer.key_map.pointer())
            .set_arg(AABBCollide_k.Args.key_bank, physics_buffer.key_bank.pointer())
            .set_arg(AABBCollide_k.Args.key_counts, physics_buffer.key_counts.pointer())
            .set_arg(AABBCollide_k.Args.key_offsets, physics_buffer.key_offsets.pointer())
            .set_arg(AABBCollide_k.Args.matches, physics_buffer.matches.pointer())
            .set_arg(AABBCollide_k.Args.used, physics_buffer.matches_used.pointer())
            .set_arg(AABBCollide_k.Args.counter, counter_pointer)
            .set_arg(AABBCollide_k.Args.x_subdivisions, physics_buffer.x_sub_divisions)
            .set_arg(AABBCollide_k.Args.key_count_length, physics_buffer.key_count_length)
            .call(arg_long(physics_buffer.get_candidate_buffer_count()));

        int count = cl_read_pinned_int(counter_buffer);
        physics_buffer.set_candidate_count(count);
    }

    public static void finalize_candidates()
    {
        if (physics_buffer.get_candidate_count() <= 0)
        {
            return;
        }

        // create an empty buffer that the kernel will use to store finalized candidates
        long final_buf_size = (long) Sizeof.cl_int2 * physics_buffer.get_candidate_count();
        var finals_data = cl_new_buffer(final_buf_size);

        // the kernel will use this value as an internal atomic counter, always initialize to zero
        int[] counter = new int[]{0};
        var dst_counter = Pointer.to(counter);
        var counter_data = cl_new_int_arg_buffer(dst_counter);

        physics_buffer.set_final_size(final_buf_size);
        physics_buffer.candidates = new GPUMemory(finals_data);

        Kernel.finalize_candidates
            .set_arg(FinalizeCandidates_k.Args.input_candidates, physics_buffer.candidate_counts.pointer())
            .set_arg(FinalizeCandidates_k.Args.match_offsets, physics_buffer.candidate_offsets.pointer())
            .set_arg(FinalizeCandidates_k.Args.matches, physics_buffer.matches.pointer())
            .set_arg(FinalizeCandidates_k.Args.used, physics_buffer.matches_used.pointer())
            .set_arg(FinalizeCandidates_k.Args.counter, Pointer.to(counter_data))
            .set_arg(FinalizeCandidates_k.Args.final_candidates, physics_buffer.candidates.pointer())
            .call(arg_long(physics_buffer.get_candidate_buffer_count()));

        clReleaseMemObject(counter_data);
    }

    public static void sat_collide()
    {
        int candidates_size = (int) physics_buffer.get_final_size() / Sizeof.cl_int;

        // candidates are pairs of integer indices, so the global size is half the count
        long[] global_work_size = new long[]{candidates_size / 2};

        cl_zero_buffer(counter_buffer, Sizeof.cl_int);

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

        Kernel.sat_collide
            .set_arg(SatCollide_k.Args.candidates, physics_buffer.candidates.pointer())
            .set_arg(SatCollide_k.Args.reactions, physics_buffer.reactions_in.pointer())
            .set_arg(SatCollide_k.Args.reaction_index, physics_buffer.reaction_index.pointer())
            .set_arg(SatCollide_k.Args.counter, counter_pointer)
            .call(global_work_size);

        int size = cl_read_pinned_int(counter_buffer);
        physics_buffer.set_reaction_count(size);
    }

    public static void scan_reactions()
    {
        scan_int_out(Buffer.point_reactions.memory.memory(), Buffer.point_offsets.memory.memory(), GPU.Memory.next_point());
        // it is important to zero out the reactions buffer after the scan. It will be reused during sorting
        Buffer.point_reactions.clear();
    }

    public static void sort_reactions()
    {
        Kernel.sort_reactions
            .set_arg(SortReactions_k.Args.reactions_in, physics_buffer.reactions_in.pointer())
            .set_arg(SortReactions_k.Args.reactions_out, physics_buffer.reactions_out.pointer())
            .set_arg(SortReactions_k.Args.reaction_index, physics_buffer.reaction_index.pointer())
            .call(arg_long(physics_buffer.get_reaction_count()));
    }

    public static void apply_reactions()
    {
        Kernel.apply_reactions
            .set_arg(ApplyReactions_k.Args.reactions, physics_buffer.reactions_out.pointer())
            .call(arg_long(GPU.Memory.next_point()));
    }

    public static void move_armatures()
    {
        Kernel.move_armatures.call(arg_long(GPU.Memory.next_armature()));
    }

    public static void resolve_constraints(int edge_steps)
    {
        boolean last_step;
        for (int i = 0; i < edge_steps; i++)
        {
            last_step = i == edge_steps - 1;
            int n = last_step
                ? 1
                : 0;

            Kernel.resolve_constraints
                .set_arg(ResolveConstraints_k.Args.process_all, Pointer.to(arg_int(n)))
                .call(arg_long(GPU.Memory.next_hull()));
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

    private static void scan_int2(cl_mem d_data, int n)
    {
        int k = work_group_count(n);
        if (k == 1)
        {
            scan_single_block_int2(d_data, n);
        }
        else
        {
            scan_multi_block_int2(d_data, n, k);
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
        long local_buffer_size = Sizeof.cl_int * max_scan_block_size;

        Kernel.scan_int_single_block
            .set_arg(ScanIntSingleBlock_k.Args.data, Pointer.to(d_data))
            .new_arg(ScanIntSingleBlock_k.Args.buffer, local_buffer_size, null)
            .set_arg(ScanIntSingleBlock_k.Args.n, Pointer.to(arg_int(n)))
            .call(local_work_default, local_work_default);
    }

    private static void scan_multi_block_int(cl_mem d_data, int n, int k)
    {
        long local_buffer_size = Sizeof.cl_int * max_scan_block_size;
        long gx = k * max_scan_block_size;
        long[] global_work_size = arg_long(gx);
        int part_size = k * 2;
        long part_buf_size = ((long) Sizeof.cl_int * ((long) part_size));

        var part_data = cl_new_buffer(part_buf_size);
        var src_data = Pointer.to(d_data);
        var src_part = Pointer.to(part_data);
        var src_n = Pointer.to(new int[]{n});

        Kernel.scan_int_multi_block
            .set_arg(ScanIntMultiBlock_k.Args.data, src_data)
            .new_arg(ScanIntMultiBlock_k.Args.buffer, local_buffer_size)
            .set_arg(ScanIntMultiBlock_k.Args.part, src_part)
            .set_arg(ScanIntMultiBlock_k.Args.n, src_n)
            .call(global_work_size, local_work_default);

        scan_int(part_data, part_size);

        Kernel.complete_int_multi_block
            .set_arg(CompleteIntMultiBlock_k.Args.data, src_data)
            .new_arg(CompleteIntMultiBlock_k.Args.buffer, local_buffer_size)
            .set_arg(CompleteIntMultiBlock_k.Args.part, src_part)
            .set_arg(CompleteIntMultiBlock_k.Args.n, src_n)
            .call(global_work_size, local_work_default);

        clReleaseMemObject(part_data);
    }

    private static void scan_single_block_int2(cl_mem d_data, int n)
    {
        long local_buffer_size = Sizeof.cl_int2 * max_scan_block_size;

        Kernel.scan_int2_single_block
            .set_arg(ScanInt2SingleBlock_k.Args.data, Pointer.to(d_data))
            .new_arg(ScanInt2SingleBlock_k.Args.buffer, local_buffer_size)
            .set_arg(ScanInt2SingleBlock_k.Args.n, Pointer.to(arg_int(n)))
            .call(local_work_default, local_work_default);
    }

    private static void scan_multi_block_int2(cl_mem d_data, int n, int k)
    {
        long local_buffer_size = Sizeof.cl_int2 * max_scan_block_size;
        long gx = k * max_scan_block_size;
        long[] global_work_size = arg_long(gx);
        int part_size = k * 2;
        long part_buf_size = ((long) Sizeof.cl_int2 * ((long) part_size));

        var part_data = cl_new_buffer(part_buf_size);
        var src_data = Pointer.to(d_data);
        var src_part = Pointer.to(part_data);
        var src_n = Pointer.to(new int[]{n});

        Kernel.scan_int2_multi_block
            .set_arg(ScanInt2MultiBlock_k.Args.data, src_data)
            .new_arg(ScanInt2MultiBlock_k.Args.buffer, local_buffer_size)
            .set_arg(ScanInt2MultiBlock_k.Args.part, src_part)
            .set_arg(ScanInt2MultiBlock_k.Args.n, src_n)
            .call(global_work_size, local_work_default);

        scan_int2(part_data, part_size);

        Kernel.complete_int2_multi_block
            .set_arg(CompleteInt2MultiBlock_k.Args.data, src_data)
            .new_arg(CompleteInt2MultiBlock_k.Args.buffer, local_buffer_size)
            .set_arg(CompleteInt2MultiBlock_k.Args.part, src_part)
            .set_arg(CompleteInt2MultiBlock_k.Args.n, src_n)
            .call(global_work_size, local_work_default);

        clReleaseMemObject(part_data);
    }

    private static void scan_single_block_int4(cl_mem d_data, int n)
    {
        long local_buffer_size = Sizeof.cl_int4 * max_scan_block_size;

        Kernel.scan_int4_single_block
            .set_arg(ScanInt4SingleBlock_k.Args.data, Pointer.to(d_data))
            .new_arg(ScanInt4SingleBlock_k.Args.buffer, local_buffer_size, null)
            .set_arg(ScanInt4SingleBlock_k.Args.n, Pointer.to(arg_int(n)))
            .call(local_work_default, local_work_default);
    }

    private static void scan_multi_block_int4(cl_mem d_data, int n, int k)
    {
        long local_buffer_size = Sizeof.cl_int4 * max_scan_block_size;
        long gx = k * max_scan_block_size;
        long[] global_work_size = arg_long(gx);
        int part_size = k * 2;
        long part_buf_size = ((long) Sizeof.cl_int4 * ((long) part_size));

        var part_data = cl_new_buffer(part_buf_size);
        var src_data = Pointer.to(d_data);
        var src_part = Pointer.to(part_data);
        var src_n = Pointer.to(new int[]{n});

        Kernel.scan_int4_multi_block
            .set_arg(ScanInt4MultiBlock_k.Args.data, src_data)
            .new_arg(ScanInt4MultiBlock_k.Args.buffer, local_buffer_size)
            .set_arg(ScanInt4MultiBlock_k.Args.part, src_part)
            .set_arg(ScanInt4MultiBlock_k.Args.n, src_n)
            .call(global_work_size, local_work_default);

        scan_int4(part_data, part_size);

        Kernel.complete_int4_multi_block
            .set_arg(CompleteInt4MultiBlock_k.Args.data, src_data)
            .new_arg(CompleteInt4MultiBlock_k.Args.buffer, local_buffer_size)
            .set_arg(CompleteInt4MultiBlock_k.Args.part, src_part)
            .set_arg(CompleteInt4MultiBlock_k.Args.n, src_n)
            .call(global_work_size, local_work_default);

        clReleaseMemObject(part_data);
    }

    private static void scan_single_block_int_out(cl_mem d_data, cl_mem o_data, int n)
    {
        long local_buffer_size = Sizeof.cl_int * max_scan_block_size;

        Kernel.scan_int_single_block_out
            .set_arg(ScanIntSingleBlockOut_k.Args.input, Pointer.to(d_data))
            .set_arg(ScanIntSingleBlockOut_k.Args.output, Pointer.to(o_data))
            .new_arg(ScanIntSingleBlockOut_k.Args.buffer, local_buffer_size)
            .set_arg(ScanIntSingleBlockOut_k.Args.n, Pointer.to(arg_int(n)))
            .call(local_work_default, local_work_default);
    }

    private static void scan_multi_block_int_out(cl_mem input_data, cl_mem o_data, int n, int k)
    {
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

        Kernel.scan_int_multi_block_out
            .set_arg(ScanIntMultiBlockOut_k.Args.input, src_data)
            .set_arg(ScanIntMultiBlockOut_k.Args.output, dst_data)
            .new_arg(ScanIntMultiBlockOut_k.Args.buffer, local_buffer_size)
            .set_arg(ScanIntMultiBlockOut_k.Args.part, src_part)
            .set_arg(ScanIntMultiBlockOut_k.Args.n, src_n)
            .call(global_work_size, local_work_default);

        scan_int(part_data, part_size);

        Kernel.complete_int_multi_block_out
            .set_arg(CompleteIntMultiBlockOut_k.Args.output, dst_data)
            .new_arg(CompleteIntMultiBlockOut_k.Args.buffer, local_buffer_size)
            .set_arg(CompleteIntMultiBlockOut_k.Args.part, src_part)
            .set_arg(CompleteIntMultiBlockOut_k.Args.n, src_n)
            .call(global_work_size, local_work_default);

        clReleaseMemObject(part_data);
    }

    private static int[] scan_single_block_deletes_out(cl_mem o1_data, cl_mem o2_data, int n)
    {
        long local_buffer_size = Sizeof.cl_int * max_scan_block_size;
        long local_buffer_size2 = Sizeof.cl_int4 * max_scan_block_size;

        var size_data = cl_new_pinned_buffer(Sizeof.cl_int * 5);
        cl_zero_buffer(size_data, Sizeof.cl_int * 5);

        var dst_data = Pointer.to(o1_data);
        var dst_data2 = Pointer.to(o2_data);

        Kernel.scan_deletes_single_block_out
            .set_arg(ScanDeletesSingleBlockOut_k.Args.output, dst_data)
            .set_arg(ScanDeletesSingleBlockOut_k.Args.output2, dst_data2)
            .set_arg(ScanDeletesSingleBlockOut_k.Args.sz, Pointer.to(size_data))
            .new_arg(ScanDeletesSingleBlockOut_k.Args.buffer, local_buffer_size)
            .new_arg(ScanDeletesSingleBlockOut_k.Args.buffer2, local_buffer_size2)
            .set_arg(ScanDeletesSingleBlockOut_k.Args.n, Pointer.to(arg_int(n)))
            .call(local_work_default, local_work_default);

        int[] sz = cl_read_pinned_int_buffer(size_data, Sizeof.cl_int * 5, 5);
        clReleaseMemObject(size_data);

        return sz;
    }

    private static int[] scan_multi_block_deletes_out(cl_mem o1_data, cl_mem o2_data, int n, int k)
    {
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

        Kernel.scan_deletes_multi_block_out
            .set_arg(ScanDeletesMultiBlockOut_k.Args.output, dst_data)
            .set_arg(ScanDeletesMultiBlockOut_k.Args.output2, dst_data2)
            .new_arg(ScanDeletesMultiBlockOut_k.Args.buffer, local_buffer_size)
            .new_arg(ScanDeletesMultiBlockOut_k.Args.buffer2, local_buffer_size2)
            .set_arg(ScanDeletesMultiBlockOut_k.Args.part, src_part)
            .set_arg(ScanDeletesMultiBlockOut_k.Args.part2, src_part2)
            .set_arg(ScanDeletesMultiBlockOut_k.Args.n, src_n)
            .call(global_work_size, local_work_default);

        // note the partial buffers are scanned and updated in-place
        scan_int(p_data, part_size);
        scan_int4(p_data2, part_size);

        var size_data = cl_new_pinned_buffer(Sizeof.cl_int * 5);
        cl_zero_buffer(size_data, Sizeof.cl_int * 5);

        Kernel.complete_deletes_multi_block_out
            .set_arg(CompleteDeletesMultiBlockOut_k.Args.output, dst_data)
            .set_arg(CompleteDeletesMultiBlockOut_k.Args.output2, dst_data2)
            .set_arg(CompleteDeletesMultiBlockOut_k.Args.sz, Pointer.to(size_data))
            .new_arg(CompleteDeletesMultiBlockOut_k.Args.buffer, local_buffer_size)
            .new_arg(CompleteDeletesMultiBlockOut_k.Args.buffer2, local_buffer_size2)
            .set_arg(CompleteDeletesMultiBlockOut_k.Args.part, src_part)
            .set_arg(CompleteDeletesMultiBlockOut_k.Args.part2, src_part2)
            .set_arg(CompleteDeletesMultiBlockOut_k.Args.n, src_n)
            .call(global_work_size, local_work_default);

        clReleaseMemObject(p_data);
        clReleaseMemObject(p_data2);

        int[] sz = cl_read_pinned_int_buffer(size_data, Sizeof.cl_int * 5, 5);
        clReleaseMemObject(size_data);

        return sz;
    }

    private static int scan_single_block_candidates_out(cl_mem d_data, cl_mem o_data, int n)
    {
        long local_buffer_size = Sizeof.cl_int * max_scan_block_size;
        var src_data = Pointer.to(d_data);
        var dst_data = Pointer.to(o_data);

        cl_zero_buffer(counter_buffer, Sizeof.cl_int);

        Kernel.scan_candidates_single_block_out
            .set_arg(ScanCandidatesSingleBlockOut_k.Args.input, src_data)
            .set_arg(ScanCandidatesSingleBlockOut_k.Args.output, dst_data)
            .set_arg(ScanCandidatesSingleBlockOut_k.Args.sz, counter_pointer)
            .new_arg(ScanCandidatesSingleBlockOut_k.Args.buffer, local_buffer_size)
            .set_arg(ScanCandidatesSingleBlockOut_k.Args.n, Pointer.to(arg_int(n)))
            .call(local_work_default, local_work_default);

        return cl_read_pinned_int(counter_buffer);
    }

    private static int scan_multi_block_candidates_out(cl_mem d_data, cl_mem o_data, int n, int k)
    {
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

        Kernel.scan_candidates_multi_block_out
            .set_arg(ScanCandidatesMultiBlockOut_k.Args.input, src_data)
            .set_arg(ScanCandidatesMultiBlockOut_k.Args.output, dst_data)
            .new_arg(ScanCandidatesMultiBlockOut_k.Args.buffer, local_buffer_size)
            .set_arg(ScanCandidatesMultiBlockOut_k.Args.part, src_part)
            .set_arg(ScanCandidatesMultiBlockOut_k.Args.n, src_n)
            .call(global_work_size, local_work_default);

        scan_int(p_data, part_size);

        cl_zero_buffer(counter_buffer, Sizeof.cl_int);

        Kernel.complete_candidates_multi_block_out
            .set_arg(CompleteCandidatesMultiBlockOut_k.Args.input, src_data)
            .set_arg(CompleteCandidatesMultiBlockOut_k.Args.output, dst_data)
            .set_arg(CompleteCandidatesMultiBlockOut_k.Args.sz, counter_pointer)
            .new_arg(CompleteCandidatesMultiBlockOut_k.Args.buffer, local_buffer_size)
            .set_arg(CompleteCandidatesMultiBlockOut_k.Args.part, src_part)
            .set_arg(CompleteCandidatesMultiBlockOut_k.Args.n, src_n)
            .call(global_work_size, local_work_default);

        clReleaseMemObject(p_data);

        return cl_read_pinned_int(counter_buffer);
    }

    private static int scan_bounds_single_block(cl_mem input_data, int n)
    {
        long local_buffer_size = Sizeof.cl_int * max_scan_block_size;

        cl_zero_buffer(counter_buffer, Sizeof.cl_int);

        Kernel.scan_bounds_single_block
            .set_arg(ScanBoundsSingleBlock_k.Args.bounds_bank_data, Pointer.to(input_data))
            .set_arg(ScanBoundsSingleBlock_k.Args.sz, counter_pointer)
            .new_arg(ScanBoundsSingleBlock_k.Args.buffer, local_buffer_size)
            .set_arg(ScanBoundsSingleBlock_k.Args.n, Pointer.to(arg_int(n)))
            .call(local_work_default, local_work_default);

        return cl_read_pinned_int(counter_buffer);
    }

    private static int scan_bounds_multi_block(cl_mem input_data, int n, int k)
    {
        long local_buffer_size = Sizeof.cl_int * max_scan_block_size;
        long gx = k * max_scan_block_size;
        long[] global_work_size = arg_long(gx);
        int part_size = k * 2;
        long part_buf_size = ((long) Sizeof.cl_int * ((long) part_size));
        var p_data = cl_new_buffer(part_buf_size);
        var src_data = Pointer.to(input_data);
        var src_part = Pointer.to(p_data);
        var src_n = Pointer.to(new int[]{n});

        Kernel.scan_bounds_multi_block
            .set_arg(ScanBoundsMultiBlock_k.Args.bounds_bank_data, src_data)
            .new_arg(ScanBoundsMultiBlock_k.Args.buffer, local_buffer_size)
            .set_arg(ScanBoundsMultiBlock_k.Args.part, src_part)
            .set_arg(ScanBoundsMultiBlock_k.Args.n, src_n)
            .call(global_work_size, local_work_default);

        scan_int(p_data, part_size);

        cl_zero_buffer(counter_buffer, Sizeof.cl_int);

        Kernel.complete_bounds_multi_block
            .set_arg(CompleteBoundsMultiBlock_k.Args.bounds_bank_data, src_data)
            .set_arg(CompleteBoundsMultiBlock_k.Args.sz, counter_pointer)
            .new_arg(CompleteBoundsMultiBlock_k.Args.buffer, local_buffer_size)
            .set_arg(CompleteBoundsMultiBlock_k.Args.part, src_part)
            .set_arg(CompleteBoundsMultiBlock_k.Args.n, src_n)
            .call(global_work_size, local_work_default);

        clReleaseMemObject(p_data);

        return cl_read_pinned_int(counter_buffer);
    }

    //#endregion

    //#region Misc. Public API

    public static cl_program gpu_p(List<String> src_strings)
    {
        String[] src = src_strings.toArray(new String[]{});
        return CLUtils.cl_p(context, device_ids, src);
    }

    public static cl_mem new_mutable_buffer(long size, Pointer src)
    {
        return cl_new_cpu_copy_buffer(size, src);
    }

    public static cl_mem new_empty_buffer(long size)
    {
        var new_buffer = cl_new_buffer(size);
        cl_zero_buffer(new_buffer, size);
        return new_buffer;
    }

    public static void clear_buffer(cl_mem mem, long size)
    {
        cl_zero_buffer(mem, size);
    }

    public static void release_buffer(cl_mem mem)
    {
        clReleaseMemObject(mem);
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
        for (Buffer buffer : Buffer.values())
        {
            if (buffer.memory != null) buffer.memory.release();
        }

        for (Program program : Program.values())
        {
            if (program.gpu != null) program.gpu.destroy();
        }

        for (cl_mem clMem : shared_mem.values())
        {
            clReleaseMemObject(clMem);
        }

        clReleaseCommandQueue(command_queue);
        clReleaseContext(context);
    }

    //#endregion
}
