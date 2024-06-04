package com.controllerface.bvge.cl;

import com.controllerface.bvge.cl.kernels.*;
import com.controllerface.bvge.cl.programs.GPUCrud;
import com.controllerface.bvge.cl.programs.ScanDeletes;
import com.controllerface.bvge.ecs.systems.UnloadedSectorSlice;
import com.controllerface.bvge.geometry.ModelRegistry;
import com.controllerface.bvge.physics.PhysicsEntityBatch;
import com.controllerface.bvge.physics.PhysicsObjects;
import com.controllerface.bvge.util.Constants;

import java.util.concurrent.BrokenBarrierException;
import java.util.concurrent.CyclicBarrier;

import static com.controllerface.bvge.cl.CLUtils.*;
import static com.controllerface.bvge.geometry.ModelRegistry.*;
import static org.lwjgl.opencl.CL10.clFinish;

public class GPUCoreMemory implements WorldContainer
{
    private final GPUProgram p_gpu_crud = new GPUCrud();
    private final GPUProgram p_scan_deletes = new ScanDeletes();

    private final GPUKernel k_compact_armature_bones;
    private final GPUKernel k_compact_edges;
    private final GPUKernel k_compact_entities;
    private final GPUKernel k_compact_hull_bones;
    private final GPUKernel k_compact_hulls;
    private final GPUKernel k_compact_points;
    private final GPUKernel k_complete_deletes_multi_block_out;
    private final GPUKernel k_count_egress_entities;
    private final GPUKernel k_create_animation_timings;
    private final GPUKernel k_create_armature_bone;
    private final GPUKernel k_create_bone;
    private final GPUKernel k_create_bone_bind_pose;
    private final GPUKernel k_create_bone_channel;
    private final GPUKernel k_create_bone_reference;
    private final GPUKernel k_create_edge;
    private final GPUKernel k_create_entity;
    private final GPUKernel k_create_hull;
    private final GPUKernel k_create_keyframe;
    private final GPUKernel k_create_mesh_face;
    private final GPUKernel k_create_mesh_reference;
    private final GPUKernel k_create_model_transform;
    private final GPUKernel k_create_point;
    private final GPUKernel k_create_texture_uv;
    private final GPUKernel k_create_vertex_reference;
    private final GPUKernel k_locate_out_of_bounds;
    private final GPUKernel k_read_position;
    private final GPUKernel k_scan_deletes_multi_block_out;
    private final GPUKernel k_scan_deletes_single_block_out;
    private final GPUKernel k_set_bone_channel_table;
    private final GPUKernel k_update_accel;
    private final GPUKernel k_update_mouse_position;

    // Bookkeeping buffers

    //#region Compaction/Shift Buffers

    /**
     * During the entity compaction process, these buffers are written to, and store the number of
     * positions that the corresponding values must shift left within their own buffers when the
     * buffer compaction occurs. Each index is aligned with the corresponding data type
     * that will be shifted. I.e. every bone in the bone buffer has a corresponding entry in the
     * bone shift buffer. Points, edges, and hulls work the same way.
     */

    private final ResizableBuffer b_armature_bone_shift;
    private final ResizableBuffer b_hull_bone_shift;
    private final ResizableBuffer b_edge_shift;
    private final ResizableBuffer b_hull_shift;
    private final ResizableBuffer b_point_shift;

    /**
     * During the deletion process, these buffers are used during the parallel scan of the relevant data
     * buffers. The partial buffers are utilized when the parallel scan occurs over multiple scan blocks,
     * and allows the output of each block to then itself be scanned, until all values have been summed.
     */

    private final ResizableBuffer b_delete_1;
    private final ResizableBuffer b_delete_2;
    private final ResizableBuffer b_delete_partial_1;
    private final ResizableBuffer b_delete_partial_2;

    //#endregion

    //#region Mirror Buffers

    /**
     * Mirror buffers are configured only for certain core buffers, and are used solely for rendering purposes.
     * Between physics simulation ticks, rendering threads use the mirror buffers to render the state of the objects
     * while the physics thread is busy calculating the data for the next frame.
     */
    private final ResizableBuffer mb_entity;
    private final ResizableBuffer mb_entity_flag;
    private final ResizableBuffer mb_entity_model_id;
    private final ResizableBuffer mb_entity_root_hull;
    private final ResizableBuffer mb_edge;
    private final ResizableBuffer mb_edge_flag;
    private final ResizableBuffer mb_hull;
    private final ResizableBuffer mb_hull_aabb;
    private final ResizableBuffer mb_hull_flag;
    private final ResizableBuffer mb_hull_entity_id;
    private final ResizableBuffer mb_hull_mesh_id;
    private final ResizableBuffer mb_hull_uv_offset;
    private final ResizableBuffer mb_hull_integrity;
    private final ResizableBuffer mb_hull_point_table;
    private final ResizableBuffer mb_hull_rotation;
    private final ResizableBuffer mb_hull_scale;
    private final ResizableBuffer mb_mirror_point;
    private final ResizableBuffer mb_point_anti_gravity;
    private final ResizableBuffer mb_point_hit_count;
    private final ResizableBuffer mb_point_vertex_reference;

    //#endregion

    // external buffers

    //#region Point Buffers

    /** float
     * x: anti-gravity magnitude for each point
     */
    private final ResizableBuffer b_point_anti_gravity;

    /** int4
     * x: bone 1 instance id
     * y: bone 2 instance id
     * z: bone 3 instance id
     * w: bone 4 instance id
     */
    private final ResizableBuffer b_point_bone_table;

    /** float4
     * x: current x position
     * y: current y position
     * z: previous x position
     * w: previous y position
     */
    private final ResizableBuffer b_point;

    /** int
     * x: reference vertex index
     */
    private final ResizableBuffer b_point_vertex_reference;

    /** int
     * x: hull index
     */
    private final ResizableBuffer b_point_hull_index;

    /** int
     * x: vertex flags (bit field)
     */
    private final ResizableBuffer b_point_flag;

    /** ushort
     * x: recent collision hit counter
     */
    private final ResizableBuffer b_point_hit_count;

    //#endregion

    //#region Edge Buffers

    /** int2
     * x: point 1 index
     * y: point 2 index
     */
    private final ResizableBuffer b_edge;

    /** int
     * x: edge flags (bit-field)
     */
    private final ResizableBuffer b_edge_flag;

    /** float
     * x: edge constraint length
     */
    private final ResizableBuffer b_edge_length;

    //#endregion

    //#region Hull Buffers

    /** float4
     * x: corner x position
     * y: corner y position
     * z: width
     * w: height
     */
    private final ResizableBuffer b_hull_aabb;

    /** int4
     * x: minimum x key index
     * y: maximum x key index
     * z: minimum y key index
     * w: maximum y key index
     */
    private final ResizableBuffer b_hull_aabb_index;

    /** int2
     * x: key bank offset
     * y: key bank size
     */
    private final ResizableBuffer b_hull_aabb_key;

    /** float4
     * x: current x position
     * y: current y position
     * z: previous x position
     * w: previous y position
     */
    private final ResizableBuffer b_hull;

    /** float2
     * x: scale x
     * y: scale y
     */
    private final ResizableBuffer b_hull_scale;

    /** int2
     * x: start point index
     * y: end point index
     */
    private final ResizableBuffer b_hull_point_table;

    /** int2
     * x: start edge index
     * y: end edge index
     */
    private final ResizableBuffer b_hull_edge_table;

    /** int
     * x: hull flags (bit-field)
     */
    private final ResizableBuffer b_hull_flag;

    /** int
     * x: entity id for aligned hull
     */
    private final ResizableBuffer b_hull_entity_id;

    /** int2
     * x: start bone
     * y: end bone
     */
    private final ResizableBuffer b_hull_bone_table;

    /** float
     * x: friction coefficient
     */
    private final ResizableBuffer b_hull_friction;

    /** float
     * x: restitution coefficient
     */
    private final ResizableBuffer b_hull_restitution;

    /** int
     * x: reference mesh id
     */
    private final ResizableBuffer b_hull_mesh_id;

    /** int
     * x: offset index of the UV to use for this hull
     */
    private final ResizableBuffer b_hull_uv_offset;

    /** float2
     * x: initial reference angle
     * y: current rotation
     */
    private final ResizableBuffer b_hull_rotation;

    /** int
     * x: the integrity (i.e. health) of the hull
     */
    private final ResizableBuffer b_hull_integrity;

    //#endregion

    //#region Entity Buffers

    /** float2
     * x: current x acceleration
     * y: current y acceleration
     */
    private final ResizableBuffer b_entity_accel;

    /** float2
     * x: the last rendered timestamp of the current animation
     * y: the last rendered timestamp of the previous animation
     */
    private final ResizableBuffer b_entity_anim_elapsed;

    /** float2
     * x: the initial time of the current blend operation
     * y: the remaining time of the current blend operation
     */
    private final ResizableBuffer b_entity_anim_blend;

    /** short2
     * x: number of ticks moving downward
     * y: number of ticks moving upward
     */
    private final ResizableBuffer b_entity_motion_state;

    /** int2
     * x: the currently running animation index
     * y: the previously running animation index
     */
    private final ResizableBuffer b_entity_anim_index;

    /** float4
     * x: current x position
     * y: current y position
     * z: previous x position
     * w: previous y position
     */
    private final ResizableBuffer b_entity;

    /** int
     * x: entity flags (bit-field)
     */
    private final ResizableBuffer b_entity_flag;

    /** int
     * x: root hull index of the aligned entity
     */
    private final ResizableBuffer b_entity_root_hull;

    /** int
     * x: model id of the aligned entity
     */
    private final ResizableBuffer b_entity_model_id;

    /** int
     * x: model transform index of the aligned entity
     */
    private final ResizableBuffer b_entity_model_transform;

    /** int2
     * x: start hull index
     * y: end hull index
     */
    private final ResizableBuffer b_entity_hull_table;

    /** int2
     * x: start bone anim index
     * y: end bone anim index
     */
    private final ResizableBuffer b_entity_bone_table;

    /** float
     * x: mass of the entity
     */
    private final ResizableBuffer b_entity_mass;

    //#endregion

    //#region Hull Bone Buffers

    /** float16
     * s0-sF: Column-major, 4x4 transformation matrix, hull bone instance
     */
    private final ResizableBuffer b_hull_bone;

    /** int
     * x: bone bind pose index (model space)
     */
    private final ResizableBuffer b_hull_bone_bind_pose_id;

    /** int
     * x: bone inverse bind pose index (mesh-space)
     */
    private final ResizableBuffer b_hull_bone_inv_bind_pose_id;

    //#endregion

    //#region Armature Bone Buffers

    /** float16
     * s0-sF: Column-major, 4x4 transformation matrix, armature bone instance
     */
    private final ResizableBuffer b_armature_bone;

    /** int
     * x: bind pose reference id
     */
    private final ResizableBuffer b_armature_bone_reference_id;

    /** int
     * x: armature bone parent id
     */
    private final ResizableBuffer b_armature_bone_parent_id;

    //#endregion

    // reference buffers

    //#region Animation Data Buffers

    /** int2
     * x: position channel start index
     * y: position channel end index
     */
    private final ResizableBuffer b_anim_bone_pos_channel;

    /** int2
     * x: rotation channel start index
     * y: rotation channel end index
     */
    private final ResizableBuffer b_anim_bone_rot_channel;

    /** int2
     * x: scaling channel start index
     * y: scaling channel end index
     */
    private final ResizableBuffer b_anim_bone_scl_channel;

    /** float
     * x: key frame timestamp
     */
    private final ResizableBuffer b_anim_frame_time;

    /** float4
     * x: vector/quaternion x
     * y: vector/quaternion y
     * z: vector/quaternion z
     * w: vector/quaternion w
     */
    private final ResizableBuffer b_anim_key_frame;

    /** float
     * x: animation duration
     */
    private final ResizableBuffer b_anim_duration;

    /** float
     * x: animation tick rate (FPS)
     */
    private final ResizableBuffer b_anim_tick_rate;

    /** int
     * x: animation timing index
     */
    private final ResizableBuffer b_anim_timing_index;

    //#endregion

    //#region Bone Buffers

    /** int2
     * x: bone channel start index
     * y: bone channel end index
     */
    private final ResizableBuffer b_bone_anim_channel_table;

    /** float16
     * s0-sF: Column-major, 4x4 transformation matrix, mesh-space bone reference (inverse bind pose)
     */
    private final ResizableBuffer b_bone_bind_pose;

    /** float16
     * s0-sF: Column-major, 4x4 transformation matrix, model-space bone reference (bind pose)
     */
    private final ResizableBuffer b_bone_reference;

    //#endregion

    //#region Model/Mesh Buffers

    /** int4
     * x: vertex 1 index
     * y: vertex 2 index
     * z: vertex 3 index
     * w: parent reference mesh ID
     */
    private final ResizableBuffer b_mesh_face;

    /** int2
     * x: start vertex index
     * y: end vertex index
     */
    private final ResizableBuffer b_mesh_vertex_table;

    /** int2
     * z: start face index
     * w: end face index
     */
    private final ResizableBuffer b_mesh_face_table;

    /** float16
     * s0-sF: Column-major, 4x4 transformation matrix
     */
    private final ResizableBuffer b_model_transform;

    //#endregion

    //#region Vertex Buffers

    /** float2
     * x: x position
     * y: y position
     */
    private final ResizableBuffer b_vertex_reference;

    /** float2
     * x: u coordinate
     * y: v coordinate
     */
    private final ResizableBuffer b_vertex_texture_uv;

    /** int2
     * x: start UV index
     * y: end UV index
     */
    private final ResizableBuffer b_vertex_uv_table;

    /** float4
     * x: bone 1 weight
     * y: bone 2 weight
     * z: bone 3 weight
     * w: bone 4 weight
     */
    private final ResizableBuffer b_vertex_weight;

    //#endregion

    private final long ptr_delete_counter;
    private final long ptr_position_buffer;
    private final long ptr_delete_sizes;
    private final long ptr_egress_sizes;

    private int hull_index            = 0;
    private int point_index           = 0;
    private int edge_index            = 0;
    private int vertex_ref_index      = 0;
    private int bone_bind_index       = 0;
    private int bone_ref_index        = 0;
    private int hull_bone_index       = 0;
    private int model_transform_index = 0;
    private int armature_bone_index   = 0;
    private int entity_index          = 0;
    private int mesh_index            = 0;
    private int face_index            = 0;
    private int uv_index              = 0;
    private int keyframe_index        = 0;
    private int bone_channel_index    = 0;
    private int animation_index       = 0;

    /**
     * When data is stored in the mirror buffer, the index values of the core memory buffers are cached
     * here, so they can be referenced by rendering tasks, which are using a mirror buffer that may
     * differ in contents. Using these cached index values allows physics and rendering tasks to run
     * concurrently without interfering with each other.
     */
    private int last_hull_index       = 0;
    private int last_point_index      = 0;
    private int last_edge_index       = 0;
    private int last_entity_index     = 0;

    private final WorldInputBuffer incoming_world_buffer;
    private final WorldOutputBuffer outgoing_world_buffer_a;
    private final WorldOutputBuffer outgoing_world_buffer_b;
    private WorldOutputBuffer active_world_output_buffer;
    private WorldOutputBuffer inactive_world_output_buffer;


    /**
     * This barrier is used to facilitate co-operation between the sector loading thread and the main loop.
     * Each iteration, the sector loader waits on this barrier once it is done loading sectors, and then the
     * main loop does the same, tripping the barrier, which it then immediately resets.
     */
    private final CyclicBarrier sector_barrier = new CyclicBarrier(3);

    public GPUCoreMemory()
    {
        ptr_delete_counter  = GPGPU.cl_new_int_arg_buffer(new int[]{ 0 });
        ptr_position_buffer = GPGPU.cl_new_pinned_buffer(CLSize.cl_float2);
        ptr_delete_sizes    = GPGPU.cl_new_pinned_buffer(CLSize.cl_int * 6);
        ptr_egress_sizes    = GPGPU.cl_new_pinned_buffer(CLSize.cl_int * 6);

        // transients
        b_hull_shift            = new TransientBuffer(GPGPU.ptr_compute_queue, CLSize.cl_int, 10_000L);
        b_edge_shift            = new TransientBuffer(GPGPU.ptr_compute_queue, CLSize.cl_int, 24_000L);
        b_point_shift           = new TransientBuffer(GPGPU.ptr_compute_queue, CLSize.cl_int, 50_000L);
        b_hull_bone_shift       = new TransientBuffer(GPGPU.ptr_compute_queue, CLSize.cl_int, 10_000L);
        b_armature_bone_shift   = new TransientBuffer(GPGPU.ptr_compute_queue, CLSize.cl_int, 24_000L);
        b_delete_1              = new TransientBuffer(GPGPU.ptr_compute_queue, CLSize.cl_int2, 10_000L);
        b_delete_2              = new TransientBuffer(GPGPU.ptr_compute_queue, CLSize.cl_int4, 20_000L);
        b_delete_partial_1      = new TransientBuffer(GPGPU.ptr_compute_queue, CLSize.cl_int2, 10_000L);
        b_delete_partial_2      = new TransientBuffer(GPGPU.ptr_compute_queue, CLSize.cl_int4, 20_000L);

        // persistent buffers
        b_anim_bone_pos_channel      = new PersistentBuffer(GPGPU.ptr_compute_queue, CLSize.cl_int2);
        b_anim_bone_rot_channel      = new PersistentBuffer(GPGPU.ptr_compute_queue, CLSize.cl_int2);
        b_anim_bone_scl_channel      = new PersistentBuffer(GPGPU.ptr_compute_queue, CLSize.cl_int2);
        b_anim_frame_time            = new PersistentBuffer(GPGPU.ptr_compute_queue, CLSize.cl_float);
        b_anim_key_frame             = new PersistentBuffer(GPGPU.ptr_compute_queue, CLSize.cl_float4);
        b_anim_duration              = new PersistentBuffer(GPGPU.ptr_compute_queue, CLSize.cl_float);
        b_anim_tick_rate             = new PersistentBuffer(GPGPU.ptr_compute_queue, CLSize.cl_float);
        b_anim_timing_index          = new PersistentBuffer(GPGPU.ptr_compute_queue, CLSize.cl_int);
        b_entity_accel               = new PersistentBuffer(GPGPU.ptr_compute_queue, CLSize.cl_float2, 10_000L);
        b_entity_anim_elapsed        = new PersistentBuffer(GPGPU.ptr_compute_queue, CLSize.cl_float2, 10_000L);
        b_entity_anim_blend          = new PersistentBuffer(GPGPU.ptr_compute_queue, CLSize.cl_float2, 10_000L);
        b_entity_motion_state        = new PersistentBuffer(GPGPU.ptr_compute_queue, CLSize.cl_short2, 10_000L);
        b_entity_anim_index          = new PersistentBuffer(GPGPU.ptr_compute_queue, CLSize.cl_int2, 10_000L);
        b_armature_bone              = new PersistentBuffer(GPGPU.ptr_compute_queue, CLSize.cl_float16);
        b_armature_bone_reference_id = new PersistentBuffer(GPGPU.ptr_compute_queue, CLSize.cl_int);
        b_armature_bone_parent_id    = new PersistentBuffer(GPGPU.ptr_compute_queue, CLSize.cl_int);
        b_entity                     = new PersistentBuffer(GPGPU.ptr_compute_queue, CLSize.cl_float4, 10_000L);
        b_entity_flag                = new PersistentBuffer(GPGPU.ptr_compute_queue, CLSize.cl_int, 10_000L);
        b_entity_root_hull           = new PersistentBuffer(GPGPU.ptr_compute_queue, CLSize.cl_int, 10_000L);
        b_entity_model_id            = new PersistentBuffer(GPGPU.ptr_compute_queue, CLSize.cl_int, 10_000L);
        b_entity_model_transform     = new PersistentBuffer(GPGPU.ptr_compute_queue, CLSize.cl_int, 10_000L);
        b_entity_hull_table          = new PersistentBuffer(GPGPU.ptr_compute_queue, CLSize.cl_int2, 10_000L);
        b_entity_bone_table          = new PersistentBuffer(GPGPU.ptr_compute_queue, CLSize.cl_int2, 10_000L);
        b_entity_mass                = new PersistentBuffer(GPGPU.ptr_compute_queue, CLSize.cl_float, 10_000L);
        b_bone_anim_channel_table    = new PersistentBuffer(GPGPU.ptr_compute_queue, CLSize.cl_int2);
        b_bone_bind_pose             = new PersistentBuffer(GPGPU.ptr_compute_queue, CLSize.cl_float16);
        b_bone_reference             = new PersistentBuffer(GPGPU.ptr_compute_queue, CLSize.cl_float16);
        b_edge                       = new PersistentBuffer(GPGPU.ptr_compute_queue, CLSize.cl_int2, 24_000L);
        b_edge_flag                  = new PersistentBuffer(GPGPU.ptr_compute_queue, CLSize.cl_int, 24_000L);
        b_edge_length                = new PersistentBuffer(GPGPU.ptr_compute_queue, CLSize.cl_float, 24_000L);
        b_hull_aabb                  = new PersistentBuffer(GPGPU.ptr_compute_queue, CLSize.cl_float4, 10_000L);
        b_hull_aabb_index            = new PersistentBuffer(GPGPU.ptr_compute_queue, CLSize.cl_int4, 10_000L);
        b_hull_aabb_key              = new PersistentBuffer(GPGPU.ptr_compute_queue, CLSize.cl_int2, 10_000L);
        b_hull_bone                  = new PersistentBuffer(GPGPU.ptr_compute_queue, CLSize.cl_float16, 10_000L);
        b_hull_bone_bind_pose_id     = new PersistentBuffer(GPGPU.ptr_compute_queue, CLSize.cl_int, 10_000L);
        b_hull_bone_inv_bind_pose_id = new PersistentBuffer(GPGPU.ptr_compute_queue, CLSize.cl_int, 10_000L);
        b_hull                       = new PersistentBuffer(GPGPU.ptr_compute_queue, CLSize.cl_float4, 10_000L);
        b_hull_scale                 = new PersistentBuffer(GPGPU.ptr_compute_queue, CLSize.cl_float2, 10_000L);
        b_hull_point_table           = new PersistentBuffer(GPGPU.ptr_compute_queue, CLSize.cl_int2, 10_000L);
        b_hull_edge_table            = new PersistentBuffer(GPGPU.ptr_compute_queue, CLSize.cl_int2, 10_000L);
        b_hull_flag                  = new PersistentBuffer(GPGPU.ptr_compute_queue, CLSize.cl_int, 10_000L);
        b_hull_bone_table            = new PersistentBuffer(GPGPU.ptr_compute_queue, CLSize.cl_int2, 10_000L);
        b_hull_entity_id             = new PersistentBuffer(GPGPU.ptr_compute_queue, CLSize.cl_int, 10_000L);
        b_hull_friction              = new PersistentBuffer(GPGPU.ptr_compute_queue, CLSize.cl_float, 10_000L);
        b_hull_restitution           = new PersistentBuffer(GPGPU.ptr_compute_queue, CLSize.cl_float, 10_000L);
        b_hull_mesh_id               = new PersistentBuffer(GPGPU.ptr_compute_queue, CLSize.cl_int, 10_000L);
        b_hull_uv_offset             = new PersistentBuffer(GPGPU.ptr_compute_queue, CLSize.cl_int, 10_000L);
        b_hull_rotation              = new PersistentBuffer(GPGPU.ptr_compute_queue, CLSize.cl_float2, 10_000L);
        b_hull_integrity             = new PersistentBuffer(GPGPU.ptr_compute_queue, CLSize.cl_int, 10_000L);
        b_mesh_face                  = new PersistentBuffer(GPGPU.ptr_compute_queue, CLSize.cl_int4);
        b_mesh_vertex_table          = new PersistentBuffer(GPGPU.ptr_compute_queue, CLSize.cl_int2);
        b_mesh_face_table            = new PersistentBuffer(GPGPU.ptr_compute_queue, CLSize.cl_int2);
        b_model_transform            = new PersistentBuffer(GPGPU.ptr_compute_queue, CLSize.cl_float16);
        b_point_anti_gravity         = new PersistentBuffer(GPGPU.ptr_compute_queue, CLSize.cl_float, 50_000L);
        b_point_bone_table           = new PersistentBuffer(GPGPU.ptr_compute_queue, CLSize.cl_int4, 50_000L);
        b_point                      = new PersistentBuffer(GPGPU.ptr_compute_queue, CLSize.cl_float4, 50_000L);
        b_point_vertex_reference     = new PersistentBuffer(GPGPU.ptr_compute_queue, CLSize.cl_int, 50_000L);
        b_point_hull_index           = new PersistentBuffer(GPGPU.ptr_compute_queue, CLSize.cl_int, 50_000L);
        b_point_flag                 = new PersistentBuffer(GPGPU.ptr_compute_queue, CLSize.cl_int, 50_000L);
        b_point_hit_count            = new PersistentBuffer(GPGPU.ptr_compute_queue, CLSize.cl_short, 50_000L);
        b_vertex_reference           = new PersistentBuffer(GPGPU.ptr_compute_queue, CLSize.cl_float2);
        b_vertex_texture_uv          = new PersistentBuffer(GPGPU.ptr_compute_queue, CLSize.cl_float2);
        b_vertex_uv_table            = new PersistentBuffer(GPGPU.ptr_compute_queue, CLSize.cl_int2);
        b_vertex_weight              = new PersistentBuffer(GPGPU.ptr_compute_queue, CLSize.cl_float4);

        // mirrors
        mb_entity                 = new PersistentBuffer(GPGPU.ptr_compute_queue, CLSize.cl_float4, 10_000L);
        mb_entity_flag            = new PersistentBuffer(GPGPU.ptr_compute_queue, CLSize.cl_int, 10_000L);
        mb_entity_model_id        = new PersistentBuffer(GPGPU.ptr_compute_queue, CLSize.cl_int, 10_000L);
        mb_entity_root_hull       = new PersistentBuffer(GPGPU.ptr_compute_queue, CLSize.cl_int, 10_000L);
        mb_edge                   = new PersistentBuffer(GPGPU.ptr_compute_queue, CLSize.cl_int2, 24_000L);
        mb_edge_flag              = new PersistentBuffer(GPGPU.ptr_compute_queue, CLSize.cl_int, 24_000L);
        mb_hull                   = new PersistentBuffer(GPGPU.ptr_compute_queue, CLSize.cl_float4, 10_000L);
        mb_hull_aabb              = new PersistentBuffer(GPGPU.ptr_compute_queue, CLSize.cl_float4, 10_000L);
        mb_hull_flag              = new PersistentBuffer(GPGPU.ptr_compute_queue, CLSize.cl_int, 10_000L);
        mb_hull_entity_id         = new PersistentBuffer(GPGPU.ptr_compute_queue, CLSize.cl_int, 10_000L);
        mb_hull_mesh_id           = new PersistentBuffer(GPGPU.ptr_compute_queue, CLSize.cl_int, 10_000L);
        mb_hull_uv_offset         = new PersistentBuffer(GPGPU.ptr_compute_queue, CLSize.cl_int, 10_000L);
        mb_hull_integrity         = new PersistentBuffer(GPGPU.ptr_compute_queue, CLSize.cl_int, 10_000L);
        mb_hull_point_table       = new PersistentBuffer(GPGPU.ptr_compute_queue, CLSize.cl_int2, 10_000L);
        mb_hull_rotation          = new PersistentBuffer(GPGPU.ptr_compute_queue, CLSize.cl_float2, 10_000L);
        mb_hull_scale             = new PersistentBuffer(GPGPU.ptr_compute_queue, CLSize.cl_float2, 10_000L);
        mb_point_hit_count        = new PersistentBuffer(GPGPU.ptr_compute_queue, CLSize.cl_short, 50_000L);
        mb_point_anti_gravity     = new PersistentBuffer(GPGPU.ptr_compute_queue, CLSize.cl_float, 50_000L);
        mb_mirror_point           = new PersistentBuffer(GPGPU.ptr_compute_queue, CLSize.cl_float4, 50_000L);
        mb_point_vertex_reference = new PersistentBuffer(GPGPU.ptr_compute_queue, CLSize.cl_int, 50_000L);

        p_gpu_crud.init();
        p_scan_deletes.init();

        // create methods

        long k_ptr_create_point = p_gpu_crud.kernel_ptr(Kernel.create_point);
        k_create_point = new CreatePoint_k(GPGPU.ptr_compute_queue, k_ptr_create_point)
            .buf_arg(CreatePoint_k.Args.points,                  b_point)
            .buf_arg(CreatePoint_k.Args.point_vertex_references, b_point_vertex_reference)
            .buf_arg(CreatePoint_k.Args.point_hull_indices,      b_point_hull_index)
            .buf_arg(CreatePoint_k.Args.point_hit_counts,        b_point_hit_count)
            .buf_arg(CreatePoint_k.Args.point_flags,             b_point_flag)
            .buf_arg(CreatePoint_k.Args.bone_tables,             b_point_bone_table);

        long k_ptr_create_texture_uv = p_gpu_crud.kernel_ptr(Kernel.create_texture_uv);
        k_create_texture_uv = new CreateTextureUV_k(GPGPU.ptr_compute_queue, k_ptr_create_texture_uv)
            .buf_arg(CreateTextureUV_k.Args.texture_uvs, b_vertex_texture_uv);

        long k_ptr_create_edge = p_gpu_crud.kernel_ptr(Kernel.create_edge);
        k_create_edge = new CreateEdge_k(GPGPU.ptr_compute_queue, k_ptr_create_edge)
            .buf_arg(CreateEdge_k.Args.edges,        b_edge)
            .buf_arg(CreateEdge_k.Args.edge_lengths, b_edge_length)
            .buf_arg(CreateEdge_k.Args.edge_flags,   b_edge_flag);

        long k_ptr_create_keyframe = p_gpu_crud.kernel_ptr(Kernel.create_keyframe);
        k_create_keyframe = new CreateKeyFrame_k(GPGPU.ptr_compute_queue, k_ptr_create_keyframe)
            .buf_arg(CreateKeyFrame_k.Args.key_frames,  b_anim_key_frame)
            .buf_arg(CreateKeyFrame_k.Args.frame_times, b_anim_frame_time);

        long k_ptr_create_vertex_reference = p_gpu_crud.kernel_ptr(Kernel.create_vertex_reference);
        k_create_vertex_reference = new CreateVertexRef_k(GPGPU.ptr_compute_queue, k_ptr_create_vertex_reference)
            .buf_arg(CreateVertexRef_k.Args.vertex_references, b_vertex_reference)
            .buf_arg(CreateVertexRef_k.Args.vertex_weights,    b_vertex_weight)
            .buf_arg(CreateVertexRef_k.Args.uv_tables,         b_vertex_uv_table);

        long k_ptr_create_bone_bind_pose = p_gpu_crud.kernel_ptr(Kernel.create_bone_bind_pose);
        k_create_bone_bind_pose = new CreateBoneBindPose_k(GPGPU.ptr_compute_queue, k_ptr_create_bone_bind_pose)
            .buf_arg(CreateBoneBindPose_k.Args.bone_bind_poses, b_bone_bind_pose);

        long k_ptr_create_bone_reference = p_gpu_crud.kernel_ptr(Kernel.create_bone_reference);
        k_create_bone_reference = new CreateBoneRef_k(GPGPU.ptr_compute_queue, k_ptr_create_bone_reference)
            .buf_arg(CreateBoneRef_k.Args.bone_references, b_bone_reference);

        long k_ptr_create_bone_channel = p_gpu_crud.kernel_ptr(Kernel.create_bone_channel);
        k_create_bone_channel = new CreateBoneChannel_k(GPGPU.ptr_compute_queue, k_ptr_create_bone_channel)
            .buf_arg(CreateBoneChannel_k.Args.animation_timing_indices, b_anim_timing_index)
            .buf_arg(CreateBoneChannel_k.Args.bone_pos_channel_tables,  b_anim_bone_pos_channel)
            .buf_arg(CreateBoneChannel_k.Args.bone_rot_channel_tables,  b_anim_bone_rot_channel)
            .buf_arg(CreateBoneChannel_k.Args.bone_scl_channel_tables,  b_anim_bone_scl_channel);

        long k_ptr_create_entity = p_gpu_crud.kernel_ptr(Kernel.create_entity);
        k_create_entity = new CreateEntity_k(GPGPU.ptr_compute_queue, k_ptr_create_entity)
            .buf_arg(CreateEntity_k.Args.entities,                 b_entity)
            .buf_arg(CreateEntity_k.Args.entity_root_hulls,        b_entity_root_hull)
            .buf_arg(CreateEntity_k.Args.entity_model_indices,     b_entity_model_id)
            .buf_arg(CreateEntity_k.Args.entity_model_transforms,  b_entity_model_transform)
            .buf_arg(CreateEntity_k.Args.entity_flags,             b_entity_flag)
            .buf_arg(CreateEntity_k.Args.entity_hull_tables,       b_entity_hull_table)
            .buf_arg(CreateEntity_k.Args.entity_bone_tables,       b_entity_bone_table)
            .buf_arg(CreateEntity_k.Args.entity_masses,            b_entity_mass)
            .buf_arg(CreateEntity_k.Args.entity_animation_indices, b_entity_anim_index)
            .buf_arg(CreateEntity_k.Args.entity_animation_elapsed, b_entity_anim_elapsed)
            .buf_arg(CreateEntity_k.Args.entity_motion_states,     b_entity_motion_state);

        long k_ptr_create_bone = p_gpu_crud.kernel_ptr(Kernel.create_hull_bone);
        k_create_bone = new CreateHullBone_k(GPGPU.ptr_compute_queue, k_ptr_create_bone)
            .buf_arg(CreateHullBone_k.Args.bones,                       b_hull_bone)
            .buf_arg(CreateHullBone_k.Args.hull_bind_pose_indicies,     b_hull_bone_bind_pose_id)
            .buf_arg(CreateHullBone_k.Args.hull_inv_bind_pose_indicies, b_hull_bone_inv_bind_pose_id);

        long k_ptr_create_armature_bone = p_gpu_crud.kernel_ptr(Kernel.create_armature_bone);
        k_create_armature_bone = new CreateArmatureBone_k(GPGPU.ptr_compute_queue, k_ptr_create_armature_bone)
            .buf_arg(CreateArmatureBone_k.Args.armature_bones,              b_armature_bone)
            .buf_arg(CreateArmatureBone_k.Args.armature_bone_reference_ids, b_armature_bone_reference_id)
            .buf_arg(CreateArmatureBone_k.Args.armature_bone_parent_ids,    b_armature_bone_parent_id);

        long k_ptr_create_model_transform = p_gpu_crud.kernel_ptr(Kernel.create_model_transform);
        k_create_model_transform = new CreateModelTransform_k(GPGPU.ptr_compute_queue, k_ptr_create_model_transform)
            .buf_arg(CreateModelTransform_k.Args.model_transforms, b_model_transform);

        long k_ptr_create_hull = p_gpu_crud.kernel_ptr(Kernel.create_hull);
        k_create_hull = new CreateHull_k(GPGPU.ptr_compute_queue, k_ptr_create_hull)
            .buf_arg(CreateHull_k.Args.hulls,             b_hull)
            .buf_arg(CreateHull_k.Args.hull_scales,       b_hull_scale)
            .buf_arg(CreateHull_k.Args.hull_rotations,    b_hull_rotation)
            .buf_arg(CreateHull_k.Args.hull_frictions,    b_hull_friction)
            .buf_arg(CreateHull_k.Args.hull_restitutions, b_hull_restitution)
            .buf_arg(CreateHull_k.Args.hull_point_tables, b_hull_point_table)
            .buf_arg(CreateHull_k.Args.hull_edge_tables,  b_hull_edge_table)
            .buf_arg(CreateHull_k.Args.hull_bone_tables,  b_hull_bone_table)
            .buf_arg(CreateHull_k.Args.hull_entity_ids,   b_hull_entity_id)
            .buf_arg(CreateHull_k.Args.hull_flags,        b_hull_flag)
            .buf_arg(CreateHull_k.Args.hull_mesh_ids,     b_hull_mesh_id)
            .buf_arg(CreateHull_k.Args.hull_uv_offsets,   b_hull_uv_offset)
            .buf_arg(CreateHull_k.Args.hull_integrity,    b_hull_integrity);

        long k_ptr_create_mesh_reference = p_gpu_crud.kernel_ptr(Kernel.create_mesh_reference);
        k_create_mesh_reference = new CreateMeshReference_k(GPGPU.ptr_compute_queue, k_ptr_create_mesh_reference)
            .buf_arg(CreateMeshReference_k.Args.mesh_vertex_tables, b_mesh_vertex_table)
            .buf_arg(CreateMeshReference_k.Args.mesh_face_tables,   b_mesh_face_table);

        long k_ptr_create_mesh_face = p_gpu_crud.kernel_ptr(Kernel.create_mesh_face);
        k_create_mesh_face = new CreateMeshFace_k(GPGPU.ptr_compute_queue, k_ptr_create_mesh_face)
            .buf_arg(CreateMeshFace_k.Args.mesh_faces, b_mesh_face);

        long k_ptr_create_animation_timings = p_gpu_crud.kernel_ptr(Kernel.create_animation_timings);
        k_create_animation_timings = new CreateAnimationTimings_k(GPGPU.ptr_compute_queue, k_ptr_create_animation_timings)
            .buf_arg(CreateAnimationTimings_k.Args.animation_durations,  b_anim_duration)
            .buf_arg(CreateAnimationTimings_k.Args.animation_tick_rates, b_anim_tick_rate);

        // read methods

        long k_ptr_read_position = p_gpu_crud.kernel_ptr(Kernel.read_position);
        k_read_position = new ReadPosition_k(GPGPU.ptr_compute_queue, k_ptr_read_position)
            .buf_arg(ReadPosition_k.Args.entities, b_entity);

        // update methods

        long k_ptr_update_accel = p_gpu_crud.kernel_ptr(Kernel.update_accel);
        k_update_accel = new UpdateAccel_k(GPGPU.ptr_compute_queue, k_ptr_update_accel)
            .buf_arg(UpdateAccel_k.Args.entity_accel, b_entity_accel);

        long k_ptr_update_mouse_position = p_gpu_crud.kernel_ptr(Kernel.update_mouse_position);
        k_update_mouse_position = new UpdateMousePosition_k(GPGPU.ptr_compute_queue, k_ptr_update_mouse_position)
            .buf_arg(UpdateMousePosition_k.Args.entity_root_hulls, b_entity_root_hull)
            .buf_arg(UpdateMousePosition_k.Args.hull_point_tables, b_hull_point_table)
            .buf_arg(UpdateMousePosition_k.Args.points,            b_point);

        long k_ptr_set_bone_channel_table = p_gpu_crud.kernel_ptr(Kernel.set_bone_channel_table);
        k_set_bone_channel_table = new SetBoneChannelTable_k(GPGPU.ptr_compute_queue, k_ptr_set_bone_channel_table)
            .buf_arg(SetBoneChannelTable_k.Args.bone_channel_tables, b_bone_anim_channel_table);

        // delete methods

        long k_ptr_locate_out_of_bounds = p_scan_deletes.kernel_ptr(Kernel.locate_out_of_bounds);
        k_locate_out_of_bounds = new LocateOutOfBounds_k(GPGPU.ptr_compute_queue, k_ptr_locate_out_of_bounds)
            .buf_arg(LocateOutOfBounds_k.Args.entity_flags, b_entity_flag);

        long k_ptr_scan_deletes_single_block_out = p_scan_deletes.kernel_ptr(Kernel.scan_deletes_single_block_out);
        k_scan_deletes_single_block_out = new ScanDeletesSingleBlockOut_k(GPGPU.ptr_compute_queue, k_ptr_scan_deletes_single_block_out)
            .ptr_arg(ScanDeletesSingleBlockOut_k.Args.sz,               ptr_delete_sizes)
            .buf_arg(ScanDeletesSingleBlockOut_k.Args.entity_flags,     b_entity_flag)
            .buf_arg(ScanDeletesSingleBlockOut_k.Args.hull_tables,      b_entity_hull_table)
            .buf_arg(ScanDeletesSingleBlockOut_k.Args.bone_tables,      b_entity_bone_table)
            .buf_arg(ScanDeletesSingleBlockOut_k.Args.point_tables,     b_hull_point_table)
            .buf_arg(ScanDeletesSingleBlockOut_k.Args.edge_tables,      b_hull_edge_table)
            .buf_arg(ScanDeletesSingleBlockOut_k.Args.hull_bone_tables, b_hull_bone_table);

        long k_ptr_scan_deletes_multi_block_out = p_scan_deletes.kernel_ptr(Kernel.scan_deletes_multi_block_out);
        k_scan_deletes_multi_block_out = new ScanDeletesMultiBlockOut_k(GPGPU.ptr_compute_queue, k_ptr_scan_deletes_multi_block_out)
            .buf_arg(ScanDeletesMultiBlockOut_k.Args.part1,            b_delete_partial_1)
            .buf_arg(ScanDeletesMultiBlockOut_k.Args.part2,            b_delete_partial_2)
            .buf_arg(ScanDeletesMultiBlockOut_k.Args.entity_flags,     b_entity_flag)
            .buf_arg(ScanDeletesMultiBlockOut_k.Args.hull_tables,      b_entity_hull_table)
            .buf_arg(ScanDeletesMultiBlockOut_k.Args.bone_tables,      b_entity_bone_table)
            .buf_arg(ScanDeletesMultiBlockOut_k.Args.point_tables,     b_hull_point_table)
            .buf_arg(ScanDeletesMultiBlockOut_k.Args.edge_tables,      b_hull_edge_table)
            .buf_arg(ScanDeletesMultiBlockOut_k.Args.hull_bone_tables, b_hull_bone_table);

        long k_ptr_complete_deletes_multi_block_out = p_scan_deletes.kernel_ptr(Kernel.complete_deletes_multi_block_out);
        k_complete_deletes_multi_block_out = new CompleteDeletesMultiBlockOut_k(GPGPU.ptr_compute_queue, k_ptr_complete_deletes_multi_block_out)
            .ptr_arg(CompleteDeletesMultiBlockOut_k.Args.sz,               ptr_delete_sizes)
            .buf_arg(CompleteDeletesMultiBlockOut_k.Args.part1,            b_delete_partial_1)
            .buf_arg(CompleteDeletesMultiBlockOut_k.Args.part2,            b_delete_partial_2)
            .buf_arg(CompleteDeletesMultiBlockOut_k.Args.entity_flags,     b_entity_flag)
            .buf_arg(CompleteDeletesMultiBlockOut_k.Args.hull_tables,      b_entity_hull_table)
            .buf_arg(CompleteDeletesMultiBlockOut_k.Args.bone_tables,      b_entity_bone_table)
            .buf_arg(CompleteDeletesMultiBlockOut_k.Args.point_tables,     b_hull_point_table)
            .buf_arg(CompleteDeletesMultiBlockOut_k.Args.edge_tables,      b_hull_edge_table)
            .buf_arg(CompleteDeletesMultiBlockOut_k.Args.hull_bone_tables, b_hull_bone_table);

        long k_ptr_compact_entities = p_scan_deletes.kernel_ptr(Kernel.compact_entities);
        k_compact_entities = new CompactEntities_k(GPGPU.ptr_compute_queue, k_ptr_compact_entities)
            .buf_arg(CompactEntities_k.Args.entities,                  b_entity)
            .buf_arg(CompactEntities_k.Args.entity_masses,             b_entity_mass)
            .buf_arg(CompactEntities_k.Args.entity_root_hulls,         b_entity_root_hull)
            .buf_arg(CompactEntities_k.Args.entity_model_indices,      b_entity_model_id)
            .buf_arg(CompactEntities_k.Args.entity_model_transforms,   b_entity_model_transform)
            .buf_arg(CompactEntities_k.Args.entity_flags,              b_entity_flag)
            .buf_arg(CompactEntities_k.Args.entity_animation_indices,  b_entity_anim_index)
            .buf_arg(CompactEntities_k.Args.entity_animation_elapsed,  b_entity_anim_elapsed)
            .buf_arg(CompactEntities_k.Args.entity_animation_blend,    b_entity_anim_blend)
            .buf_arg(CompactEntities_k.Args.entity_motion_states,      b_entity_motion_state)
            .buf_arg(CompactEntities_k.Args.entity_entity_hull_tables, b_entity_hull_table)
            .buf_arg(CompactEntities_k.Args.entity_bone_tables,        b_entity_bone_table)
            .buf_arg(CompactEntities_k.Args.hull_bone_tables,          b_hull_bone_table)
            .buf_arg(CompactEntities_k.Args.hull_entity_ids,           b_hull_entity_id)
            .buf_arg(CompactEntities_k.Args.hull_point_tables,         b_hull_point_table)
            .buf_arg(CompactEntities_k.Args.hull_edge_tables,          b_hull_edge_table)
            .buf_arg(CompactEntities_k.Args.points,                    b_point)
            .buf_arg(CompactEntities_k.Args.point_hull_indices,        b_point_hull_index)
            .buf_arg(CompactEntities_k.Args.point_bone_tables,         b_point_bone_table)
            .buf_arg(CompactEntities_k.Args.armature_bone_parent_ids,  b_armature_bone_parent_id)
            .buf_arg(CompactEntities_k.Args.hull_bind_pose_indices,    b_hull_bone_bind_pose_id)
            .buf_arg(CompactEntities_k.Args.edges,                     b_edge)
            .buf_arg(CompactEntities_k.Args.hull_bone_shift,           b_hull_bone_shift)
            .buf_arg(CompactEntities_k.Args.point_shift,               b_point_shift)
            .buf_arg(CompactEntities_k.Args.edge_shift,                b_edge_shift)
            .buf_arg(CompactEntities_k.Args.hull_shift,                b_hull_shift)
            .buf_arg(CompactEntities_k.Args.armature_bone_shift,       b_armature_bone_shift);

        long k_ptr_compact_hulls = p_scan_deletes.kernel_ptr(Kernel.compact_hulls);
        k_compact_hulls = new CompactHulls_k(GPGPU.ptr_compute_queue, k_ptr_compact_hulls)
            .buf_arg(CompactHulls_k.Args.hull_shift,        b_hull_shift)
            .buf_arg(CompactHulls_k.Args.hulls,             b_hull)
            .buf_arg(CompactHulls_k.Args.hull_scales,       b_hull_scale)
            .buf_arg(CompactHulls_k.Args.hull_mesh_ids,     b_hull_mesh_id)
            .buf_arg(CompactHulls_k.Args.hull_uv_offsets,   b_hull_uv_offset)
            .buf_arg(CompactHulls_k.Args.hull_rotations,    b_hull_rotation)
            .buf_arg(CompactHulls_k.Args.hull_frictions,    b_hull_friction)
            .buf_arg(CompactHulls_k.Args.hull_restitutions, b_hull_restitution)
            .buf_arg(CompactHulls_k.Args.hull_integrity,    b_hull_integrity)
            .buf_arg(CompactHulls_k.Args.hull_bone_tables,  b_hull_bone_table)
            .buf_arg(CompactHulls_k.Args.hull_entity_ids,   b_hull_entity_id)
            .buf_arg(CompactHulls_k.Args.hull_flags,        b_hull_flag)
            .buf_arg(CompactHulls_k.Args.hull_point_tables, b_hull_point_table)
            .buf_arg(CompactHulls_k.Args.hull_edge_tables,  b_hull_edge_table)
            .buf_arg(CompactHulls_k.Args.bounds,            b_hull_aabb)
            .buf_arg(CompactHulls_k.Args.bounds_index_data, b_hull_aabb_index)
            .buf_arg(CompactHulls_k.Args.bounds_bank_data,  b_hull_aabb_key);

        long k_ptr_compact_edges = p_scan_deletes.kernel_ptr(Kernel.compact_edges);
        k_compact_edges = new CompactEdges_k(GPGPU.ptr_compute_queue, k_ptr_compact_edges)
            .buf_arg(CompactEdges_k.Args.edge_shift,   b_edge_shift)
            .buf_arg(CompactEdges_k.Args.edges,        b_edge)
            .buf_arg(CompactEdges_k.Args.edge_lengths, b_edge_length)
            .buf_arg(CompactEdges_k.Args.edge_flags,   b_edge_flag);

        long k_ptr_compact_points = p_scan_deletes.kernel_ptr(Kernel.compact_points);
        k_compact_points = new CompactPoints_k(GPGPU.ptr_compute_queue, k_ptr_compact_points)
            .buf_arg(CompactPoints_k.Args.point_shift,             b_point_shift)
            .buf_arg(CompactPoints_k.Args.points,                  b_point)
            .buf_arg(CompactPoints_k.Args.anti_gravity,            b_point_anti_gravity)
            .buf_arg(CompactPoints_k.Args.point_vertex_references, b_point_vertex_reference)
            .buf_arg(CompactPoints_k.Args.point_hull_indices,      b_point_hull_index)
            .buf_arg(CompactPoints_k.Args.point_flags,             b_point_flag)
            .buf_arg(CompactPoints_k.Args.point_hit_counts,        b_point_hit_count)
            .buf_arg(CompactPoints_k.Args.bone_tables,             b_point_bone_table);

        long k_ptr_compact_hull_bones = p_scan_deletes.kernel_ptr(Kernel.compact_hull_bones);
        k_compact_hull_bones = new CompactHullBones_k(GPGPU.ptr_compute_queue, k_ptr_compact_hull_bones)
            .buf_arg(CompactHullBones_k.Args.hull_bone_shift,             b_hull_bone_shift)
            .buf_arg(CompactHullBones_k.Args.bone_instances,              b_hull_bone)
            .buf_arg(CompactHullBones_k.Args.hull_bind_pose_indicies,     b_hull_bone_bind_pose_id)
            .buf_arg(CompactHullBones_k.Args.hull_inv_bind_pose_indicies, b_hull_bone_inv_bind_pose_id);

        long k_ptr_compact_armature_bones = p_scan_deletes.kernel_ptr(Kernel.compact_armature_bones);
        k_compact_armature_bones = new CompactArmatureBones_k(GPGPU.ptr_compute_queue, k_ptr_compact_armature_bones)
            .buf_arg(CompactArmatureBones_k.Args.armature_bone_shift,         b_armature_bone_shift)
            .buf_arg(CompactArmatureBones_k.Args.armature_bones,              b_armature_bone)
            .buf_arg(CompactArmatureBones_k.Args.armature_bone_reference_ids, b_armature_bone_reference_id)
            .buf_arg(CompactArmatureBones_k.Args.armature_bone_parent_ids,    b_armature_bone_parent_id);

        long k_ptr_count_egress_candidates = p_gpu_crud.kernel_ptr(Kernel.count_egress_entities);
        k_count_egress_entities = new CountEgressEntities_k(GPGPU.ptr_compute_queue, k_ptr_count_egress_candidates)
            .buf_arg(CountEgressEntities_k.Args.entity_flags,       b_entity_flag)
            .buf_arg(CountEgressEntities_k.Args.entity_hull_tables, b_entity_hull_table)
            .buf_arg(CountEgressEntities_k.Args.entity_bone_tables, b_entity_bone_table)
            .buf_arg(CountEgressEntities_k.Args.hull_point_tables,  b_hull_point_table)
            .buf_arg(CountEgressEntities_k.Args.hull_edge_tables,   b_hull_edge_table)
            .buf_arg(CountEgressEntities_k.Args.hull_bone_tables,   b_hull_bone_table)
            .ptr_arg(CountEgressEntities_k.Args.counters,           ptr_egress_sizes);

        this.incoming_world_buffer = new WorldInputBuffer(GPGPU.ptr_sector_queue, this);
        this.outgoing_world_buffer_a = new WorldOutputBuffer(GPGPU.ptr_sector_queue, this);
        this.outgoing_world_buffer_b = new WorldOutputBuffer(GPGPU.ptr_sector_queue, this);
        this.active_world_output_buffer = outgoing_world_buffer_a;
        this.inactive_world_output_buffer = outgoing_world_buffer_b;
    }

    public ResizableBuffer buffer(BufferType bufferType)
    {
        return switch (bufferType)
        {
            case ANIM_FRAME_TIME               -> b_anim_frame_time;
            case ANIM_KEY_FRAME                -> b_anim_key_frame;
            case ANIM_POS_CHANNEL              -> b_anim_bone_pos_channel;
            case ANIM_ROT_CHANNEL              -> b_anim_bone_rot_channel;
            case ANIM_SCL_CHANNEL              -> b_anim_bone_scl_channel;
            case ANIM_DURATION                 -> b_anim_duration;
            case ANIM_TICK_RATE                -> b_anim_tick_rate;
            case ANIM_TIMING_INDEX             -> b_anim_timing_index;
            case ENTITY                        -> b_entity;
            case ENTITY_ACCEL                  -> b_entity_accel;
            case ENTITY_ANIM_BLEND             -> b_entity_anim_blend;
            case ENTITY_ANIM_ELAPSED           -> b_entity_anim_elapsed;
            case ENTITY_MOTION_STATE           -> b_entity_motion_state;
            case ENTITY_ANIM_INDEX             -> b_entity_anim_index;
            case ARMATURE_BONE                 -> b_armature_bone;
            case ARMATURE_BONE_REFERENCE_ID    -> b_armature_bone_reference_id;
            case ARMATURE_BONE_PARENT_ID       -> b_armature_bone_parent_id;
            case ENTITY_FLAG                   -> b_entity_flag;
            case ENTITY_BONE_TABLE             -> b_entity_bone_table;
            case ENTITY_HULL_TABLE             -> b_entity_hull_table;
            case ENTITY_MASS                   -> b_entity_mass;
            case ENTITY_MODEL_ID               -> b_entity_model_id;
            case ENTITY_ROOT_HULL              -> b_entity_root_hull;
            case ENTITY_TRANSFORM_ID           -> b_entity_model_transform;
            case BONE_ANIM_TABLE               -> b_bone_anim_channel_table;
            case BONE_BIND_POSE                -> b_bone_bind_pose;
            case BONE_REFERENCE                -> b_bone_reference;
            case EDGE                          -> b_edge;
            case EDGE_FLAG                     -> b_edge_flag;
            case EDGE_LENGTH                   -> b_edge_length;
            case HULL                          -> b_hull;
            case HULL_SCALE                    -> b_hull_scale;
            case HULL_AABB                     -> b_hull_aabb;
            case HULL_AABB_INDEX               -> b_hull_aabb_index;
            case HULL_AABB_KEY_TABLE           -> b_hull_aabb_key;
            case HULL_BONE                     -> b_hull_bone;
            case HULL_ENTITY_ID                -> b_hull_entity_id;
            case HULL_BONE_TABLE               -> b_hull_bone_table;
            case HULL_BONE_BIND_POSE           -> b_hull_bone_bind_pose_id;
            case HULL_BONE_INV_BIND_POSE       -> b_hull_bone_inv_bind_pose_id;
            case HULL_POINT_TABLE              -> b_hull_point_table;
            case HULL_EDGE_TABLE               -> b_hull_edge_table;
            case HULL_FLAG                     -> b_hull_flag;
            case HULL_FRICTION                 -> b_hull_friction;
            case HULL_RESTITUTION              -> b_hull_restitution;
            case HULL_INTEGRITY                -> b_hull_integrity;
            case HULL_MESH_ID                  -> b_hull_mesh_id;
            case HULL_UV_OFFSET                -> b_hull_uv_offset;
            case HULL_ROTATION                 -> b_hull_rotation;
            case MESH_FACE                     -> b_mesh_face;
            case MESH_VERTEX_TABLE             -> b_mesh_vertex_table;
            case MESH_FACE_TABLE               -> b_mesh_face_table;
            case MODEL_TRANSFORM               -> b_model_transform;
            case POINT                         -> b_point;
            case POINT_ANTI_GRAV               -> b_point_anti_gravity;
            case POINT_BONE_TABLE              -> b_point_bone_table;
            case POINT_VERTEX_REFERENCE        -> b_point_vertex_reference;
            case POINT_HULL_INDEX              -> b_point_hull_index;
            case POINT_FLAG                    -> b_point_flag;
            case POINT_HIT_COUNT               -> b_point_hit_count;
            case VERTEX_REFERENCE              -> b_vertex_reference;
            case VERTEX_TEXTURE_UV             -> b_vertex_texture_uv;
            case VERTEX_UV_TABLE               -> b_vertex_uv_table;
            case VERTEX_WEIGHT                 -> b_vertex_weight;

            case MIRROR_EDGE                   -> mb_edge;
            case MIRROR_HULL                   -> mb_hull;
            case MIRROR_ENTITY                 -> mb_entity;
            case MIRROR_ENTITY_FLAG            -> mb_entity_flag;
            case MIRROR_POINT                  -> mb_mirror_point;
            case MIRROR_ENTITY_MODEL_ID        -> mb_entity_model_id;
            case MIRROR_ENTITY_ROOT_HULL       -> mb_entity_root_hull;
            case MIRROR_EDGE_FLAG              -> mb_edge_flag;
            case MIRROR_HULL_AABB              -> mb_hull_aabb;
            case MIRROR_HULL_ENTITY_ID         -> mb_hull_entity_id;
            case MIRROR_HULL_FLAG              -> mb_hull_flag;
            case MIRROR_HULL_MESH_ID           -> mb_hull_mesh_id;
            case MIRROR_HULL_UV_OFFSET         -> mb_hull_uv_offset;
            case MIRROR_HULL_INTEGRITY         -> mb_hull_integrity;
            case MIRROR_HULL_POINT_TABLE       -> mb_hull_point_table;
            case MIRROR_HULL_ROTATION          -> mb_hull_rotation;
            case MIRROR_HULL_SCALE             -> mb_hull_scale;
            case MIRROR_POINT_ANTI_GRAV        -> mb_point_anti_gravity;
            case MIRROR_POINT_HIT_COUNT        -> mb_point_hit_count;
            case MIRROR_POINT_VERTEX_REFERENCE -> mb_point_vertex_reference;
        };
    }

    public void mirror_buffers_ex()
    {
        mb_entity.mirror(b_entity);
        mb_entity_flag.mirror(b_entity_flag);
        mb_entity_model_id.mirror(b_entity_model_id);
        mb_entity_root_hull.mirror(b_entity_root_hull);
        mb_edge.mirror(b_edge);
        mb_edge_flag.mirror(b_edge_flag);
        mb_hull.mirror(b_hull);
        mb_hull_aabb.mirror(b_hull_aabb);
        mb_hull_entity_id.mirror(b_hull_entity_id);
        mb_hull_flag.mirror(b_hull_flag);
        mb_hull_mesh_id.mirror(b_hull_mesh_id);
        mb_hull_uv_offset.mirror(b_hull_uv_offset);
        mb_hull_integrity.mirror(b_hull_integrity);
        mb_hull_point_table.mirror(b_hull_point_table);
        mb_hull_rotation.mirror(b_hull_rotation);
        mb_hull_scale.mirror(b_hull_scale);
        mb_point_hit_count.mirror(b_point_hit_count);
        mb_point_anti_gravity.mirror(b_point_anti_gravity);
        mb_mirror_point.mirror(b_point);
        mb_point_vertex_reference.mirror(b_point_vertex_reference);

        last_edge_index     = edge_index;
        last_entity_index   = entity_index;
        last_hull_index     = hull_index;
        last_point_index    = point_index;
    }

    // index methods

    public int next_mesh()
    {
        return mesh_index;
    }

    public int next_entity()
    {
        return entity_index;
    }

    public int next_hull()
    {
        return hull_index;
    }

    public int next_point()
    {
        return point_index;
    }

    public int next_edge()
    {
        return edge_index;
    }

    public int next_hull_bone()
    {
        return hull_bone_index;
    }

    public int next_armature_bone()
    {
        return armature_bone_index;
    }

    public int last_point()
    {
        return last_point_index;
    }

    public int last_entity()
    {
        return last_entity_index;
    }

    public int last_hull()
    {
        return last_hull_index;
    }

    public int last_edge()
    {
        return last_edge_index;
    }


    public void swap_egress_buffers()
    {
        var t = active_world_output_buffer;
        active_world_output_buffer = inactive_world_output_buffer;
        inactive_world_output_buffer = t;

        inactive_egress_counts[0] = active_egress_counts[0];
        inactive_egress_counts[1] = active_egress_counts[1];
        inactive_egress_counts[2] = active_egress_counts[2];
        inactive_egress_counts[3] = active_egress_counts[3];
        inactive_egress_counts[4] = active_egress_counts[4];
        inactive_egress_counts[5] = active_egress_counts[5];
    }

    public void reset_sector()
    {
        sector_barrier.reset();
    }

    public void await_sector()
    {
        try
        {
            sector_barrier.await();
        }
        catch (InterruptedException _) { }
        catch (BrokenBarrierException e)
        {
            throw new RuntimeException(e);
        }
    }

    public void load_entity_batch(PhysicsEntityBatch batch)
    {

        for (var entity : batch.entities)
        {

        }


        for (var solid : batch.blocks)
        {
            if (solid.dynamic())
            {
                PhysicsObjects.base_block(incoming_world_buffer, solid.x(), solid.y(), solid.size(), solid.mass(), solid.friction(), solid.restitution(), solid.flags(), solid.material());
            }
            else
            {
                int flags = solid.flags() | Constants.HullFlags.IS_STATIC._int;
                PhysicsObjects.base_block(incoming_world_buffer, solid.x(), solid.y(), solid.size(), solid.mass(), solid.friction(), solid.restitution(), flags, solid.material());
            }
        }
        for (var shard : batch.shards)
        {
            int id = shard.spike() ? ModelRegistry.BASE_SPIKE_INDEX : BASE_SHARD_INDEX;
            PhysicsObjects.tri(incoming_world_buffer, shard.x(), shard.y(), shard.size(), shard.flags(), shard.mass(), shard.friction(), shard.restitution(), id, shard.material());
        }
        for (var liquid : batch.liquids)
        {
            PhysicsObjects.liquid_particle(incoming_world_buffer, liquid.x(), liquid.y(), liquid.size(), liquid.mass(), liquid.friction(), liquid.restitution(), liquid.flags(), liquid.point_flags(), liquid.particle_fluid());
        }
    }

    private int[] active_egress_counts = new int[6];
    private int[] inactive_egress_counts = new int[6];

    public int[] last_egress_counts()
    {
        return inactive_egress_counts;
    }

    public void clear_egress_counts()
    {
        active_egress_counts[0] = 0;
        active_egress_counts[1] = 0;
        active_egress_counts[2] = 0;
        active_egress_counts[3] = 0;
        active_egress_counts[4] = 0;
        active_egress_counts[5] = 0;
    }

    public void transfer_egress_buffer(int[] egress_counts)
    {
        active_egress_counts[0] = egress_counts[0];
        active_egress_counts[1] = egress_counts[1];
        active_egress_counts[2] = egress_counts[2];
        active_egress_counts[3] = egress_counts[3];
        active_egress_counts[4] = egress_counts[4];
        active_egress_counts[5] = egress_counts[5];

        clFinish(GPGPU.ptr_compute_queue);
        active_world_output_buffer.pull_from_parent(entity_index, egress_counts);
        clFinish(GPGPU.ptr_sector_queue);
    }

    public void transfer_world_output(UnloadedSectorSlice unloaded_sectors, int[] egress_counts)
    {
        inactive_world_output_buffer.unload_sector(unloaded_sectors, egress_counts);
        clFinish(GPGPU.ptr_sector_queue);
    }

    public void transfer_world_input()
    {
        int point_count         = incoming_world_buffer.next_point();
        int edge_count          = incoming_world_buffer.next_edge();
        int hull_count          = incoming_world_buffer.next_hull();
        int entity_count        = incoming_world_buffer.next_entity();
        int hull_bone_count     = incoming_world_buffer.next_hull_bone();
        int armature_bone_count = incoming_world_buffer.next_armature_bone();

        int total = point_count
            + edge_count
            + hull_count
            + entity_count
            + hull_bone_count
            + armature_bone_count;

        if (total == 0) return;

        int point_capacity         = point_count + next_point();
        int edge_capacity          = edge_count + next_edge();
        int hull_capacity          = hull_count + next_hull();
        int entity_capacity        = entity_count + next_entity();
        int hull_bone_capacity     = hull_bone_count + next_hull_bone();
        int armature_bone_capacity = armature_bone_count + next_armature_bone();

        b_point.ensure_capacity(point_capacity);
        b_point_anti_gravity.ensure_capacity(point_capacity);
        b_point_vertex_reference.ensure_capacity(point_capacity);
        b_point_hull_index.ensure_capacity(point_capacity);
        b_point_flag.ensure_capacity(point_capacity);
        b_point_hit_count.ensure_capacity(point_capacity);
        b_point_bone_table.ensure_capacity(point_capacity);

        b_edge.ensure_capacity(edge_capacity);
        b_edge_length.ensure_capacity(edge_capacity);
        b_edge_flag.ensure_capacity(edge_capacity);

        b_hull.ensure_capacity(hull_capacity);
        b_hull_scale.ensure_capacity(hull_capacity);
        b_hull_mesh_id.ensure_capacity(hull_capacity);
        b_hull_uv_offset.ensure_capacity(hull_capacity);
        b_hull_rotation.ensure_capacity(hull_capacity);
        b_hull_integrity.ensure_capacity(hull_capacity);
        b_hull_point_table.ensure_capacity(hull_capacity);
        b_hull_edge_table.ensure_capacity(hull_capacity);
        b_hull_flag.ensure_capacity(hull_capacity);
        b_hull_bone_table.ensure_capacity(hull_capacity);
        b_hull_entity_id.ensure_capacity(hull_capacity);
        b_hull_friction.ensure_capacity(hull_capacity);
        b_hull_restitution.ensure_capacity(hull_capacity);
        b_hull_aabb.ensure_capacity(hull_capacity);
        b_hull_aabb_index.ensure_capacity(hull_capacity);
        b_hull_aabb_key.ensure_capacity(hull_capacity);

        b_entity.ensure_capacity(entity_capacity);
        b_entity_flag.ensure_capacity(entity_capacity);
        b_entity_root_hull.ensure_capacity(entity_capacity);
        b_entity_model_id.ensure_capacity(entity_capacity);
        b_entity_model_transform.ensure_capacity(entity_capacity);
        b_entity_accel.ensure_capacity(entity_capacity);
        b_entity_mass.ensure_capacity(entity_capacity);
        b_entity_anim_index.ensure_capacity(entity_capacity);
        b_entity_anim_elapsed.ensure_capacity(entity_capacity);
        b_entity_anim_blend.ensure_capacity(entity_capacity);
        b_entity_motion_state.ensure_capacity(entity_capacity);
        b_entity_hull_table.ensure_capacity(entity_capacity);
        b_entity_bone_table.ensure_capacity(entity_capacity);

        b_hull_bone.ensure_capacity(hull_bone_capacity);
        b_hull_bone_bind_pose_id.ensure_capacity(hull_bone_capacity);
        b_hull_bone_inv_bind_pose_id.ensure_capacity(hull_bone_capacity);

        b_armature_bone.ensure_capacity(armature_bone_capacity);
        b_armature_bone_reference_id.ensure_capacity(armature_bone_capacity);
        b_armature_bone_parent_id.ensure_capacity(armature_bone_capacity);

        clFinish(GPGPU.ptr_compute_queue);
        incoming_world_buffer.merge_into_parent(this);
        clFinish(GPGPU.ptr_sector_queue);

        point_index += point_count;
        edge_index += edge_count;
        hull_index += hull_count;
        entity_index += entity_count;
        hull_bone_index += hull_bone_count;
        armature_bone_index += armature_bone_count;
    }

    public int new_animation_timings(float duration, float tick_rate)
    {
        int capacity = animation_index + 1;

        b_anim_duration.ensure_capacity(capacity);
        b_anim_tick_rate.ensure_capacity(capacity);

        k_create_animation_timings
            .set_arg(CreateAnimationTimings_k.Args.target, animation_index)
            .set_arg(CreateAnimationTimings_k.Args.new_animation_duration, duration)
            .set_arg(CreateAnimationTimings_k.Args.new_animation_tick_rate, tick_rate)
            .call(GPGPU.global_single_size);

        return animation_index++;
    }

    public int new_bone_channel(int anim_timing_index, int[] pos_table, int[] rot_table, int[] scl_table)
    {
        int capacity = bone_channel_index + 1;
        b_anim_timing_index.ensure_capacity(capacity);
        b_anim_bone_pos_channel.ensure_capacity(capacity);
        b_anim_bone_rot_channel.ensure_capacity(capacity);
        b_anim_bone_scl_channel.ensure_capacity(capacity);

        k_create_bone_channel
            .set_arg(CreateBoneChannel_k.Args.target, bone_channel_index)
            .set_arg(CreateBoneChannel_k.Args.new_animation_timing_index, anim_timing_index)
            .set_arg(CreateBoneChannel_k.Args.new_bone_pos_channel_table, pos_table)
            .set_arg(CreateBoneChannel_k.Args.new_bone_rot_channel_table, rot_table)
            .set_arg(CreateBoneChannel_k.Args.new_bone_scl_channel_table, scl_table)
            .call(GPGPU.global_single_size);

        return bone_channel_index++;
    }

    public int new_keyframe(float[] frame, float time)
    {
        int capacity = keyframe_index + 1;
        b_anim_key_frame.ensure_capacity(capacity);
        b_anim_frame_time.ensure_capacity(capacity);

        k_create_keyframe
            .set_arg(CreateKeyFrame_k.Args.target, keyframe_index)
            .set_arg(CreateKeyFrame_k.Args.new_keyframe, frame)
            .set_arg(CreateKeyFrame_k.Args.new_frame_time, time)
            .call(GPGPU.global_single_size);

        return keyframe_index++;
    }

    public int new_texture_uv(float u, float v)
    {
        int capacity = uv_index + 1;
        b_vertex_texture_uv.ensure_capacity(capacity);

        k_create_texture_uv
            .set_arg(CreateTextureUV_k.Args.target, uv_index)
            .set_arg(CreateTextureUV_k.Args.new_texture_uv, arg_float2(u, v))
            .call(GPGPU.global_single_size);

        return uv_index++;
    }

    public int new_edge(int p1, int p2, float l, int flags)
    {
        int required_capacity = edge_index + 1;
        b_edge.ensure_capacity(required_capacity);
        b_edge_length.ensure_capacity(required_capacity);
        b_edge_flag.ensure_capacity(required_capacity);

        k_create_edge
            .set_arg(CreateEdge_k.Args.target, edge_index)
            .set_arg(CreateEdge_k.Args.new_edge, arg_int2(p1, p2))
            .set_arg(CreateEdge_k.Args.new_edge_length, l)
            .set_arg(CreateEdge_k.Args.new_edge_flag, flags)
            .call(GPGPU.global_single_size);

        return edge_index++;
    }

    public int new_point(float[] position, int[] bone_ids, int vertex_index, int hull_index, int hit_count, int flags)
    {
        int capacity = point_index + 1;
        b_point.ensure_capacity(capacity);
        b_point_anti_gravity.ensure_capacity(capacity);
        b_point_vertex_reference.ensure_capacity(capacity);
        b_point_hull_index.ensure_capacity(capacity);
        b_point_flag.ensure_capacity(capacity);
        b_point_hit_count.ensure_capacity(capacity);
        b_point_bone_table.ensure_capacity(capacity);

        var new_point = new float[]{position[0], position[1], position[0], position[1]};
        k_create_point
            .set_arg(CreatePoint_k.Args.target, point_index)
            .set_arg(CreatePoint_k.Args.new_point, new_point)
            .set_arg(CreatePoint_k.Args.new_point_vertex_reference, vertex_index)
            .set_arg(CreatePoint_k.Args.new_point_hull_index, hull_index)
            .set_arg(CreatePoint_k.Args.new_point_hit_count, (short) hit_count)
            .set_arg(CreatePoint_k.Args.new_point_flags, flags)
            .set_arg(CreatePoint_k.Args.new_bone_table, bone_ids)
            .call(GPGPU.global_single_size);

        return point_index++;
    }

    public int new_hull(int mesh_id,
                        float[] position,
                        float[] scale,
                        float[] rotation,
                        int[] point_table,
                        int[] edge_table,
                        int[] bone_table,
                        float friction,
                        float restitution,
                        int entity_id,
                        int uv_offset,
                        int flags)
    {
        int capacity = hull_index + 1;
        b_hull.ensure_capacity(capacity);
        b_hull_scale.ensure_capacity(capacity);
        b_hull_mesh_id.ensure_capacity(capacity);
        b_hull_uv_offset.ensure_capacity(capacity);
        b_hull_rotation.ensure_capacity(capacity);
        b_hull_integrity.ensure_capacity(capacity);
        b_hull_point_table.ensure_capacity(capacity);
        b_hull_edge_table.ensure_capacity(capacity);
        b_hull_flag.ensure_capacity(capacity);
        b_hull_bone_table.ensure_capacity(capacity);
        b_hull_entity_id.ensure_capacity(capacity);
        b_hull_friction.ensure_capacity(capacity);
        b_hull_restitution.ensure_capacity(capacity);
        b_hull_aabb.ensure_capacity(capacity);
        b_hull_aabb_index.ensure_capacity(capacity);
        b_hull_aabb_key.ensure_capacity(capacity);

        k_create_hull
            .set_arg(CreateHull_k.Args.target, hull_index)
            .set_arg(CreateHull_k.Args.new_hull, arg_float4(position[0], position[1], position[0], position[1]))
            .set_arg(CreateHull_k.Args.new_hull_scale, scale)
            .set_arg(CreateHull_k.Args.new_rotation, rotation)
            .set_arg(CreateHull_k.Args.new_friction, friction)
            .set_arg(CreateHull_k.Args.new_restitution, restitution)
            .set_arg(CreateHull_k.Args.new_point_table, point_table)
            .set_arg(CreateHull_k.Args.new_edge_table, edge_table)
            .set_arg(CreateHull_k.Args.new_bone_table, bone_table)
            .set_arg(CreateHull_k.Args.new_entity_id, entity_id)
            .set_arg(CreateHull_k.Args.new_flags, flags)
            .set_arg(CreateHull_k.Args.new_hull_mesh_id, mesh_id)
            .set_arg(CreateHull_k.Args.new_hull_uv_offset, uv_offset)
            .set_arg(CreateHull_k.Args.new_hull_integrity, 100)
            .call(GPGPU.global_single_size);

        return hull_index++;
    }

    public int new_mesh_reference(int[] vertex_table, int[] face_table)
    {
        int capacity = mesh_index + 1;

        b_mesh_vertex_table.ensure_capacity(capacity);
        b_mesh_face_table.ensure_capacity(capacity);

        k_create_mesh_reference
            .set_arg(CreateMeshReference_k.Args.target, mesh_index)
            .set_arg(CreateMeshReference_k.Args.new_mesh_vertex_table, vertex_table)
            .set_arg(CreateMeshReference_k.Args.new_mesh_face_table, face_table)
            .call(GPGPU.global_single_size);

        return mesh_index++;
    }

    public int new_mesh_face(int[] face)
    {
        int capacity = face_index + 1;
        b_mesh_face.ensure_capacity(capacity);

        k_create_mesh_face
            .set_arg(CreateMeshFace_k.Args.target, face_index)
            .set_arg(CreateMeshFace_k.Args.new_mesh_face, face)
            .call(GPGPU.global_single_size);

        return face_index++;
    }

    public int new_entity(float x, float y,
                          int[] hull_table,
                          int[] bone_table,
                          float mass,
                          int anim_index,
                          float anim_time,
                          int root_hull,
                          int model_id,
                          int model_transform_id,
                          int flags)
    {
        int capacity = entity_index + 1;
        b_entity.ensure_capacity(capacity);
        b_entity_flag.ensure_capacity(capacity);
        b_entity_root_hull.ensure_capacity(capacity);
        b_entity_model_id.ensure_capacity(capacity);
        b_entity_model_transform.ensure_capacity(capacity);
        b_entity_accel.ensure_capacity(capacity);
        b_entity_mass.ensure_capacity(capacity);
        b_entity_anim_index.ensure_capacity(capacity);
        b_entity_anim_elapsed.ensure_capacity(capacity);
        b_entity_anim_blend.ensure_capacity(capacity);
        b_entity_motion_state.ensure_capacity(capacity);
        b_entity_hull_table.ensure_capacity(capacity);
        b_entity_bone_table.ensure_capacity(capacity);

        k_create_entity
            .set_arg(CreateEntity_k.Args.target, entity_index)
            .set_arg(CreateEntity_k.Args.new_entity, arg_float4(x, y, x, y))
            .set_arg(CreateEntity_k.Args.new_entity_root_hull, root_hull)
            .set_arg(CreateEntity_k.Args.new_entity_model_id, model_id)
            .set_arg(CreateEntity_k.Args.new_entity_model_transform, model_transform_id)
            .set_arg(CreateEntity_k.Args.new_entity_flags, flags)
            .set_arg(CreateEntity_k.Args.new_entity_hull_table, hull_table)
            .set_arg(CreateEntity_k.Args.new_entity_bone_table, bone_table)
            .set_arg(CreateEntity_k.Args.new_entity_mass, mass)
            .set_arg(CreateEntity_k.Args.new_entity_animation_index, arg_int2(anim_index, -1))
            .set_arg(CreateEntity_k.Args.new_entity_animation_time, arg_float2(0.0f, 0.0f)) // todo: maybe remove these zero init ones
            .set_arg(CreateEntity_k.Args.new_entity_animation_state, arg_short2((short) 0, (short) 0))
            .call(GPGPU.global_single_size);

        return entity_index++;
    }

    public int new_vertex_reference(float x, float y, float[] weights, int[] uv_table)
    {
        int capacity = vertex_ref_index + 1;
        b_vertex_reference.ensure_capacity(capacity);
        b_vertex_weight.ensure_capacity(capacity);
        b_vertex_uv_table.ensure_capacity(capacity);

        k_create_vertex_reference
            .set_arg(CreateVertexRef_k.Args.target, vertex_ref_index)
            .set_arg(CreateVertexRef_k.Args.new_vertex_reference, arg_float2(x, y))
            .set_arg(CreateVertexRef_k.Args.new_vertex_weights, weights)
            .set_arg(CreateVertexRef_k.Args.new_uv_table, uv_table)
            .call(GPGPU.global_single_size);

        return vertex_ref_index++;
    }

    public int new_bone_bind_pose(float[] bone_data)
    {
        int capacity = bone_bind_index + 1;
        b_bone_bind_pose.ensure_capacity(capacity);
        b_bone_anim_channel_table.ensure_capacity(capacity); // note: filled in later

        k_create_bone_bind_pose
            .set_arg(CreateBoneBindPose_k.Args.target,bone_bind_index)
            .set_arg(CreateBoneBindPose_k.Args.new_bone_bind_pose, bone_data)
            .call(GPGPU.global_single_size);

        return bone_bind_index++;
    }

    public int new_bone_reference(float[] bone_data)
    {
        int capacity = bone_ref_index + 1;
        b_bone_reference.ensure_capacity(capacity);

        k_create_bone_reference
            .set_arg(CreateBoneRef_k.Args.target, bone_ref_index)
            .set_arg(CreateBoneRef_k.Args.new_bone_reference, bone_data)
            .call(GPGPU.global_single_size);

        return bone_ref_index++;
    }

    public int new_hull_bone(float[] bone_data, int bind_pose_id, int inv_bind_pose_id)
    {
        int capacity = hull_bone_index + 1;
        b_hull_bone.ensure_capacity(capacity);
        b_hull_bone_bind_pose_id.ensure_capacity(capacity);
        b_hull_bone_inv_bind_pose_id.ensure_capacity(capacity);

        k_create_bone
            .set_arg(CreateHullBone_k.Args.target, hull_bone_index)
            .set_arg(CreateHullBone_k.Args.new_bone, bone_data)
            .set_arg(CreateHullBone_k.Args.new_hull_bind_pose_id, bind_pose_id)
            .set_arg(CreateHullBone_k.Args.new_hull_inv_bind_pose_id, inv_bind_pose_id)
            .call(GPGPU.global_single_size);

        return hull_bone_index++;
    }

    public int new_armature_bone(int bone_reference, int bone_parent_id, float[] bone_data)
    {
        int capacity = armature_bone_index + 1;
        b_armature_bone.ensure_capacity(capacity);
        b_armature_bone_reference_id.ensure_capacity(capacity);
        b_armature_bone_parent_id.ensure_capacity(capacity);

        k_create_armature_bone
            .set_arg(CreateArmatureBone_k.Args.target, armature_bone_index)
            .set_arg(CreateArmatureBone_k.Args.new_armature_bone, bone_data)
            .set_arg(CreateArmatureBone_k.Args.new_armature_bone_reference, bone_reference)
            .set_arg(CreateArmatureBone_k.Args.new_armature_bone_parent_id, bone_parent_id)
            .call(GPGPU.global_single_size);

        return armature_bone_index++;
    }

    @Override
    public void merge_into_parent(WorldContainer parent)
    {
        throw new UnsupportedOperationException("Cannot merge core memory");
    }


    public int new_model_transform(float[] transform_data)
    {
        int capacity = model_transform_index + 1;
        b_model_transform.ensure_capacity(capacity);

        k_create_model_transform
            .set_arg(CreateModelTransform_k.Args.target, model_transform_index)
            .set_arg(CreateModelTransform_k.Args.new_model_transform, transform_data)
            .call(GPGPU.global_single_size);

        return model_transform_index++;
    }

    public void set_bone_channel_table(int bind_pose_target, int[] channel_table)
    {
        k_set_bone_channel_table
            .set_arg(SetBoneChannelTable_k.Args.target, bind_pose_target)
            .set_arg(SetBoneChannelTable_k.Args.new_bone_channel_table, channel_table)
            .call(GPGPU.global_single_size);
    }

    public void update_accel(int entity_index, float acc_x, float acc_y)
    {
        k_update_accel
            .set_arg(UpdateAccel_k.Args.target, entity_index)
            .set_arg(UpdateAccel_k.Args.new_value, arg_float2(acc_x, acc_y))
            .call(GPGPU.global_single_size);
    }

    public void update_position(int entity_index, float x, float y)
    {
        k_update_mouse_position
            .set_arg(UpdateMousePosition_k.Args.target, entity_index)
            .set_arg(UpdateMousePosition_k.Args.new_value, arg_float2(x, y))
            .call(GPGPU.global_single_size);
    }

    public float[] read_position(int entity_index)
    {
        GPGPU.cl_zero_buffer(GPGPU.ptr_compute_queue, ptr_position_buffer, CLSize.cl_float2);

        k_read_position
            .ptr_arg(ReadPosition_k.Args.output, ptr_position_buffer)
            .set_arg(ReadPosition_k.Args.target, entity_index)
            .call(GPGPU.global_single_size);

        return GPGPU.cl_read_pinned_float_buffer(GPGPU.ptr_compute_queue, ptr_position_buffer, CLSize.cl_float, 2);
    }

    public int[] count_egress_entities()
    {
        GPGPU.cl_zero_buffer(GPGPU.ptr_compute_queue, ptr_egress_sizes, CLSize.cl_int * 6);
        k_count_egress_entities.call(arg_long(entity_index));
        return GPGPU.cl_read_pinned_int_buffer(GPGPU.ptr_compute_queue, ptr_egress_sizes, CLSize.cl_int, 6);
    }

    public void delete_and_compact()
    {
        //k_locate_out_of_bounds.call(arg_long(entity_index));

        b_delete_1.ensure_capacity(entity_index);
        b_delete_2.ensure_capacity(entity_index);

        int[] shift_counts = scan_deletes(b_delete_1.pointer(), b_delete_2.pointer(), entity_index);

        if (shift_counts[4] == 0)
        {
            return;
        }

        b_hull_shift.ensure_capacity(hull_index);
        b_edge_shift.ensure_capacity(edge_index);
        b_point_shift.ensure_capacity(point_index);
        b_hull_bone_shift.ensure_capacity(hull_bone_index);
        b_armature_bone_shift.ensure_capacity(armature_bone_index);

        b_hull_shift.clear();
        b_edge_shift.clear();
        b_point_shift.clear();
        b_hull_bone_shift.clear();
        b_armature_bone_shift.clear();

        k_compact_entities
            .ptr_arg(CompactEntities_k.Args.buffer_in_1, b_delete_1.pointer())
            .ptr_arg(CompactEntities_k.Args.buffer_in_2, b_delete_2.pointer());

        linearize_kernel(k_compact_entities, entity_index);
        linearize_kernel(k_compact_hull_bones, hull_bone_index);
        linearize_kernel(k_compact_points, point_index);
        linearize_kernel(k_compact_edges, edge_index);
        linearize_kernel(k_compact_hulls, hull_index);
        linearize_kernel(k_compact_armature_bones, armature_bone_index);

        compact_buffers(shift_counts);
    }

    private void linearize_kernel(GPUKernel kernel, int object_count)
    {
        int offset = 0;
        for (long remaining = object_count; remaining > 0; remaining -= GPGPU.max_work_group_size)
        {
            int count = (int) Math.min(GPGPU.max_work_group_size, remaining);
            var sz = count == GPGPU.max_work_group_size
                ? GPGPU.local_work_default
                : arg_long(count);
            kernel.call(sz, sz, arg_long(offset));
            offset += count;
        }
    }

    public int[] scan_deletes(long o1_data_ptr, long o2_data_ptr, int n)
    {
        int k = GPGPU.work_group_count(n);
        if (k == 1)
        {
            return scan_single_block_deletes_out(o1_data_ptr, o2_data_ptr, n);
        }
        else
        {
            return scan_multi_block_deletes_out(o1_data_ptr, o2_data_ptr, n, k);
        }
    }

    private int[] scan_single_block_deletes_out(long o1_data_ptr, long o2_data_ptr, int n)
    {
        long local_buffer_size = CLSize.cl_int2 * GPGPU.max_scan_block_size;
        long local_buffer_size2 = CLSize.cl_int4 * GPGPU.max_scan_block_size;

        GPGPU.cl_zero_buffer(GPGPU.ptr_compute_queue, ptr_delete_sizes, CLSize.cl_int * 6);

        k_scan_deletes_single_block_out
            .ptr_arg(ScanDeletesSingleBlockOut_k.Args.output, o1_data_ptr)
            .ptr_arg(ScanDeletesSingleBlockOut_k.Args.output2, o2_data_ptr)
            .loc_arg(ScanDeletesSingleBlockOut_k.Args.buffer, local_buffer_size)
            .loc_arg(ScanDeletesSingleBlockOut_k.Args.buffer2, local_buffer_size2)
            .set_arg(ScanDeletesSingleBlockOut_k.Args.n, n)
            .call(GPGPU.local_work_default, GPGPU.local_work_default);

        return GPGPU.cl_read_pinned_int_buffer(GPGPU.ptr_compute_queue, ptr_delete_sizes, CLSize.cl_int, 6);
    }

    private int[] scan_multi_block_deletes_out(long o1_data_ptr, long o2_data_ptr, int n, int k)
    {
        long local_buffer_size = CLSize.cl_int2 * GPGPU.max_scan_block_size;
        long local_buffer_size2 = CLSize.cl_int4 * GPGPU.max_scan_block_size;

        long gx = k * GPGPU.max_scan_block_size;
        long[] global_work_size = arg_long(gx);
        int part_size = k * 2;

        b_delete_partial_1.ensure_capacity(part_size);
        b_delete_partial_2.ensure_capacity(part_size);

        k_scan_deletes_multi_block_out
            .ptr_arg(ScanDeletesMultiBlockOut_k.Args.output1, o1_data_ptr)
            .ptr_arg(ScanDeletesMultiBlockOut_k.Args.output2, o2_data_ptr)
            .loc_arg(ScanDeletesMultiBlockOut_k.Args.buffer1, local_buffer_size)
            .loc_arg(ScanDeletesMultiBlockOut_k.Args.buffer2, local_buffer_size2)
            .set_arg(ScanDeletesMultiBlockOut_k.Args.n, n)
            .call(global_work_size, GPGPU.local_work_default);

        // note the partial buffers are scanned and updated in-place
        GPGPU.scan_int2(b_delete_partial_1.pointer(), part_size);
        GPGPU.scan_int4(b_delete_partial_2.pointer(), part_size);

        GPGPU.cl_zero_buffer(GPGPU.ptr_compute_queue, ptr_delete_sizes, CLSize.cl_int * 6);

        k_complete_deletes_multi_block_out
            .ptr_arg(CompleteDeletesMultiBlockOut_k.Args.output1, o1_data_ptr)
            .ptr_arg(CompleteDeletesMultiBlockOut_k.Args.output2, o2_data_ptr)
            .loc_arg(CompleteDeletesMultiBlockOut_k.Args.buffer1, local_buffer_size)
            .loc_arg(CompleteDeletesMultiBlockOut_k.Args.buffer2, local_buffer_size2)
            .set_arg(CompleteDeletesMultiBlockOut_k.Args.n, n)
            .call(global_work_size, GPGPU.local_work_default);

        return GPGPU.cl_read_pinned_int_buffer(GPGPU.ptr_compute_queue, ptr_delete_sizes, CLSize.cl_int, 6);
    }

    private void compact_buffers(int[] shift_counts)
    {
        edge_index          -= (shift_counts[0]);
        hull_bone_index     -= (shift_counts[1]);
        point_index         -= (shift_counts[2]);
        hull_index          -= (shift_counts[3]);
        entity_index        -= (shift_counts[4]);
        armature_bone_index -= (shift_counts[5]);
    }

    // todo: implement entity rotations and update this
    public static void rotate_hull(int hull_index, float angle)
    {
//        var pnt_index = Pointer.to(arg_int(hull_index));
//        var pnt_angle = Pointer.to(arg_float(angle));
//
//        clSetKernelArg(_k.get(Kernel.rotate_hull), 0, CLSize.cl_mem, Memory.hulls.gpu.pointer());
//        clSetKernelArg(_k.get(Kernel.rotate_hull), 1, CLSize.cl_mem, Memory.hull_element_table.gpu.pointer());
//        clSetKernelArg(_k.get(Kernel.rotate_hull), 2, CLSize.cl_mem, Memory.points.gpu.pointer());
//        clSetKernelArg(_k.get(Kernel.rotate_hull), 3, CLSize.cl_int, pnt_index);
//        clSetKernelArg(_k.get(Kernel.rotate_hull), 4, CLSize.cl_float, pnt_angle);
//
//        k_call(command_queue, _k.get(Kernel.rotate_hull), global_single_size);
    }

    @Override
    public void destroy()
    {
        incoming_world_buffer.destroy();
        outgoing_world_buffer_a.destroy();
        outgoing_world_buffer_b.destroy();

        p_gpu_crud.destroy();
        p_scan_deletes.destroy();
        b_hull_shift.release();
        b_edge_shift.release();
        b_point_shift.release();
        b_hull_bone_shift.release();
        b_armature_bone_shift.release();
        b_delete_1.release();
        b_delete_2.release();
        b_delete_partial_1.release();
        b_delete_partial_2.release();

        b_edge.release();
        b_edge_length.release();
        b_edge_flag.release();
        b_hull.release();
        b_hull_scale.release();
        b_hull_mesh_id.release();
        b_hull_uv_offset.release();
        b_hull_rotation.release();
        b_hull_integrity.release();
        b_hull_point_table.release();
        b_hull_edge_table.release();
        b_hull_flag.release();
        b_hull_bone_table.release();
        b_hull_entity_id.release();
        b_hull_aabb.release();
        b_hull_aabb_index.release();
        b_hull_aabb_key.release();
        b_hull_bone.release();
        b_hull_bone_bind_pose_id.release();
        b_hull_bone_inv_bind_pose_id.release();
        b_hull_friction.release();
        b_hull_restitution.release();
        b_point.release();
        b_point_anti_gravity.release();
        b_point_vertex_reference.release();
        b_point_hull_index.release();
        b_point_flag.release();
        b_point_hit_count.release();
        b_point_bone_table.release();
        b_vertex_reference.release();
        b_vertex_weight.release();
        b_vertex_texture_uv.release();
        b_vertex_uv_table.release();
        b_model_transform.release();
        b_bone_reference.release();
        b_bone_bind_pose.release();
        b_bone_anim_channel_table.release();
        b_mesh_vertex_table.release();
        b_mesh_face_table.release();
        b_mesh_face.release();
        b_anim_key_frame.release();
        b_anim_frame_time.release();
        b_anim_bone_pos_channel.release();
        b_anim_bone_rot_channel.release();
        b_anim_bone_scl_channel.release();
        b_anim_duration.release();
        b_anim_tick_rate.release();
        b_anim_timing_index.release();
        b_armature_bone.release();
        b_armature_bone_reference_id.release();
        b_armature_bone_parent_id.release();
        b_entity.release();
        b_entity_flag.release();
        b_entity_root_hull.release();
        b_entity_model_id.release();
        b_entity_model_transform.release();
        b_entity_accel.release();
        b_entity_mass.release();
        b_entity_anim_index.release();
        b_entity_anim_elapsed.release();
        b_entity_anim_blend.release();
        b_entity_motion_state.release();
        b_entity_hull_table.release();
        b_entity_bone_table.release();

        debug();

        GPGPU.cl_release_buffer(ptr_delete_counter);
        GPGPU.cl_release_buffer(ptr_position_buffer);
        GPGPU.cl_release_buffer(ptr_delete_sizes);
    }

    private void debug()
    {
        long total = 0;
        total += b_hull_shift.debug_data();
        total += b_edge_shift.debug_data();
        total += b_point_shift.debug_data();
        total += b_hull_bone_shift.debug_data();
        total += b_armature_bone_shift.debug_data();
        total += b_delete_1.debug_data();
        total += b_delete_2.debug_data();
        total += b_delete_partial_1.debug_data();
        total += b_delete_partial_2.debug_data();
        total += b_edge.debug_data();
        total += b_edge_length.debug_data();
        total += b_edge_flag.debug_data();
        total += b_hull.debug_data();
        total += b_hull_scale.debug_data();
        total += b_hull_mesh_id.debug_data();
        total += b_hull_uv_offset.debug_data();
        total += b_hull_rotation.debug_data();
        total += b_hull_integrity.debug_data();
        total += b_hull_point_table.debug_data();
        total += b_hull_edge_table.debug_data();
        total += b_hull_flag.debug_data();
        total += b_hull_bone_table.debug_data();
        total += b_hull_entity_id.debug_data();
        total += b_hull_aabb.debug_data();
        total += b_hull_aabb_index.debug_data();
        total += b_hull_aabb_key.debug_data();
        total += b_hull_bone.debug_data();
        total += b_hull_bone_bind_pose_id.debug_data();
        total += b_hull_bone_inv_bind_pose_id.debug_data();
        total += b_hull_friction.debug_data();
        total += b_hull_restitution.debug_data();
        total += b_point.debug_data();
        total += b_point_anti_gravity.debug_data();
        total += b_point_vertex_reference.debug_data();
        total += b_point_hull_index.debug_data();
        total += b_point_flag.debug_data();
        total += b_point_hit_count.debug_data();
        total += b_point_bone_table.debug_data();
        total += b_vertex_reference.debug_data();
        total += b_vertex_weight.debug_data();
        total += b_vertex_texture_uv.debug_data();
        total += b_vertex_uv_table.debug_data();
        total += b_model_transform.debug_data();
        total += b_bone_reference.debug_data();
        total += b_bone_bind_pose.debug_data();
        total += b_bone_anim_channel_table.debug_data();
        total += b_mesh_vertex_table.debug_data();
        total += b_mesh_face_table.debug_data();
        total += b_mesh_face.debug_data();
        total += b_anim_key_frame.debug_data();
        total += b_anim_frame_time.debug_data();
        total += b_anim_bone_pos_channel.debug_data();
        total += b_anim_bone_rot_channel.debug_data();
        total += b_anim_bone_scl_channel.debug_data();
        total += b_anim_duration.debug_data();
        total += b_anim_tick_rate.debug_data();
        total += b_anim_timing_index.debug_data();
        total += b_armature_bone.debug_data();
        total += b_armature_bone_reference_id.debug_data();
        total += b_armature_bone_parent_id.debug_data();
        total += b_entity.debug_data();
        total += b_entity_flag.debug_data();
        total += b_entity_root_hull.debug_data();
        total += b_entity_model_id.debug_data();
        total += b_entity_model_transform.debug_data();
        total += b_entity_accel.debug_data();
        total += b_entity_mass.debug_data();
        total += b_entity_anim_index.debug_data();
        total += b_entity_anim_elapsed.debug_data();
        total += b_entity_anim_blend.debug_data();
        total += b_entity_motion_state.debug_data();
        total += b_entity_hull_table.debug_data();
        total += b_entity_bone_table.debug_data();

        //System.out.println("---------------------------");
        System.out.println("Core Memory Usage: MB " + ((float) total / 1024f / 1024f));
    }
}
