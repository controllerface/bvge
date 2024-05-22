package com.controllerface.bvge.cl;

import com.controllerface.bvge.cl.kernels.*;
import com.controllerface.bvge.cl.programs.GPUCrud;
import com.controllerface.bvge.cl.programs.ScanDeletes;
import com.controllerface.bvge.physics.PhysicsEntityBatch;

import static com.controllerface.bvge.cl.CLUtils.*;

public class GPUCoreMemory
{
    private final GPUProgram gpu_crud = new GPUCrud();
    private final GPUProgram scan_deletes = new ScanDeletes();

    private final GPUKernel compact_armature_bones_k;
    private final GPUKernel compact_entities_k;
    private final GPUKernel compact_hull_bones_k;
    private final GPUKernel compact_edges_k;
    private final GPUKernel compact_hulls_k;
    private final GPUKernel compact_points_k;
    private final GPUKernel complete_deletes_multi_block_out_k;
    private final GPUKernel create_animation_timings_k;
    private final GPUKernel create_armature_bone_k;
    private final GPUKernel create_entity_k;
    private final GPUKernel create_bone_bind_pose_k;
    private final GPUKernel create_bone_channel_k;
    private final GPUKernel create_bone_k;
    private final GPUKernel create_bone_reference_k;
    private final GPUKernel create_edge_k;
    private final GPUKernel create_hull_k;
    private final GPUKernel create_keyframe_k;
    private final GPUKernel create_mesh_face_k;
    private final GPUKernel create_mesh_reference_k;
    private final GPUKernel create_model_transform_k;
    private final GPUKernel create_point_k;
    private final GPUKernel create_texture_uv_k;
    private final GPUKernel create_vertex_reference_k;
    private final GPUKernel locate_out_of_bounds_k;
    private final GPUKernel read_position_k;
    private final GPUKernel scan_deletes_multi_block_out_k;
    private final GPUKernel scan_deletes_single_block_out_k;
    private final GPUKernel set_bone_channel_table_k;
    private final GPUKernel update_accel_k;
    private final GPUKernel update_mouse_position_k;

    // internal buffers

    //#region Compaction/Shift Buffers

    /**
     * During the entity compaction process, these buffers are written to, and store the number of
     * positions that the corresponding values must shift left within their own buffers when the
     * buffer compaction occurs. Each index is aligned with the corresponding data type
     * that will be shifted. I.e. every bone in the bone buffer has a corresponding entry in the
     * bone shift buffer. Points, edges, and hulls work the same way.
     */

    private final ResizableBuffer armature_bone_shift;
    private final ResizableBuffer hull_bone_shift;
    private final ResizableBuffer edge_shift;
    private final ResizableBuffer hull_shift;
    private final ResizableBuffer point_shift;

    /**
     * During the deletion process, these buffers are used during the parallel scan of the relevant data
     * buffers. The partial buffers are utilized when the parallel scan occurs over multiple scan blocks,
     * and allows the output of each block to then itself be scanned, until all values have been summed.
     */

    private final ResizableBuffer delete_buffer_1;
    private final ResizableBuffer delete_buffer_2;
    private final ResizableBuffer delete_partial_buffer_1;
    private final ResizableBuffer delete_partial_buffer_2;

    //#endregion

    // external buffers

    //#region Animation Data Buffers

    /** int2
     * x: position channel start index
     * y: position channel end index
     */
    private final ResizableBuffer anim_bone_pos_channel_buffer;

    /** int2
     * x: rotation channel start index
     * y: rotation channel end index
     */
    private final ResizableBuffer anim_bone_rot_channel_buffer;

    /** int2
     * x: scaling channel start index
     * y: scaling channel end index
     */
    private final ResizableBuffer anim_bone_scl_channel_buffer;

    /** float
     * x: key frame timestamp
     */
    private final ResizableBuffer anim_frame_time_buffer;

    /** float4
     * x: vector/quaternion x
     * y: vector/quaternion y
     * z: vector/quaternion z
     * w: vector/quaternion w
     */
    private final ResizableBuffer anim_key_frame_buffer;

    /** float
     * x: animation duration
     */
    private final ResizableBuffer anim_duration_buffer;

    /** float
     * x: animation tick rate (FPS)
     */
    private final ResizableBuffer anim_tick_rate_buffer;

    /** int
     * x: animation timing index
     */
    private final ResizableBuffer anim_timing_index_buffer;

    //#endregion

    //#region Armature Buffers

    /** float16
     * s0-sF: Column-major, 4x4 transformation matrix, armature bone instance
     */
    private final ResizableBuffer armature_bone_buffer;

    /** int
     * x: bind pose reference id
     */
    private final ResizableBuffer armature_bone_reference_id_buffer;

    /** int
     * x: armature bone parent id
     */
    private final ResizableBuffer armature_bone_parent_id_buffer;

    //#endregion

    //#region Entity Buffers

    /** float2
     * x: current x acceleration
     * y: current y acceleration
     */
    private final ResizableBuffer entity_accel_buffer;

    /** float2
     * x: the last rendered timestamp of the current animation
     * y: the last rendered timestamp of the previous animation
     */
    private final ResizableBuffer entity_anim_elapsed_buffer;

    /** float2
     * x: the initial time of the current blend operation
     * y: the remaining time of the current blend operation
     */
    private final ResizableBuffer entity_anim_blend_buffer;

    /** short2
     * x: number of ticks moving downward
     * y: number of ticks moving upward
     */
    private final ResizableBuffer entity_motion_state_buffer;

    /** int2
     * x: the currently running animation index
     * y: the previously running animation index
     */
    private final ResizableBuffer entity_anim_index_buffer;

    /** float4
     * x: current x position
     * y: current y position
     * z: previous x position
     * w: previous y position
     */
    private final ResizableBuffer entity_buffer;

    /** int
     * x: entity flags (bit-field)
     */
    private final ResizableBuffer entity_flag_buffer;

    /** int
     * x: root hull index of the aligned entity
     */
    private final ResizableBuffer entity_root_hull_buffer;

    /** int
     * x: model id of the aligned entity
     */
    private final ResizableBuffer entity_model_id_buffer;

    /** int
     * x: model transform index of the aligned entity
     */
    private final ResizableBuffer entity_model_transform_buffer;

    /** int2
     * x: start hull index
     * y: end hull index
     */
    private final ResizableBuffer entity_hull_table_buffer;

    /** int2
     * x: start bone anim index
     * y: end bone anim index
     */
    private final ResizableBuffer entity_bone_table_buffer;

    /** float
     * x: mass of the entity
     */
    private final ResizableBuffer entity_mass_buffer;

    //#endregion

    //#region Bone Buffers

    /** int2
     * x: bone channel start index
     * y: bone channel end index
     */
    private final ResizableBuffer bone_anim_channel_table_buffer;

    /** float16
     * s0-sF: Column-major, 4x4 transformation matrix, mesh-space bone reference (inverse bind pose)
     */
    private final ResizableBuffer bone_bind_pose_buffer;

    /** float16
     * s0-sF: Column-major, 4x4 transformation matrix, model-space bone reference (bind pose)
     */
    private final ResizableBuffer bone_reference_buffer;

    //#endregion

    //#region Edge Buffers

    /** int2
     * x: point 1 index
     * y: point 2 index
     */
    private final ResizableBuffer edge_buffer;

    /** int
     * x: edge flags (bit-field)
     */
    private final ResizableBuffer edge_flag_buffer;

    /** float
     * x: edge constraint length
     */
    private final ResizableBuffer edge_length_buffer;

    //#endregion

    //#region Hull Buffers

    /** float4
     * x: corner x position
     * y: corner y position
     * z: width
     * w: height
     */
    private final ResizableBuffer hull_aabb_b;

    /** int4
     * x: minimum x key index
     * y: maximum x key index
     * z: minimum y key index
     * w: maximum y key index
     */
    private final ResizableBuffer hull_aabb_index_b;

    /** int2
     * x: key bank offset
     * y: key bank size
     */
    private final ResizableBuffer hull_aabb_key_b;

    /** float16
     * s0-sF: Column-major, 4x4 transformation matrix, hull bone instance
     */
    private final ResizableBuffer hull_bone_b;

    /** int
     * x: bone bind pose index (model space)
     */
    private final ResizableBuffer hull_bone_bind_pose_id_b;

    /** int
     * x: bone inverse bind pose index (mesh-space)
     */
    private final ResizableBuffer hull_bone_inv_bind_pose_id_b;

    /** float4
     * x: current x position
     * y: current y position
     * z: previous x position
     * w: previous y position
     */
    private final ResizableBuffer hull_b;

    /** float2
     * x: scale x
     * y: scale y
     */
    private final ResizableBuffer hull_scale_b;

    /** int2
     * x: start point index
     * y: end point index
     */
    private final ResizableBuffer hull_point_table_b;

    /** int2
     * x: start edge index
     * y: end edge index
     */
    private final ResizableBuffer hull_edge_table_b;

    /** int
     * x: hull flags (bit-field)
     */
    private final ResizableBuffer hull_flag_b;

    /** int
     * x: entity id for aligned hull
     */
    private final ResizableBuffer hull_entity_id_b;

    /** int2
     * x: start bone
     * y: end bone
     */
    private final ResizableBuffer hull_bone_table_b;

    /** float
     * x: friction coefficient
     */
    private final ResizableBuffer hull_friction_b;

    /** float
     * x: restitution coefficient
     */
    private final ResizableBuffer hull_restitution_b;

    /** int
     * x: reference mesh id
     */
    private final ResizableBuffer hull_mesh_id_b;

    /** int
     * x: offset index of the UV to use for this hull
     */
    private final ResizableBuffer hull_uv_offset_b;

    /** float2
     * x: initial reference angle
     * y: current rotation
     */
    private final ResizableBuffer hull_rotation_b;

    /** int
     * x: the integrity (i.e. health) of the hull
     */
    private final ResizableBuffer hull_integrity_b;

    //#endregion

    //#region Model/Mesh Buffers

    /** int4
     * x: vertex 1 index
     * y: vertex 2 index
     * z: vertex 3 index
     * w: parent reference mesh ID
     */
    private final ResizableBuffer mesh_face_buffer;

    /** int2
     * x: start vertex index
     * y: end vertex index
     */
    private final ResizableBuffer mesh_vertex_table_buffer;

    /** int2
     * z: start face index
     * w: end face index
     */
    private final ResizableBuffer mesh_face_table_buffer;

    /** float16
     * s0-sF: Column-major, 4x4 transformation matrix
     */
    private final ResizableBuffer model_transform_buffer;

    //#endregion

    //#region Point Buffers

    /** float
     * x: anti-gravity magnitude for each point
     */
    private final ResizableBuffer point_anti_gravity_buffer;

    /** int4
     * x: bone 1 instance id
     * y: bone 2 instance id
     * z: bone 3 instance id
     * w: bone 4 instance id
     */
    private final ResizableBuffer point_bone_table_buffer;

    /** float4
     * x: current x position
     * y: current y position
     * z: previous x position
     * w: previous y position
     */
    private final ResizableBuffer point_buffer;

    /** int
     * x: reference vertex index
     */
    private final ResizableBuffer point_vertex_reference_buffer;

    /** int
     * x: hull index
     */
    private final ResizableBuffer point_hull_index_buffer;

    /** int
     * x: vertex flags (bit field)
     */
    private final ResizableBuffer point_flag_buffer;

    /** ushort
     * x: recent collision hit counter
     */
    private final ResizableBuffer point_hit_count_buffer;

    //#endregion

    //#region Vertex Buffers

    /** float2
     * x: x position
     * y: y position
     */
    private final ResizableBuffer vertex_reference_buffer;

    /** float2
     * x: u coordinate
     * y: v coordinate
     */
    private final ResizableBuffer vertex_texture_uv_buffer;

    /** int2
     * x: start UV index
     * y: end UV index
     */
    private final ResizableBuffer vertex_uv_table_buffer;

    /** float4
     * x: bone 1 weight
     * y: bone 2 weight
     * z: bone 3 weight
     * w: bone 4 weight
     */
    private final ResizableBuffer vertex_weight_buffer;

    //#endregion

    //#region Mirror Buffers

    /**
     * Mirror buffers are configured only for certain core buffers, and are used solely for rendering purposes.
     * Between physics simulation ticks, rendering threads use the mirror buffers to render the state of the objects
     * while the physics thread is busy calculating the data for the next frame.
     */
    private final ResizableBuffer mirror_entity_buffer;
    private final ResizableBuffer mirror_entity_flag_buffer;
    private final ResizableBuffer mirror_entity_model_id_buffer;
    private final ResizableBuffer mirror_entity_root_hull_buffer;
    private final ResizableBuffer mirror_edge_buffer;
    private final ResizableBuffer mirror_edge_flag_buffer;
    private final ResizableBuffer mirror_hull_buffer;
    private final ResizableBuffer mirror_hull_aabb_buffer;
    private final ResizableBuffer mirror_hull_flag_buffer;
    private final ResizableBuffer mirror_hull_entity_id_buffer;
    private final ResizableBuffer mirror_hull_mesh_id_buffer;
    private final ResizableBuffer mirror_hull_uv_offset_buffer;
    private final ResizableBuffer mirror_hull_integrity_buffer;
    private final ResizableBuffer mirror_hull_point_table_buffer;
    private final ResizableBuffer mirror_hull_rotation_buffer;
    private final ResizableBuffer mirror_hull_scale_buffer;
    private final ResizableBuffer mirror_point_buffer;
    private final ResizableBuffer mirror_point_anti_gravity_buffer;
    private final ResizableBuffer mirror_point_hit_count_buffer;
    private final ResizableBuffer mirror_point_vertex_reference_buffer;

    //#endregion

    private final long delete_counter_ptr;
    private final long position_buffer_ptr;
    private final long delete_sizes_ptr;

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
    private int last_entity_index = 0;

    public GPUCoreMemory()
    {
        delete_counter_ptr      = GPGPU.cl_new_int_arg_buffer(new int[]{ 0 });
        position_buffer_ptr     = GPGPU.cl_new_pinned_buffer(CLSize.cl_float2);
        delete_sizes_ptr        = GPGPU.cl_new_pinned_buffer(CLSize.cl_int * 6);

        // transients
        hull_shift              = new TransientBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_int, 10_000L);
        edge_shift              = new TransientBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_int, 24_000L);
        point_shift             = new TransientBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_int, 50_000L);
        hull_bone_shift         = new TransientBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_int, 10_000L);
        armature_bone_shift     = new TransientBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_int, 24_000L);
        delete_buffer_1         = new TransientBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_int2, 10_000L);
        delete_buffer_2         = new TransientBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_int4, 20_000L);
        delete_partial_buffer_1 = new TransientBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_int2, 10_000L);
        delete_partial_buffer_2 = new TransientBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_int4, 20_000L);

        // persistent buffers
        anim_bone_pos_channel_buffer      = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_int2);
        anim_bone_rot_channel_buffer      = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_int2);
        anim_bone_scl_channel_buffer      = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_int2);
        anim_frame_time_buffer            = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_float);
        anim_key_frame_buffer             = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_float4);
        anim_duration_buffer              = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_float);
        anim_tick_rate_buffer             = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_float);
        anim_timing_index_buffer          = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_int);
        entity_accel_buffer               = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_float2, 10_000L);
        entity_anim_elapsed_buffer        = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_float2, 10_000L);
        entity_anim_blend_buffer          = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_float2, 10_000L);
        entity_motion_state_buffer        = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_short2, 10_000L);
        entity_anim_index_buffer          = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_int2, 10_000L);
        armature_bone_buffer              = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_float16);
        armature_bone_reference_id_buffer = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_int);
        armature_bone_parent_id_buffer    = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_int);
        entity_buffer                     = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_float4, 10_000L);
        entity_flag_buffer                = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_int, 10_000L);
        entity_root_hull_buffer           = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_int, 10_000L);
        entity_model_id_buffer            = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_int, 10_000L);
        entity_model_transform_buffer     = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_int, 10_000L);
        entity_hull_table_buffer          = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_int2, 10_000L);
        entity_bone_table_buffer          = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_int2, 10_000L);
        entity_mass_buffer                = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_float, 10_000L);
        bone_anim_channel_table_buffer    = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_int2);
        bone_bind_pose_buffer             = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_float16);
        bone_reference_buffer             = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_float16);
        edge_buffer                       = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_int2, 24_000L);
        edge_flag_buffer                  = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_int, 24_000L);
        edge_length_buffer                = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_float, 24_000L);
        hull_aabb_b                       = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_float4, 10_000L);
        hull_aabb_index_b                 = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_int4, 10_000L);
        hull_aabb_key_b                   = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_int2, 10_000L);
        hull_bone_b                       = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_float16, 10_000L);
        hull_bone_bind_pose_id_b          = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_int, 10_000L);
        hull_bone_inv_bind_pose_id_b      = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_int, 10_000L);
        hull_b                            = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_float4, 10_000L);
        hull_scale_b                      = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_float2, 10_000L);
        hull_point_table_b                = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_int2, 10_000L);
        hull_edge_table_b                 = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_int2, 10_000L);
        hull_flag_b                       = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_int, 10_000L);
        hull_bone_table_b                 = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_int2, 10_000L);
        hull_entity_id_b                  = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_int, 10_000L);
        hull_friction_b                   = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_float, 10_000L);
        hull_restitution_b                = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_float, 10_000L);
        hull_mesh_id_b                    = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_int, 10_000L);
        hull_uv_offset_b                  = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_int, 10_000L);
        hull_rotation_b                   = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_float2, 10_000L);
        hull_integrity_b                  = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_int, 10_000L);
        mesh_face_buffer                  = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_int4);
        mesh_vertex_table_buffer          = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_int2);
        mesh_face_table_buffer            = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_int2);
        model_transform_buffer            = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_float16);
        point_anti_gravity_buffer         = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_float, 50_000L);
        point_bone_table_buffer           = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_int4, 50_000L);
        point_buffer                      = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_float4, 50_000L);
        point_vertex_reference_buffer     = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_int, 50_000L);
        point_hull_index_buffer           = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_int, 50_000L);
        point_flag_buffer                 = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_int, 50_000L);
        point_hit_count_buffer            = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_ushort, 50_000L);
        vertex_reference_buffer           = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_float2);
        vertex_texture_uv_buffer          = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_float2);
        vertex_uv_table_buffer            = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_int2);
        vertex_weight_buffer              = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_float4);

        // mirrors
        mirror_entity_buffer                 = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_float4, 10_000L);
        mirror_entity_flag_buffer            = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_int, 10_000L);
        mirror_entity_model_id_buffer        = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_int, 10_000L);
        mirror_entity_root_hull_buffer       = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_int, 10_000L);
        mirror_edge_buffer                   = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_int2, 24_000L);
        mirror_edge_flag_buffer              = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_int, 24_000L);
        mirror_hull_buffer                   = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_float4, 10_000L);
        mirror_hull_aabb_buffer              = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_float4, 10_000L);
        mirror_hull_flag_buffer              = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_int, 10_000L);
        mirror_hull_entity_id_buffer         = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_int, 10_000L);
        mirror_hull_mesh_id_buffer           = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_int, 10_000L);
        mirror_hull_uv_offset_buffer         = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_int, 10_000L);
        mirror_hull_integrity_buffer         = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_int, 10_000L);
        mirror_hull_point_table_buffer       = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_int2, 10_000L);
        mirror_hull_rotation_buffer          = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_float2, 10_000L);
        mirror_hull_scale_buffer             = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_float2, 10_000L);
        mirror_point_hit_count_buffer        = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_ushort, 50_000L);
        mirror_point_anti_gravity_buffer     = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_float, 50_000L);
        mirror_point_buffer                  = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_float4, 50_000L);
        mirror_point_vertex_reference_buffer = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_int, 50_000L);

        gpu_crud.init();
        scan_deletes.init();

        // create methods

        long create_point_k_ptr = gpu_crud.kernel_ptr(Kernel.create_point);
        create_point_k = new CreatePoint_k(GPGPU.cl_cmd_queue_ptr, create_point_k_ptr)
            .buf_arg(CreatePoint_k.Args.points, point_buffer)
            .buf_arg(CreatePoint_k.Args.point_vertex_references, point_vertex_reference_buffer)
            .buf_arg(CreatePoint_k.Args.point_hull_indices, point_hull_index_buffer)
            .buf_arg(CreatePoint_k.Args.point_flags, point_flag_buffer)
            .buf_arg(CreatePoint_k.Args.bone_tables, point_bone_table_buffer);

        long create_texture_uv_ptr = gpu_crud.kernel_ptr(Kernel.create_texture_uv);
        create_texture_uv_k = new CreateTextureUV_k(GPGPU.cl_cmd_queue_ptr, create_texture_uv_ptr)
            .buf_arg(CreateTextureUV_k.Args.texture_uvs, vertex_texture_uv_buffer);

        long create_edge_k_ptr = gpu_crud.kernel_ptr(Kernel.create_edge);
        create_edge_k = new CreateEdge_k(GPGPU.cl_cmd_queue_ptr, create_edge_k_ptr)
            .buf_arg(CreateEdge_k.Args.edges, edge_buffer)
            .buf_arg(CreateEdge_k.Args.edge_lengths, edge_length_buffer)
            .buf_arg(CreateEdge_k.Args.edge_flags, edge_flag_buffer);

        long create_keyframe_k_ptr = gpu_crud.kernel_ptr(Kernel.create_keyframe);
        create_keyframe_k = new CreateKeyFrame_k(GPGPU.cl_cmd_queue_ptr, create_keyframe_k_ptr)
            .buf_arg(CreateKeyFrame_k.Args.key_frames, anim_key_frame_buffer)
            .buf_arg(CreateKeyFrame_k.Args.frame_times, anim_frame_time_buffer);

        long create_vertex_reference_k_ptr = gpu_crud.kernel_ptr(Kernel.create_vertex_reference);
        create_vertex_reference_k = new CreateVertexRef_k(GPGPU.cl_cmd_queue_ptr, create_vertex_reference_k_ptr)
            .buf_arg(CreateVertexRef_k.Args.vertex_references, vertex_reference_buffer)
            .buf_arg(CreateVertexRef_k.Args.vertex_weights, vertex_weight_buffer)
            .buf_arg(CreateVertexRef_k.Args.uv_tables, vertex_uv_table_buffer);

        long create_bone_bind_pose_k_ptr = gpu_crud.kernel_ptr(Kernel.create_bone_bind_pose);
        create_bone_bind_pose_k = new CreateBoneBindPose_k(GPGPU.cl_cmd_queue_ptr, create_bone_bind_pose_k_ptr)
            .buf_arg(CreateBoneBindPose_k.Args.bone_bind_poses, bone_bind_pose_buffer);

        long create_bone_reference_k_ptr = gpu_crud.kernel_ptr(Kernel.create_bone_reference);
        create_bone_reference_k = new CreateBoneRef_k(GPGPU.cl_cmd_queue_ptr, create_bone_reference_k_ptr)
            .buf_arg(CreateBoneRef_k.Args.bone_references, bone_reference_buffer);

        long create_bone_channel_k_ptr = gpu_crud.kernel_ptr(Kernel.create_bone_channel);
        create_bone_channel_k = new CreateBoneChannel_k(GPGPU.cl_cmd_queue_ptr, create_bone_channel_k_ptr)
            .buf_arg(CreateBoneChannel_k.Args.animation_timing_indices, anim_timing_index_buffer)
            .buf_arg(CreateBoneChannel_k.Args.bone_pos_channel_tables, anim_bone_pos_channel_buffer)
            .buf_arg(CreateBoneChannel_k.Args.bone_rot_channel_tables, anim_bone_rot_channel_buffer)
            .buf_arg(CreateBoneChannel_k.Args.bone_scl_channel_tables, anim_bone_scl_channel_buffer);

        long create_entity_k_ptr = gpu_crud.kernel_ptr(Kernel.create_entity);
        create_entity_k = new CreateEntity_k(GPGPU.cl_cmd_queue_ptr, create_entity_k_ptr)
            .buf_arg(CreateEntity_k.Args.entities, entity_buffer)
            .buf_arg(CreateEntity_k.Args.entity_root_hulls, entity_root_hull_buffer)
            .buf_arg(CreateEntity_k.Args.entity_model_indices, entity_model_id_buffer)
            .buf_arg(CreateEntity_k.Args.entity_model_transforms, entity_model_transform_buffer)
            .buf_arg(CreateEntity_k.Args.entity_flags, entity_flag_buffer)
            .buf_arg(CreateEntity_k.Args.entity_hull_tables, entity_hull_table_buffer)
            .buf_arg(CreateEntity_k.Args.entity_bone_tables, entity_bone_table_buffer)
            .buf_arg(CreateEntity_k.Args.entity_masses, entity_mass_buffer)
            .buf_arg(CreateEntity_k.Args.entity_animation_indices, entity_anim_index_buffer)
            .buf_arg(CreateEntity_k.Args.entity_animation_elapsed, entity_anim_elapsed_buffer)
            .buf_arg(CreateEntity_k.Args.entity_motion_states, entity_motion_state_buffer);

        long create_bone_k_ptr = gpu_crud.kernel_ptr(Kernel.create_hull_bone);
        create_bone_k = new CreateHullBone_k(GPGPU.cl_cmd_queue_ptr, create_bone_k_ptr)
            .buf_arg(CreateHullBone_k.Args.bones, hull_bone_b)
            .buf_arg(CreateHullBone_k.Args.hull_bind_pose_indicies, hull_bone_bind_pose_id_b)
            .buf_arg(CreateHullBone_k.Args.hull_inv_bind_pose_indicies, hull_bone_inv_bind_pose_id_b);

        long create_armature_bone_k_ptr = gpu_crud.kernel_ptr(Kernel.create_armature_bone);
        create_armature_bone_k = new CreateArmatureBone_k(GPGPU.cl_cmd_queue_ptr, create_armature_bone_k_ptr)
            .buf_arg(CreateArmatureBone_k.Args.armature_bones, armature_bone_buffer)
            .buf_arg(CreateArmatureBone_k.Args.armature_bone_reference_ids, armature_bone_reference_id_buffer)
            .buf_arg(CreateArmatureBone_k.Args.armature_bone_parent_ids, armature_bone_parent_id_buffer);

        long create_model_transform_k_ptr = gpu_crud.kernel_ptr(Kernel.create_model_transform);
        create_model_transform_k = new CreateModelTransform_k(GPGPU.cl_cmd_queue_ptr, create_model_transform_k_ptr)
            .buf_arg(CreateModelTransform_k.Args.model_transforms, model_transform_buffer);

        long create_hull_k_ptr = gpu_crud.kernel_ptr(Kernel.create_hull);
        create_hull_k = new CreateHull_k(GPGPU.cl_cmd_queue_ptr, create_hull_k_ptr)
            .buf_arg(CreateHull_k.Args.hulls, hull_b)
            .buf_arg(CreateHull_k.Args.hull_scales, hull_scale_b)
            .buf_arg(CreateHull_k.Args.hull_rotations, hull_rotation_b)
            .buf_arg(CreateHull_k.Args.hull_frictions, hull_friction_b)
            .buf_arg(CreateHull_k.Args.hull_restitutions, hull_restitution_b)
            .buf_arg(CreateHull_k.Args.hull_point_tables, hull_point_table_b)
            .buf_arg(CreateHull_k.Args.hull_edge_tables, hull_edge_table_b)
            .buf_arg(CreateHull_k.Args.hull_bone_tables, hull_bone_table_b)
            .buf_arg(CreateHull_k.Args.hull_entity_ids, hull_entity_id_b)
            .buf_arg(CreateHull_k.Args.hull_flags, hull_flag_b)
            .buf_arg(CreateHull_k.Args.hull_mesh_ids, hull_mesh_id_b)
            .buf_arg(CreateHull_k.Args.hull_uv_offsets, hull_uv_offset_b)
            .buf_arg(CreateHull_k.Args.hull_integrity, hull_integrity_b);

        long create_mesh_reference_k_ptr = gpu_crud.kernel_ptr(Kernel.create_mesh_reference);
        create_mesh_reference_k = new CreateMeshReference_k(GPGPU.cl_cmd_queue_ptr, create_mesh_reference_k_ptr)
            .buf_arg(CreateMeshReference_k.Args.mesh_vertex_tables, mesh_vertex_table_buffer)
            .buf_arg(CreateMeshReference_k.Args.mesh_face_tables, mesh_face_table_buffer);

        long create_mesh_face_k_ptr = gpu_crud.kernel_ptr(Kernel.create_mesh_face);
        create_mesh_face_k = new CreateMeshFace_k(GPGPU.cl_cmd_queue_ptr, create_mesh_face_k_ptr)
            .buf_arg(CreateMeshFace_k.Args.mesh_faces, mesh_face_buffer);

        long create_animation_timings_k_ptr = gpu_crud.kernel_ptr(Kernel.create_animation_timings);
        create_animation_timings_k = new CreateAnimationTimings_k(GPGPU.cl_cmd_queue_ptr, create_animation_timings_k_ptr)
            .buf_arg(CreateAnimationTimings_k.Args.animation_durations, anim_duration_buffer)
            .buf_arg(CreateAnimationTimings_k.Args.animation_tick_rates, anim_tick_rate_buffer);

        // read methods

        long read_position_k_ptr = gpu_crud.kernel_ptr(Kernel.read_position);
        read_position_k = new ReadPosition_k(GPGPU.cl_cmd_queue_ptr, read_position_k_ptr)
            .buf_arg(ReadPosition_k.Args.entities, entity_buffer);

        // update methods

        long update_accel_k_ptr = gpu_crud.kernel_ptr(Kernel.update_accel);
        update_accel_k = new UpdateAccel_k(GPGPU.cl_cmd_queue_ptr, update_accel_k_ptr)
            .buf_arg(UpdateAccel_k.Args.entity_accel, entity_accel_buffer);

        long update_mouse_position_k_ptr = gpu_crud.kernel_ptr(Kernel.update_mouse_position);
        update_mouse_position_k = new UpdateMousePosition_k(GPGPU.cl_cmd_queue_ptr, update_mouse_position_k_ptr)
            .buf_arg(UpdateMousePosition_k.Args.entity_root_hulls, entity_root_hull_buffer)
            .buf_arg(UpdateMousePosition_k.Args.hull_point_tables, hull_point_table_b)
            .buf_arg(UpdateMousePosition_k.Args.points, point_buffer);

        long set_bone_channel_table_k_ptr = gpu_crud.kernel_ptr(Kernel.set_bone_channel_table);
        set_bone_channel_table_k = new SetBoneChannelTable_k(GPGPU.cl_cmd_queue_ptr, set_bone_channel_table_k_ptr)
            .buf_arg(SetBoneChannelTable_k.Args.bone_channel_tables, bone_anim_channel_table_buffer);

        // delete methods

        long locate_out_of_bounds_k_ptr = scan_deletes.kernel_ptr(Kernel.locate_out_of_bounds);
        locate_out_of_bounds_k = new LocateOutOfBounds_k(GPGPU.cl_cmd_queue_ptr, locate_out_of_bounds_k_ptr)
            .buf_arg(LocateOutOfBounds_k.Args.hull_tables, entity_hull_table_buffer)
            .buf_arg(LocateOutOfBounds_k.Args.hull_flags, hull_flag_b)
            .buf_arg(LocateOutOfBounds_k.Args.entity_flags, entity_flag_buffer);

        long scan_deletes_single_block_out_k_ptr = scan_deletes.kernel_ptr(Kernel.scan_deletes_single_block_out);
        scan_deletes_single_block_out_k = new ScanDeletesSingleBlockOut_k(GPGPU.cl_cmd_queue_ptr, scan_deletes_single_block_out_k_ptr)
            .ptr_arg(ScanDeletesSingleBlockOut_k.Args.sz, delete_sizes_ptr)
            .buf_arg(ScanDeletesSingleBlockOut_k.Args.entity_flags, entity_flag_buffer)
            .buf_arg(ScanDeletesSingleBlockOut_k.Args.hull_tables, entity_hull_table_buffer)
            .buf_arg(ScanDeletesSingleBlockOut_k.Args.bone_tables, entity_bone_table_buffer)
            .buf_arg(ScanDeletesSingleBlockOut_k.Args.point_tables, hull_point_table_b)
            .buf_arg(ScanDeletesSingleBlockOut_k.Args.edge_tables, hull_edge_table_b)
            .buf_arg(ScanDeletesSingleBlockOut_k.Args.hull_bone_tables, hull_bone_table_b);

        long scan_deletes_multi_block_out_k_ptr = scan_deletes.kernel_ptr(Kernel.scan_deletes_multi_block_out);
        scan_deletes_multi_block_out_k = new ScanDeletesMultiBlockOut_k(GPGPU.cl_cmd_queue_ptr, scan_deletes_multi_block_out_k_ptr)
            .buf_arg(ScanDeletesMultiBlockOut_k.Args.part1, delete_partial_buffer_1)
            .buf_arg(ScanDeletesMultiBlockOut_k.Args.part2, delete_partial_buffer_2)
            .buf_arg(ScanDeletesMultiBlockOut_k.Args.entity_flags, entity_flag_buffer)
            .buf_arg(ScanDeletesMultiBlockOut_k.Args.hull_tables, entity_hull_table_buffer)
            .buf_arg(ScanDeletesMultiBlockOut_k.Args.bone_tables, entity_bone_table_buffer)
            .buf_arg(ScanDeletesMultiBlockOut_k.Args.point_tables, hull_point_table_b)
            .buf_arg(ScanDeletesMultiBlockOut_k.Args.edge_tables, hull_edge_table_b)
            .buf_arg(ScanDeletesMultiBlockOut_k.Args.hull_bone_tables, hull_bone_table_b);

        long complete_deletes_multi_block_out_k_ptr = scan_deletes.kernel_ptr(Kernel.complete_deletes_multi_block_out);
        complete_deletes_multi_block_out_k = new CompleteDeletesMultiBlockOut_k(GPGPU.cl_cmd_queue_ptr, complete_deletes_multi_block_out_k_ptr)
            .ptr_arg(CompleteDeletesMultiBlockOut_k.Args.sz, delete_sizes_ptr)
            .buf_arg(CompleteDeletesMultiBlockOut_k.Args.part1, delete_partial_buffer_1)
            .buf_arg(CompleteDeletesMultiBlockOut_k.Args.part2, delete_partial_buffer_2)
            .buf_arg(CompleteDeletesMultiBlockOut_k.Args.entity_flags, entity_flag_buffer)
            .buf_arg(CompleteDeletesMultiBlockOut_k.Args.hull_tables, entity_hull_table_buffer)
            .buf_arg(CompleteDeletesMultiBlockOut_k.Args.bone_tables, entity_bone_table_buffer)
            .buf_arg(CompleteDeletesMultiBlockOut_k.Args.point_tables, hull_point_table_b)
            .buf_arg(CompleteDeletesMultiBlockOut_k.Args.edge_tables, hull_edge_table_b)
            .buf_arg(CompleteDeletesMultiBlockOut_k.Args.hull_bone_tables, hull_bone_table_b);

        long compact_entities_k_ptr = scan_deletes.kernel_ptr(Kernel.compact_entities);
        compact_entities_k = new CompactEntities_k(GPGPU.cl_cmd_queue_ptr, compact_entities_k_ptr)
            .buf_arg(CompactEntities_k.Args.entities, entity_buffer)
            .buf_arg(CompactEntities_k.Args.entity_masses, entity_mass_buffer)
            .buf_arg(CompactEntities_k.Args.entity_root_hulls, entity_root_hull_buffer)
            .buf_arg(CompactEntities_k.Args.entity_model_indices, entity_model_id_buffer)
            .buf_arg(CompactEntities_k.Args.entity_model_transforms, entity_model_transform_buffer)
            .buf_arg(CompactEntities_k.Args.entity_flags, entity_flag_buffer)
            .buf_arg(CompactEntities_k.Args.entity_animation_indices, entity_anim_index_buffer)
            .buf_arg(CompactEntities_k.Args.entity_animation_elapsed, entity_anim_elapsed_buffer)
            .buf_arg(CompactEntities_k.Args.entity_animation_blend, entity_anim_blend_buffer)
            .buf_arg(CompactEntities_k.Args.entity_motion_states, entity_motion_state_buffer)
            .buf_arg(CompactEntities_k.Args.entity_entity_hull_tables, entity_hull_table_buffer)
            .buf_arg(CompactEntities_k.Args.entity_bone_tables, entity_bone_table_buffer)
            .buf_arg(CompactEntities_k.Args.hull_bone_tables, hull_bone_table_b)
            .buf_arg(CompactEntities_k.Args.hull_entity_ids, hull_entity_id_b)
            .buf_arg(CompactEntities_k.Args.hull_point_tables, hull_point_table_b)
            .buf_arg(CompactEntities_k.Args.hull_edge_tables, hull_edge_table_b)
            .buf_arg(CompactEntities_k.Args.points, point_buffer)
            .buf_arg(CompactEntities_k.Args.point_hull_indices, point_hull_index_buffer)
            .buf_arg(CompactEntities_k.Args.point_bone_tables, point_bone_table_buffer)
            .buf_arg(CompactEntities_k.Args.armature_bone_parent_ids, armature_bone_parent_id_buffer)
            .buf_arg(CompactEntities_k.Args.hull_bind_pose_indices, hull_bone_bind_pose_id_b)
            .buf_arg(CompactEntities_k.Args.edges, edge_buffer)
            .buf_arg(CompactEntities_k.Args.hull_bone_shift, hull_bone_shift)
            .buf_arg(CompactEntities_k.Args.point_shift, point_shift)
            .buf_arg(CompactEntities_k.Args.edge_shift, edge_shift)
            .buf_arg(CompactEntities_k.Args.hull_shift, hull_shift)
            .buf_arg(CompactEntities_k.Args.armature_bone_shift, armature_bone_shift);

        long compact_hulls_k_ptr = scan_deletes.kernel_ptr(Kernel.compact_hulls);
        compact_hulls_k = new CompactHulls_k(GPGPU.cl_cmd_queue_ptr, compact_hulls_k_ptr)
            .buf_arg(CompactHulls_k.Args.hull_shift, hull_shift)
            .buf_arg(CompactHulls_k.Args.hulls, hull_b)
            .buf_arg(CompactHulls_k.Args.hull_scales, hull_scale_b)
            .buf_arg(CompactHulls_k.Args.hull_mesh_ids, hull_mesh_id_b)
            .buf_arg(CompactHulls_k.Args.hull_uv_offsets, hull_uv_offset_b)
            .buf_arg(CompactHulls_k.Args.hull_rotations, hull_rotation_b)
            .buf_arg(CompactHulls_k.Args.hull_frictions, hull_friction_b)
            .buf_arg(CompactHulls_k.Args.hull_restitutions, hull_restitution_b)
            .buf_arg(CompactHulls_k.Args.hull_integrity, hull_integrity_b)
            .buf_arg(CompactHulls_k.Args.hull_bone_tables, hull_bone_table_b)
            .buf_arg(CompactHulls_k.Args.hull_entity_ids, hull_entity_id_b)
            .buf_arg(CompactHulls_k.Args.hull_flags, hull_flag_b)
            .buf_arg(CompactHulls_k.Args.hull_point_tables, hull_point_table_b)
            .buf_arg(CompactHulls_k.Args.hull_edge_tables, hull_edge_table_b)
            .buf_arg(CompactHulls_k.Args.bounds, hull_aabb_b)
            .buf_arg(CompactHulls_k.Args.bounds_index_data, hull_aabb_index_b)
            .buf_arg(CompactHulls_k.Args.bounds_bank_data, hull_aabb_key_b);

        long compact_edges_k_ptr = scan_deletes.kernel_ptr(Kernel.compact_edges);
        compact_edges_k = new CompactEdges_k(GPGPU.cl_cmd_queue_ptr, compact_edges_k_ptr)
            .buf_arg(CompactEdges_k.Args.edge_shift, edge_shift)
            .buf_arg(CompactEdges_k.Args.edges, edge_buffer)
            .buf_arg(CompactEdges_k.Args.edge_lengths, edge_length_buffer)
            .buf_arg(CompactEdges_k.Args.edge_flags, edge_flag_buffer);

        long compact_points_k_ptr = scan_deletes.kernel_ptr(Kernel.compact_points);
        compact_points_k = new CompactPoints_k(GPGPU.cl_cmd_queue_ptr, compact_points_k_ptr)
            .buf_arg(CompactPoints_k.Args.point_shift, point_shift)
            .buf_arg(CompactPoints_k.Args.points, point_buffer)
            .buf_arg(CompactPoints_k.Args.anti_gravity, point_anti_gravity_buffer)
            .buf_arg(CompactPoints_k.Args.point_vertex_references, point_vertex_reference_buffer)
            .buf_arg(CompactPoints_k.Args.point_hull_indices, point_hull_index_buffer)
            .buf_arg(CompactPoints_k.Args.point_flags, point_flag_buffer)
            .buf_arg(CompactPoints_k.Args.point_hit_counts, point_hit_count_buffer)
            .buf_arg(CompactPoints_k.Args.bone_tables, point_bone_table_buffer);

        long compact_hull_bones_k_ptr = scan_deletes.kernel_ptr(Kernel.compact_hull_bones);
        compact_hull_bones_k = new CompactHullBones_k(GPGPU.cl_cmd_queue_ptr, compact_hull_bones_k_ptr)
            .buf_arg(CompactHullBones_k.Args.hull_bone_shift, hull_bone_shift)
            .buf_arg(CompactHullBones_k.Args.bone_instances, hull_bone_b)
            .buf_arg(CompactHullBones_k.Args.hull_bind_pose_indicies, hull_bone_bind_pose_id_b)
            .buf_arg(CompactHullBones_k.Args.hull_inv_bind_pose_indicies, hull_bone_inv_bind_pose_id_b);

        long compact_armature_bones_k_ptr = scan_deletes.kernel_ptr(Kernel.compact_armature_bones);
        compact_armature_bones_k = new CompactArmatureBones_k(GPGPU.cl_cmd_queue_ptr, compact_armature_bones_k_ptr)
            .buf_arg(CompactArmatureBones_k.Args.armature_bone_shift, armature_bone_shift)
            .buf_arg(CompactArmatureBones_k.Args.armature_bones, armature_bone_buffer)
            .buf_arg(CompactArmatureBones_k.Args.armature_bone_reference_ids, armature_bone_reference_id_buffer)
            .buf_arg(CompactArmatureBones_k.Args.armature_bone_parent_ids, armature_bone_parent_id_buffer);
    }

    public ResizableBuffer buffer(BufferType bufferType)
    {
        return switch (bufferType)
        {
            case ANIM_FRAME_TIME               -> anim_frame_time_buffer;
            case ANIM_KEY_FRAME                -> anim_key_frame_buffer;
            case ANIM_POS_CHANNEL              -> anim_bone_pos_channel_buffer;
            case ANIM_ROT_CHANNEL              -> anim_bone_rot_channel_buffer;
            case ANIM_SCL_CHANNEL              -> anim_bone_scl_channel_buffer;
            case ANIM_DURATION                 -> anim_duration_buffer;
            case ANIM_TICK_RATE                -> anim_tick_rate_buffer;
            case ANIM_TIMING_INDEX             -> anim_timing_index_buffer;
            case ENTITY -> entity_buffer;
            case ENTITY_ACCEL -> entity_accel_buffer;
            case ENTITY_ANIM_BLEND -> entity_anim_blend_buffer;
            case ENTITY_ANIM_ELAPSED -> entity_anim_elapsed_buffer;
            case ENTITY_MOTION_STATE -> entity_motion_state_buffer;
            case ENTITY_ANIM_INDEX -> entity_anim_index_buffer;
            case ARMATURE_BONE                 -> armature_bone_buffer;
            case ARMATURE_BONE_REFERENCE_ID    -> armature_bone_reference_id_buffer;
            case ARMATURE_BONE_PARENT_ID       -> armature_bone_parent_id_buffer;
            case ENTITY_FLAG -> entity_flag_buffer;
            case ENTITY_BONE_TABLE -> entity_bone_table_buffer;
            case ENTITY_HULL_TABLE -> entity_hull_table_buffer;
            case ENTITY_MASS -> entity_mass_buffer;
            case ENTITY_MODEL_ID -> entity_model_id_buffer;
            case ENTITY_ROOT_HULL -> entity_root_hull_buffer;
            case ENTITY_TRANSFORM_ID -> entity_model_transform_buffer;
            case BONE_ANIM_TABLE               -> bone_anim_channel_table_buffer;
            case BONE_BIND_POSE                -> bone_bind_pose_buffer;
            case BONE_REFERENCE                -> bone_reference_buffer;
            case EDGE                          -> edge_buffer;
            case EDGE_FLAG                     -> edge_flag_buffer;
            case EDGE_LENGTH                   -> edge_length_buffer;
            case HULL                          -> hull_b;
            case HULL_SCALE                    -> hull_scale_b;
            case HULL_AABB                     -> hull_aabb_b;
            case HULL_AABB_INDEX               -> hull_aabb_index_b;
            case HULL_AABB_KEY_TABLE           -> hull_aabb_key_b;
            case HULL_BONE                     -> hull_bone_b;
            case HULL_ENTITY_ID                -> hull_entity_id_b;
            case HULL_BONE_TABLE               -> hull_bone_table_b;
            case HULL_BONE_BIND_POSE           -> hull_bone_bind_pose_id_b;
            case HULL_BONE_INV_BIND_POSE       -> hull_bone_inv_bind_pose_id_b;
            case HULL_POINT_TABLE              -> hull_point_table_b;
            case HULL_EDGE_TABLE               -> hull_edge_table_b;
            case HULL_FLAG                     -> hull_flag_b;
            case HULL_FRICTION                 -> hull_friction_b;
            case HULL_RESTITUTION              -> hull_restitution_b;
            case HULL_INTEGRITY                -> hull_integrity_b;
            case HULL_MESH_ID                  -> hull_mesh_id_b;
            case HULL_UV_OFFSET                -> hull_uv_offset_b;
            case HULL_ROTATION                 -> hull_rotation_b;
            case MESH_FACE                     -> mesh_face_buffer;
            case MESH_VERTEX_TABLE             -> mesh_vertex_table_buffer;
            case MESH_FACE_TABLE               -> mesh_face_table_buffer;
            case MODEL_TRANSFORM               -> model_transform_buffer;
            case POINT                         -> point_buffer;
            case POINT_ANTI_GRAV               -> point_anti_gravity_buffer;
            case POINT_BONE_TABLE              -> point_bone_table_buffer;
            case POINT_VERTEX_REFERENCE        -> point_vertex_reference_buffer;
            case POINT_HULL_INDEX              -> point_hull_index_buffer;
            case POINT_FLAG                    -> point_flag_buffer;
            case POINT_HIT_COUNT               -> point_hit_count_buffer;
            case VERTEX_REFERENCE              -> vertex_reference_buffer;
            case VERTEX_TEXTURE_UV             -> vertex_texture_uv_buffer;
            case VERTEX_UV_TABLE               -> vertex_uv_table_buffer;
            case VERTEX_WEIGHT                 -> vertex_weight_buffer;

            case MIRROR_EDGE                   -> mirror_edge_buffer;
            case MIRROR_HULL                   -> mirror_hull_buffer;
            case MIRROR_ENTITY                 -> mirror_entity_buffer;
            case MIRROR_ENTITY_FLAG            -> mirror_entity_flag_buffer;
            case MIRROR_POINT                  -> mirror_point_buffer;
            case MIRROR_ENTITY_MODEL_ID        -> mirror_entity_model_id_buffer;
            case MIRROR_ENTITY_ROOT_HULL       -> mirror_entity_root_hull_buffer;
            case MIRROR_EDGE_FLAG              -> mirror_edge_flag_buffer;
            case MIRROR_HULL_AABB              -> mirror_hull_aabb_buffer;
            case MIRROR_HULL_ENTITY_ID         -> mirror_hull_entity_id_buffer;
            case MIRROR_HULL_FLAG              -> mirror_hull_flag_buffer;
            case MIRROR_HULL_MESH_ID           -> mirror_hull_mesh_id_buffer;
            case MIRROR_HULL_UV_OFFSET         -> mirror_hull_uv_offset_buffer;
            case MIRROR_HULL_INTEGRITY         -> mirror_hull_integrity_buffer;
            case MIRROR_HULL_POINT_TABLE       -> mirror_hull_point_table_buffer;
            case MIRROR_HULL_ROTATION          -> mirror_hull_rotation_buffer;
            case MIRROR_HULL_SCALE             -> mirror_hull_scale_buffer;
            case MIRROR_POINT_ANTI_GRAV        -> mirror_point_anti_gravity_buffer;
            case MIRROR_POINT_HIT_COUNT        -> mirror_point_hit_count_buffer;
            case MIRROR_POINT_VERTEX_REFERENCE -> mirror_point_vertex_reference_buffer;
        };
    }

    public void mirror_buffers_ex()
    {
        mirror_entity_buffer.mirror_buffer(entity_buffer);
        mirror_entity_flag_buffer.mirror_buffer(entity_flag_buffer);
        mirror_entity_model_id_buffer.mirror_buffer(entity_model_id_buffer);
        mirror_entity_root_hull_buffer.mirror_buffer(entity_root_hull_buffer);
        mirror_edge_buffer.mirror_buffer(edge_buffer);
        mirror_edge_flag_buffer.mirror_buffer(edge_flag_buffer);
        mirror_hull_buffer.mirror_buffer(hull_b);
        mirror_hull_aabb_buffer.mirror_buffer(hull_aabb_b);
        mirror_hull_entity_id_buffer.mirror_buffer(hull_entity_id_b);
        mirror_hull_flag_buffer.mirror_buffer(hull_flag_b);
        mirror_hull_mesh_id_buffer.mirror_buffer(hull_mesh_id_b);
        mirror_hull_uv_offset_buffer.mirror_buffer(hull_uv_offset_b);
        mirror_hull_integrity_buffer.mirror_buffer(hull_integrity_b);
        mirror_hull_point_table_buffer.mirror_buffer(hull_point_table_b);
        mirror_hull_rotation_buffer.mirror_buffer(hull_rotation_b);
        mirror_hull_scale_buffer.mirror_buffer(hull_scale_b);
        mirror_point_hit_count_buffer.mirror_buffer(point_hit_count_buffer);
        mirror_point_anti_gravity_buffer.mirror_buffer(point_anti_gravity_buffer);
        mirror_point_buffer.mirror_buffer(point_buffer);
        mirror_point_vertex_reference_buffer.mirror_buffer(point_vertex_reference_buffer);

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


    private int vec2_stride  = 2;
    private int vec4_stride  = 4;
    private int vec16_stride = 16;

    public void process_entity_batch(PhysicsEntityBatch batch)
    {
        int edge_count = batch.edges.size();
        if (edge_count > 0)
        {
            int edge_capacity = edge_index + edge_count;
            edge_buffer.ensure_capacity(edge_capacity);
            edge_length_buffer.ensure_capacity(edge_capacity);
            edge_flag_buffer.ensure_capacity(edge_capacity);
            int[] edge_copy = new int[edge_count * vec2_stride];
            float[] edge_length_copy = new float[edge_count];
            int[] edge_flag_copy = new int[edge_count];
            int next_edge = 0;
            int next_edge_2 = 0;
            for (var edge : batch.edges)
            {
                edge_copy[next_edge_2] = edge.p1();
                edge_copy[next_edge_2 + 1] = edge.p2();
                edge_length_copy[next_edge] = edge.l();
                edge_flag_copy[next_edge] = edge.flags();
                next_edge_2 += vec2_stride;
                next_edge++;
            }
            GPGPU.cl_write_int_buffer(GPGPU.cl_cmd_queue_ptr, edge_buffer.buffer_pointer, edge_index, edge_copy);
            GPGPU.cl_write_float_buffer(GPGPU.cl_cmd_queue_ptr, edge_length_buffer.buffer_pointer, edge_index, edge_length_copy);
            GPGPU.cl_write_int_buffer(GPGPU.cl_cmd_queue_ptr, edge_flag_buffer.buffer_pointer, edge_index, edge_flag_copy);
            edge_index += edge_count;
        }


        int point_count = batch.points.size();
        if (point_count > 0)
        {
            int point_capacity = point_index + point_count;
            point_buffer.ensure_capacity(point_capacity);
            point_vertex_reference_buffer.ensure_capacity(point_capacity);
            point_hull_index_buffer.ensure_capacity(point_capacity);
            point_flag_buffer.ensure_capacity(point_capacity);
            point_bone_table_buffer.ensure_capacity(point_capacity);
            point_anti_gravity_buffer.ensure_capacity(point_capacity); // transient
            point_hit_count_buffer.ensure_capacity(point_capacity);    // transient
            float[] point_copy = new float[point_count * vec4_stride];
            int[] point_vertex_ref_copy = new int[point_count];
            int[] point_hull_index_copy = new int[point_count];
            int[] point_flag_copy = new int[point_count];
            int[] point_bone_table_copy = new int[point_count * vec4_stride];
            int next_point = 0;
            int next_point_4 = 0;
            for (var point : batch.points)
            {
                point_copy[next_point_4] = point.position()[0];
                point_copy[next_point_4 + 1] = point.position()[1];
                point_copy[next_point_4 + 2] = point.position()[0];
                point_copy[next_point_4 + 3] = point.position()[1];
                point_bone_table_copy[next_point_4] = point.bone_ids()[0];
                point_bone_table_copy[next_point_4 + 1] = point.bone_ids()[1];
                point_bone_table_copy[next_point_4 + 2] = point.bone_ids()[2];
                point_bone_table_copy[next_point_4 + 3] = point.bone_ids()[3];
                point_vertex_ref_copy[next_point] = point.vertex_index();
                point_hull_index_copy[next_point] = point.hull_index();
                point_flag_copy[next_point] = point.flags();
                next_point_4 += vec4_stride;
                next_point++;
            }
            GPGPU.cl_write_float_buffer(GPGPU.cl_cmd_queue_ptr, point_buffer.buffer_pointer, point_index, point_copy);
            GPGPU.cl_write_int_buffer(GPGPU.cl_cmd_queue_ptr, point_bone_table_buffer.buffer_pointer, point_index, point_bone_table_copy);
            GPGPU.cl_write_int_buffer(GPGPU.cl_cmd_queue_ptr, point_vertex_reference_buffer.buffer_pointer, point_index, point_vertex_ref_copy);
            GPGPU.cl_write_int_buffer(GPGPU.cl_cmd_queue_ptr, point_hull_index_buffer.buffer_pointer, point_index, point_hull_index_copy);
            GPGPU.cl_write_int_buffer(GPGPU.cl_cmd_queue_ptr, point_flag_buffer.buffer_pointer, point_index, point_flag_copy);
            point_index += point_count;
        }


        int hull_count = batch.hulls.size();
        if (hull_count > 0)
        {
            int hull_capacity = hull_index + hull_count;
            hull_b.ensure_capacity(hull_capacity);
            hull_scale_b.ensure_capacity(hull_capacity);
            hull_rotation_b.ensure_capacity(hull_capacity);
            hull_friction_b.ensure_capacity(hull_capacity);
            hull_restitution_b.ensure_capacity(hull_capacity);
            hull_point_table_b.ensure_capacity(hull_capacity);
            hull_edge_table_b.ensure_capacity(hull_capacity);
            hull_bone_table_b.ensure_capacity(hull_capacity);
            hull_entity_id_b.ensure_capacity(hull_capacity);
            hull_flag_b.ensure_capacity(hull_capacity);
            hull_mesh_id_b.ensure_capacity(hull_capacity);
            hull_uv_offset_b.ensure_capacity(hull_capacity);
            hull_integrity_b.ensure_capacity(hull_capacity);
            hull_aabb_b.ensure_capacity(hull_capacity);       // transient
            hull_aabb_index_b.ensure_capacity(hull_capacity); // transient
            hull_aabb_key_b.ensure_capacity(hull_capacity);   // transient
            float[] hulls_copy = new float[hull_count * vec4_stride];
            float[] hull_scales_copy = new float[hull_count * vec2_stride];
            float[] hull_rotations_copy = new float[hull_count * vec2_stride];
            float[] hull_frictions_copy = new float[hull_count];
            float[] hull_restitutions_copy = new float[hull_count];
            int[] hull_point_tables_copy = new int[hull_count * vec2_stride];
            int[] hull_edge_tables_copy = new int[hull_count * vec2_stride];
            int[] bone_tables_copy = new int[hull_count * vec2_stride];
            int[] hull_entity_ids_copy = new int[hull_count];
            int[] hull_flags_copy = new int[hull_count];
            int[] hull_mesh_ids_copy = new int[hull_count];
            int[] hull_uv_offsets_copy = new int[hull_count];
            int[] hull_integrity_copy = new int[hull_count];
            int next_hull = 0;
            int next_hull_2 = 0;
            int next_hull_4 = 0;
            for (var hull : batch.hulls)
            {
                hulls_copy[next_hull_4] = hull.position()[0];
                hulls_copy[next_hull_4 + 1] = hull.position()[1];
                hulls_copy[next_hull_4 + 2] = hull.position()[0];
                hulls_copy[next_hull_4 + 3] = hull.position()[1];
                hull_scales_copy[next_hull_2] = hull.scale()[0];
                hull_scales_copy[next_hull_2 + 1] = hull.scale()[1];
                hull_rotations_copy[next_hull_2] = hull.rotation()[0];
                hull_rotations_copy[next_hull_2 + 1] = hull.rotation()[1];
                hull_frictions_copy[next_hull] = hull.friction();
                hull_restitutions_copy[next_hull] = hull.restitution();
                hull_point_tables_copy[next_hull_2] = hull.point_table()[0];
                hull_point_tables_copy[next_hull_2 + 1] = hull.point_table()[1];
                hull_edge_tables_copy[next_hull_2] = hull.edge_table()[0];
                hull_edge_tables_copy[next_hull_2 + 1] = hull.edge_table()[1];
                bone_tables_copy[next_hull_2] = hull.bone_table()[0];
                bone_tables_copy[next_hull_2 + 1] = hull.bone_table()[1];
                hull_entity_ids_copy[next_hull] = hull.entity_id();
                hull_flags_copy[next_hull] = hull.flags();
                hull_mesh_ids_copy[next_hull] = hull.mesh_id();
                hull_uv_offsets_copy[next_hull] = hull.uv_offset();
                hull_integrity_copy[next_hull] = 100; // todo: this needs state save/load to work correctly
                next_hull_2 += vec2_stride;
                next_hull_4 += vec4_stride;
                next_hull++;
            }
            GPGPU.cl_write_float_buffer(GPGPU.cl_cmd_queue_ptr, hull_b.buffer_pointer, hull_index, hulls_copy);
            GPGPU.cl_write_float_buffer(GPGPU.cl_cmd_queue_ptr, hull_scale_b.buffer_pointer, hull_index, hull_scales_copy);
            GPGPU.cl_write_float_buffer(GPGPU.cl_cmd_queue_ptr, hull_rotation_b.buffer_pointer, hull_index, hull_rotations_copy);
            GPGPU.cl_write_float_buffer(GPGPU.cl_cmd_queue_ptr, hull_friction_b.buffer_pointer, hull_index, hull_frictions_copy);
            GPGPU.cl_write_float_buffer(GPGPU.cl_cmd_queue_ptr, hull_restitution_b.buffer_pointer, hull_index, hull_restitutions_copy);
            GPGPU.cl_write_int_buffer(GPGPU.cl_cmd_queue_ptr, hull_point_table_b.buffer_pointer, hull_index, hull_point_tables_copy);
            GPGPU.cl_write_int_buffer(GPGPU.cl_cmd_queue_ptr, hull_edge_table_b.buffer_pointer, hull_index, hull_edge_tables_copy);
            GPGPU.cl_write_int_buffer(GPGPU.cl_cmd_queue_ptr, hull_bone_table_b.buffer_pointer, hull_index, bone_tables_copy);
            GPGPU.cl_write_int_buffer(GPGPU.cl_cmd_queue_ptr, hull_entity_id_b.buffer_pointer, hull_index, hull_entity_ids_copy);
            GPGPU.cl_write_int_buffer(GPGPU.cl_cmd_queue_ptr, hull_flag_b.buffer_pointer, hull_index, hull_flags_copy);
            GPGPU.cl_write_int_buffer(GPGPU.cl_cmd_queue_ptr, hull_mesh_id_b.buffer_pointer, hull_index, hull_mesh_ids_copy);
            GPGPU.cl_write_int_buffer(GPGPU.cl_cmd_queue_ptr, hull_uv_offset_b.buffer_pointer, hull_index, hull_uv_offsets_copy);
            GPGPU.cl_write_int_buffer(GPGPU.cl_cmd_queue_ptr, hull_integrity_b.buffer_pointer, hull_index, hull_integrity_copy);
            hull_index += hull_count;
        }


        int entity_count = batch.entities.size();
        if (entity_count > 0)
        {
            int entity_capacity = entity_index + entity_count;
            entity_buffer.ensure_capacity(entity_capacity);
            entity_anim_elapsed_buffer.ensure_capacity(entity_capacity);
            entity_motion_state_buffer.ensure_capacity(entity_capacity);
            entity_anim_index_buffer.ensure_capacity(entity_capacity);
            entity_hull_table_buffer.ensure_capacity(entity_capacity);
            entity_bone_table_buffer.ensure_capacity(entity_capacity);
            entity_mass_buffer.ensure_capacity(entity_capacity);
            entity_root_hull_buffer.ensure_capacity(entity_capacity);
            entity_model_id_buffer.ensure_capacity(entity_capacity);
            entity_model_transform_buffer.ensure_capacity(entity_capacity);
            entity_flag_buffer.ensure_capacity(entity_capacity);
            entity_accel_buffer.ensure_capacity(entity_capacity);      // transient
            entity_anim_blend_buffer.ensure_capacity(entity_capacity); // transient
            float[] entities_copy = new float[entity_count * vec4_stride];
            float[] entity_animation_elapsed_copy = new float[entity_count * vec2_stride];
            short[] entity_motion_states_copy = new short[entity_count * vec2_stride];
            int[] entity_animation_indices_copy = new int[entity_count * vec2_stride];
            int[] entity_hull_tables_copy = new int[entity_count * vec2_stride];
            int[] entity_bone_tables_copy = new int[entity_count * vec2_stride];
            float[] entity_masses_copy = new float[entity_count];
            int[] entity_root_hulls_copy = new int[entity_count];
            int[] entity_model_indices_copy = new int[entity_count];
            int[] entity_model_transforms_copy = new int[entity_count];
            int[] entity_flags_copy = new int[entity_count];
            int next_entity = 0;
            int next_entity_2 = 0;
            int next_entity_4 = 0;
            for (var entity : batch.entities)
            {
                entities_copy[next_entity_4] = entity.x();
                entities_copy[next_entity_4 + 1] = entity.y();
                entities_copy[next_entity_4 + 2] = entity.x();
                entities_copy[next_entity_4 + 3] = entity.y();
                entity_animation_elapsed_copy[next_entity_2] = entity.anim_time();
                entity_animation_elapsed_copy[next_entity_2 + 1] = 0.0f;
                entity_motion_states_copy[next_entity_2] = (short) 0;
                entity_motion_states_copy[next_entity_2 + 1] = (short) 0;
                entity_animation_indices_copy[next_entity_2] = entity.anim_index();
                entity_animation_indices_copy[next_entity_2 + 1] = -1;
                entity_hull_tables_copy[next_entity_2] = entity.hull_table()[0];
                entity_hull_tables_copy[next_entity_2 + 1] = entity.hull_table()[1];
                entity_bone_tables_copy[next_entity_2] = entity.bone_table()[0];
                entity_bone_tables_copy[next_entity_2 + 1] = entity.bone_table()[1];
                entity_masses_copy[next_entity] = entity.mass();
                entity_root_hulls_copy[next_entity] = entity.root_hull();
                entity_model_indices_copy[next_entity] = entity.model_id();
                entity_model_transforms_copy[next_entity] = entity.model_transform_id();
                entity_flags_copy[next_entity] = entity.flags();
                next_entity_2 += vec2_stride;
                next_entity_4 += vec4_stride;
                next_entity++;
            }
            GPGPU.cl_write_float_buffer(GPGPU.cl_cmd_queue_ptr, entity_buffer.buffer_pointer, entity_index, entities_copy);
            GPGPU.cl_write_float_buffer(GPGPU.cl_cmd_queue_ptr, entity_anim_elapsed_buffer.buffer_pointer, entity_index, entity_animation_elapsed_copy);
            GPGPU.cl_write_short_buffer(GPGPU.cl_cmd_queue_ptr, entity_motion_state_buffer.buffer_pointer, entity_index, entity_motion_states_copy);
            GPGPU.cl_write_int_buffer(GPGPU.cl_cmd_queue_ptr, entity_anim_index_buffer.buffer_pointer, entity_index, entity_animation_indices_copy);
            GPGPU.cl_write_int_buffer(GPGPU.cl_cmd_queue_ptr, entity_hull_table_buffer.buffer_pointer, entity_index, entity_hull_tables_copy);
            GPGPU.cl_write_int_buffer(GPGPU.cl_cmd_queue_ptr, entity_bone_table_buffer.buffer_pointer, entity_index, entity_bone_tables_copy);
            GPGPU.cl_write_float_buffer(GPGPU.cl_cmd_queue_ptr, entity_mass_buffer.buffer_pointer, entity_index, entity_masses_copy);
            GPGPU.cl_write_int_buffer(GPGPU.cl_cmd_queue_ptr, entity_root_hull_buffer.buffer_pointer, entity_index, entity_root_hulls_copy);
            GPGPU.cl_write_int_buffer(GPGPU.cl_cmd_queue_ptr, entity_model_id_buffer.buffer_pointer, entity_index, entity_model_indices_copy);
            GPGPU.cl_write_int_buffer(GPGPU.cl_cmd_queue_ptr, entity_model_transform_buffer.buffer_pointer, entity_index, entity_model_transforms_copy);
            GPGPU.cl_write_int_buffer(GPGPU.cl_cmd_queue_ptr, entity_flag_buffer.buffer_pointer, entity_index, entity_flags_copy);
            entity_index += entity_count;
        }


        int hull_bone_count = batch.hull_bones.size();
        if (hull_bone_count > 0)
        {
            int hull_bone_capacity = hull_bone_index + hull_bone_count;
            hull_bone_b.ensure_capacity(hull_bone_capacity);
            hull_bone_bind_pose_id_b.ensure_capacity(hull_bone_capacity);
            hull_bone_inv_bind_pose_id_b.ensure_capacity(hull_bone_capacity);
            float[] hull_bones_copy = new float[hull_bone_count * vec16_stride];
            int[] hull_bone_bind_pose_id_copy = new int[hull_bone_count];
            int[] hull_bone_inv_bind_pose_id_copy = new int[hull_bone_count];
            int next_hull_bone = 0;
            int next_hull_bone_16 = 0;
            for (var hull_bone : batch.hull_bones)
            {
                hull_bones_copy[next_hull_bone_16] = hull_bone.bone_data()[0];
                hull_bones_copy[next_hull_bone_16 + 1] = hull_bone.bone_data()[1];
                hull_bones_copy[next_hull_bone_16 + 2] = hull_bone.bone_data()[2];
                hull_bones_copy[next_hull_bone_16 + 3] = hull_bone.bone_data()[3];
                hull_bones_copy[next_hull_bone_16 + 4] = hull_bone.bone_data()[4];
                hull_bones_copy[next_hull_bone_16 + 5] = hull_bone.bone_data()[5];
                hull_bones_copy[next_hull_bone_16 + 6] = hull_bone.bone_data()[6];
                hull_bones_copy[next_hull_bone_16 + 7] = hull_bone.bone_data()[7];
                hull_bones_copy[next_hull_bone_16 + 8] = hull_bone.bone_data()[8];
                hull_bones_copy[next_hull_bone_16 + 9] = hull_bone.bone_data()[9];
                hull_bones_copy[next_hull_bone_16 + 10] = hull_bone.bone_data()[10];
                hull_bones_copy[next_hull_bone_16 + 11] = hull_bone.bone_data()[11];
                hull_bones_copy[next_hull_bone_16 + 12] = hull_bone.bone_data()[12];
                hull_bones_copy[next_hull_bone_16 + 13] = hull_bone.bone_data()[13];
                hull_bones_copy[next_hull_bone_16 + 14] = hull_bone.bone_data()[14];
                hull_bones_copy[next_hull_bone_16 + 15] = hull_bone.bone_data()[15];
                hull_bone_bind_pose_id_copy[next_hull_bone] = hull_bone.bind_pose_id();
                hull_bone_inv_bind_pose_id_copy[next_hull_bone] = hull_bone.inv_bind_pose_id();
                next_hull_bone_16 += vec16_stride;
                next_hull_bone++;
            }
            GPGPU.cl_write_float_buffer(GPGPU.cl_cmd_queue_ptr, hull_bone_b.buffer_pointer, hull_bone_index, hull_bones_copy);
            GPGPU.cl_write_int_buffer(GPGPU.cl_cmd_queue_ptr, hull_bone_bind_pose_id_b.buffer_pointer, hull_bone_index, hull_bone_bind_pose_id_copy);
            GPGPU.cl_write_int_buffer(GPGPU.cl_cmd_queue_ptr, hull_bone_inv_bind_pose_id_b.buffer_pointer, hull_bone_index, hull_bone_inv_bind_pose_id_copy);
            hull_bone_index += hull_bone_count;
        }


        int armature_bone_count = batch.armature_bones.size();
        if (armature_bone_count > 0)
        {
            int armature_bone_capacity = armature_bone_index + armature_bone_count;
            armature_bone_buffer.ensure_capacity(armature_bone_capacity);
            armature_bone_reference_id_buffer.ensure_capacity(armature_bone_capacity);
            armature_bone_parent_id_buffer.ensure_capacity(armature_bone_capacity);
            float[] armature_bones_copy = new float[armature_bone_count * vec16_stride];
            int[] armature_bone_reference_id_copy = new int[armature_bone_count];
            int[] armature_bone_parent_id_copy = new int[armature_bone_count];
            int next_armature_bone = 0;
            int next_armature_bone_16 = 0;
            for (var armature_bone : batch.armature_bones)
            {
                armature_bones_copy[next_armature_bone_16] = armature_bone.bone_data()[0];
                armature_bones_copy[next_armature_bone_16 + 1] = armature_bone.bone_data()[1];
                armature_bones_copy[next_armature_bone_16 + 2] = armature_bone.bone_data()[2];
                armature_bones_copy[next_armature_bone_16 + 3] = armature_bone.bone_data()[3];
                armature_bones_copy[next_armature_bone_16 + 4] = armature_bone.bone_data()[4];
                armature_bones_copy[next_armature_bone_16 + 5] = armature_bone.bone_data()[5];
                armature_bones_copy[next_armature_bone_16 + 6] = armature_bone.bone_data()[6];
                armature_bones_copy[next_armature_bone_16 + 7] = armature_bone.bone_data()[7];
                armature_bones_copy[next_armature_bone_16 + 8] = armature_bone.bone_data()[8];
                armature_bones_copy[next_armature_bone_16 + 9] = armature_bone.bone_data()[9];
                armature_bones_copy[next_armature_bone_16 + 10] = armature_bone.bone_data()[10];
                armature_bones_copy[next_armature_bone_16 + 11] = armature_bone.bone_data()[11];
                armature_bones_copy[next_armature_bone_16 + 12] = armature_bone.bone_data()[12];
                armature_bones_copy[next_armature_bone_16 + 13] = armature_bone.bone_data()[13];
                armature_bones_copy[next_armature_bone_16 + 14] = armature_bone.bone_data()[14];
                armature_bones_copy[next_armature_bone_16 + 15] = armature_bone.bone_data()[15];
                armature_bone_reference_id_copy[next_armature_bone] = armature_bone.bone_reference();
                armature_bone_parent_id_copy[next_armature_bone] = armature_bone.bone_parent_id();
                next_armature_bone_16 += vec16_stride;
                next_armature_bone++;
            }

            GPGPU.cl_write_float_buffer(GPGPU.cl_cmd_queue_ptr, armature_bone_buffer.buffer_pointer, armature_bone_index, armature_bones_copy);
            GPGPU.cl_write_int_buffer(GPGPU.cl_cmd_queue_ptr, armature_bone_reference_id_buffer.buffer_pointer, armature_bone_index, armature_bone_reference_id_copy);
            GPGPU.cl_write_int_buffer(GPGPU.cl_cmd_queue_ptr, armature_bone_parent_id_buffer.buffer_pointer, armature_bone_index, armature_bone_parent_id_copy);
            armature_bone_index += armature_bone_count;
        }

    }



    public int new_animation_timings(float duration, float tick_rate)
    {
        int capacity = animation_index + 1;

        anim_duration_buffer.ensure_capacity(capacity);
        anim_tick_rate_buffer.ensure_capacity(capacity);

        create_animation_timings_k
            .set_arg(CreateAnimationTimings_k.Args.target, animation_index)
            .set_arg(CreateAnimationTimings_k.Args.new_animation_duration, duration)
            .set_arg(CreateAnimationTimings_k.Args.new_animation_tick_rate, tick_rate)
            .call(GPGPU.global_single_size);

        return animation_index++;
    }

    public int new_bone_channel(int anim_timing_index, int[] pos_table, int[] rot_table, int[] scl_table)
    {
        int capacity = bone_channel_index + 1;
        anim_timing_index_buffer.ensure_capacity(capacity);
        anim_bone_pos_channel_buffer.ensure_capacity(capacity);
        anim_bone_rot_channel_buffer.ensure_capacity(capacity);
        anim_bone_scl_channel_buffer.ensure_capacity(capacity);

        create_bone_channel_k
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
        anim_key_frame_buffer.ensure_capacity(capacity);
        anim_frame_time_buffer.ensure_capacity(capacity);

        create_keyframe_k
            .set_arg(CreateKeyFrame_k.Args.target, keyframe_index)
            .set_arg(CreateKeyFrame_k.Args.new_keyframe, frame)
            .set_arg(CreateKeyFrame_k.Args.new_frame_time, time)
            .call(GPGPU.global_single_size);

        return keyframe_index++;
    }

    public int new_texture_uv(float u, float v)
    {
        int capacity = uv_index + 1;
        vertex_texture_uv_buffer.ensure_capacity(capacity);

        create_texture_uv_k
            .set_arg(CreateTextureUV_k.Args.target, uv_index)
            .set_arg(CreateTextureUV_k.Args.new_texture_uv, arg_float2(u, v))
            .call(GPGPU.global_single_size);

        return uv_index++;
    }

    public int new_edge(int p1, int p2, float l, int flags)
    {
        int required_capacity = edge_index + 1;
        edge_buffer.ensure_capacity(required_capacity);
        edge_length_buffer.ensure_capacity(required_capacity);
        edge_flag_buffer.ensure_capacity(required_capacity);

        create_edge_k
            .set_arg(CreateEdge_k.Args.target, edge_index)
            .set_arg(CreateEdge_k.Args.new_edge, arg_int2(p1, p2))
            .set_arg(CreateEdge_k.Args.new_edge_length, l)
            .set_arg(CreateEdge_k.Args.new_edge_flag, flags)
            .call(GPGPU.global_single_size);

        return edge_index++;
    }

    public int new_point(float[] position, int[] bone_ids, int vertex_index, int hull_index, int flags)
    {
        int capacity = point_index + 1;
        point_buffer.ensure_capacity(capacity);
        point_anti_gravity_buffer.ensure_capacity(capacity);
        point_vertex_reference_buffer.ensure_capacity(capacity);
        point_hull_index_buffer.ensure_capacity(capacity);
        point_flag_buffer.ensure_capacity(capacity);
        point_hit_count_buffer.ensure_capacity(capacity);
        point_bone_table_buffer.ensure_capacity(capacity);

        var new_point = new float[]{position[0], position[1], position[0], position[1]};
        create_point_k
            .set_arg(CreatePoint_k.Args.target, point_index)
            .set_arg(CreatePoint_k.Args.new_point, new_point)
            .set_arg(CreatePoint_k.Args.new_point_vertex_reference, vertex_index)
            .set_arg(CreatePoint_k.Args.new_point_hull_index, hull_index)
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
        hull_b.ensure_capacity(capacity);
        hull_scale_b.ensure_capacity(capacity);
        hull_mesh_id_b.ensure_capacity(capacity);
        hull_uv_offset_b.ensure_capacity(capacity);
        hull_rotation_b.ensure_capacity(capacity);
        hull_integrity_b.ensure_capacity(capacity);
        hull_point_table_b.ensure_capacity(capacity);
        hull_edge_table_b.ensure_capacity(capacity);
        hull_flag_b.ensure_capacity(capacity);
        hull_bone_table_b.ensure_capacity(capacity);
        hull_entity_id_b.ensure_capacity(capacity);
        hull_friction_b.ensure_capacity(capacity);
        hull_restitution_b.ensure_capacity(capacity);
        hull_aabb_b.ensure_capacity(capacity);
        hull_aabb_index_b.ensure_capacity(capacity);
        hull_aabb_key_b.ensure_capacity(capacity);

        create_hull_k
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

        mesh_vertex_table_buffer.ensure_capacity(capacity);
        mesh_face_table_buffer.ensure_capacity(capacity);

        create_mesh_reference_k
            .set_arg(CreateMeshReference_k.Args.target, mesh_index)
            .set_arg(CreateMeshReference_k.Args.new_mesh_vertex_table, vertex_table)
            .set_arg(CreateMeshReference_k.Args.new_mesh_face_table, face_table)
            .call(GPGPU.global_single_size);

        return mesh_index++;
    }

    public int new_mesh_face(int[] face)
    {
        int capacity = face_index + 1;
        mesh_face_buffer.ensure_capacity(capacity);

        create_mesh_face_k
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
        entity_buffer.ensure_capacity(capacity);
        entity_flag_buffer.ensure_capacity(capacity);
        entity_root_hull_buffer.ensure_capacity(capacity);
        entity_model_id_buffer.ensure_capacity(capacity);
        entity_model_transform_buffer.ensure_capacity(capacity);
        entity_accel_buffer.ensure_capacity(capacity);
        entity_mass_buffer.ensure_capacity(capacity);
        entity_anim_index_buffer.ensure_capacity(capacity);
        entity_anim_elapsed_buffer.ensure_capacity(capacity);
        entity_anim_blend_buffer.ensure_capacity(capacity);
        entity_motion_state_buffer.ensure_capacity(capacity);
        entity_hull_table_buffer.ensure_capacity(capacity);
        entity_bone_table_buffer.ensure_capacity(capacity);

        create_entity_k
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
        vertex_reference_buffer.ensure_capacity(capacity);
        vertex_weight_buffer.ensure_capacity(capacity);
        vertex_uv_table_buffer.ensure_capacity(capacity);

        create_vertex_reference_k
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
        bone_bind_pose_buffer.ensure_capacity(capacity);
        bone_anim_channel_table_buffer.ensure_capacity(capacity); // note: filled in later

        create_bone_bind_pose_k
            .set_arg(CreateBoneBindPose_k.Args.target,bone_bind_index)
            .set_arg(CreateBoneBindPose_k.Args.new_bone_bind_pose, bone_data)
            .call(GPGPU.global_single_size);

        return bone_bind_index++;
    }

    public int new_bone_reference(float[] bone_data)
    {
        int capacity = bone_ref_index + 1;
        bone_reference_buffer.ensure_capacity(capacity);

        create_bone_reference_k
            .set_arg(CreateBoneRef_k.Args.target, bone_ref_index)
            .set_arg(CreateBoneRef_k.Args.new_bone_reference, bone_data)
            .call(GPGPU.global_single_size);

        return bone_ref_index++;
    }

    public int new_hull_bone(float[] bone_data, int bind_pose_id, int inv_bind_pose_id)
    {
        int capacity = hull_bone_index + 1;
        hull_bone_b.ensure_capacity(capacity);
        hull_bone_bind_pose_id_b.ensure_capacity(capacity);
        hull_bone_inv_bind_pose_id_b.ensure_capacity(capacity);

        create_bone_k
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
        armature_bone_buffer.ensure_capacity(capacity);
        armature_bone_reference_id_buffer.ensure_capacity(capacity);
        armature_bone_parent_id_buffer.ensure_capacity(capacity);

        create_armature_bone_k
            .set_arg(CreateArmatureBone_k.Args.target, armature_bone_index)
            .set_arg(CreateArmatureBone_k.Args.new_armature_bone, bone_data)
            .set_arg(CreateArmatureBone_k.Args.new_armature_bone_reference, bone_reference)
            .set_arg(CreateArmatureBone_k.Args.new_armature_bone_parent_id, bone_parent_id)
            .call(GPGPU.global_single_size);

        return armature_bone_index++;
    }

    public int new_model_transform(float[] transform_data)
    {
        int capacity = model_transform_index + 1;
        model_transform_buffer.ensure_capacity(capacity);

        create_model_transform_k
            .set_arg(CreateModelTransform_k.Args.target, model_transform_index)
            .set_arg(CreateModelTransform_k.Args.new_model_transform, transform_data)
            .call(GPGPU.global_single_size);

        return model_transform_index++;
    }

    public void set_bone_channel_table(int bind_pose_target, int[] channel_table)
    {
        set_bone_channel_table_k
            .set_arg(SetBoneChannelTable_k.Args.target, bind_pose_target)
            .set_arg(SetBoneChannelTable_k.Args.new_bone_channel_table, channel_table)
            .call(GPGPU.global_single_size);
    }

    public void update_accel(int entity_index, float acc_x, float acc_y)
    {
        update_accel_k
            .set_arg(UpdateAccel_k.Args.target, entity_index)
            .set_arg(UpdateAccel_k.Args.new_value, arg_float2(acc_x, acc_y))
            .call(GPGPU.global_single_size);
    }

    public void update_position(int entity_index, float x, float y)
    {
        update_mouse_position_k
            .set_arg(UpdateMousePosition_k.Args.target, entity_index)
            .set_arg(UpdateMousePosition_k.Args.new_value, arg_float2(x, y))
            .call(GPGPU.global_single_size);
    }

    public float[] read_position(int entity_index)
    {
        GPGPU.cl_zero_buffer(GPGPU.cl_cmd_queue_ptr, position_buffer_ptr, CLSize.cl_float2);

        read_position_k
            .ptr_arg(ReadPosition_k.Args.output, position_buffer_ptr)
            .set_arg(ReadPosition_k.Args.target, entity_index)
            .call(GPGPU.global_single_size);

        return GPGPU.cl_read_pinned_float_buffer(GPGPU.cl_cmd_queue_ptr, position_buffer_ptr, CLSize.cl_float2, 2);
    }

    public void delete_and_compact()
    {
        GPGPU.cl_zero_buffer(GPGPU.cl_cmd_queue_ptr, delete_counter_ptr, CLSize.cl_int);

        locate_out_of_bounds_k
            .ptr_arg(LocateOutOfBounds_k.Args.counter, delete_counter_ptr)
            .call(arg_long(entity_index));

        delete_buffer_1.ensure_capacity(entity_index);
        delete_buffer_2.ensure_capacity(entity_index);

        int[] shift_counts = scan_deletes(delete_buffer_1.pointer(), delete_buffer_2.pointer(), entity_index);

        if (shift_counts[4] == 0)
        {
            return;
        }

        hull_shift.ensure_capacity(hull_index);
        edge_shift.ensure_capacity(edge_index);
        point_shift.ensure_capacity(point_index);
        hull_bone_shift.ensure_capacity(hull_bone_index);
        armature_bone_shift.ensure_capacity(armature_bone_index);

        hull_shift.clear();
        edge_shift.clear();
        point_shift.clear();
        hull_bone_shift.clear();
        armature_bone_shift.clear();

        compact_entities_k
            .ptr_arg(CompactEntities_k.Args.buffer_in_1, delete_buffer_1.pointer())
            .ptr_arg(CompactEntities_k.Args.buffer_in_2, delete_buffer_2.pointer());

        linearize_kernel(compact_entities_k, entity_index);
        linearize_kernel(compact_hull_bones_k, hull_bone_index);
        linearize_kernel(compact_points_k, point_index);
        linearize_kernel(compact_edges_k, edge_index);
        linearize_kernel(compact_hulls_k, hull_index);
        linearize_kernel(compact_armature_bones_k, armature_bone_index);

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

        GPGPU.cl_zero_buffer(GPGPU.cl_cmd_queue_ptr, delete_sizes_ptr, CLSize.cl_int * 6);

        scan_deletes_single_block_out_k
            .ptr_arg(ScanDeletesSingleBlockOut_k.Args.output, o1_data_ptr)
            .ptr_arg(ScanDeletesSingleBlockOut_k.Args.output2, o2_data_ptr)
            .loc_arg(ScanDeletesSingleBlockOut_k.Args.buffer, local_buffer_size)
            .loc_arg(ScanDeletesSingleBlockOut_k.Args.buffer2, local_buffer_size2)
            .set_arg(ScanDeletesSingleBlockOut_k.Args.n, n)
            .call(GPGPU.local_work_default, GPGPU.local_work_default);

        return GPGPU.cl_read_pinned_int_buffer(GPGPU.cl_cmd_queue_ptr, delete_sizes_ptr, CLSize.cl_int * 6, 6);
    }

    private int[] scan_multi_block_deletes_out(long o1_data_ptr, long o2_data_ptr, int n, int k)
    {
        long local_buffer_size = CLSize.cl_int2 * GPGPU.max_scan_block_size;
        long local_buffer_size2 = CLSize.cl_int4 * GPGPU.max_scan_block_size;

        long gx = k * GPGPU.max_scan_block_size;
        long[] global_work_size = arg_long(gx);
        int part_size = k * 2;

        delete_partial_buffer_1.ensure_capacity(part_size);
        delete_partial_buffer_2.ensure_capacity(part_size);

        scan_deletes_multi_block_out_k
            .ptr_arg(ScanDeletesMultiBlockOut_k.Args.output1, o1_data_ptr)
            .ptr_arg(ScanDeletesMultiBlockOut_k.Args.output2, o2_data_ptr)
            .loc_arg(ScanDeletesMultiBlockOut_k.Args.buffer1, local_buffer_size)
            .loc_arg(ScanDeletesMultiBlockOut_k.Args.buffer2, local_buffer_size2)
            .set_arg(ScanDeletesMultiBlockOut_k.Args.n, n)
            .call(global_work_size, GPGPU.local_work_default);

        // note the partial buffers are scanned and updated in-place
        GPGPU.scan_int2(delete_partial_buffer_1.pointer(), part_size);
        GPGPU.scan_int4(delete_partial_buffer_2.pointer(), part_size);

        GPGPU.cl_zero_buffer(GPGPU.cl_cmd_queue_ptr, delete_sizes_ptr, CLSize.cl_int * 6);

        complete_deletes_multi_block_out_k
            .ptr_arg(CompleteDeletesMultiBlockOut_k.Args.output1, o1_data_ptr)
            .ptr_arg(CompleteDeletesMultiBlockOut_k.Args.output2, o2_data_ptr)
            .loc_arg(CompleteDeletesMultiBlockOut_k.Args.buffer1, local_buffer_size)
            .loc_arg(CompleteDeletesMultiBlockOut_k.Args.buffer2, local_buffer_size2)
            .set_arg(CompleteDeletesMultiBlockOut_k.Args.n, n)
            .call(global_work_size, GPGPU.local_work_default);

        return GPGPU.cl_read_pinned_int_buffer(GPGPU.cl_cmd_queue_ptr, delete_sizes_ptr, CLSize.cl_int * 6, 6);
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

    public void destroy()
    {
//        System.out.println("--- shutting down --- ");
//
//        System.out.println("hulls      : " + hull_index);
//        System.out.println("points     : " + point_index);
//        System.out.println("edges      : " + edge_index);
//        System.out.println("bones      : " + bone_index);
//        System.out.println("armatures  : " + armature_index);

        gpu_crud.destroy();
        scan_deletes.destroy();
        hull_shift.release();
        edge_shift.release();
        point_shift.release();
        hull_bone_shift.release();
        armature_bone_shift.release();
        delete_buffer_1.release();
        delete_buffer_2.release();
        delete_partial_buffer_1.release();
        delete_partial_buffer_2.release();

        edge_buffer.release();
        edge_length_buffer.release();
        edge_flag_buffer.release();
        hull_b.release();
        hull_scale_b.release();
        hull_mesh_id_b.release();
        hull_uv_offset_b.release();
        hull_rotation_b.release();
        hull_integrity_b.release();
        hull_point_table_b.release();
        hull_edge_table_b.release();
        hull_flag_b.release();
        hull_bone_table_b.release();
        hull_entity_id_b.release();
        hull_aabb_b.release();
        hull_aabb_index_b.release();
        hull_aabb_key_b.release();
        hull_bone_b.release();
        hull_bone_bind_pose_id_b.release();
        hull_bone_inv_bind_pose_id_b.release();
        hull_friction_b.release();
        hull_restitution_b.release();
        point_buffer.release();
        point_anti_gravity_buffer.release();
        point_vertex_reference_buffer.release();
        point_hull_index_buffer.release();
        point_flag_buffer.release();

        point_hit_count_buffer.release();

        point_bone_table_buffer.release();
        vertex_reference_buffer.release();
        vertex_weight_buffer.release();
        vertex_texture_uv_buffer.release();
        vertex_uv_table_buffer.release();
        model_transform_buffer.release();
        bone_reference_buffer.release();
        bone_bind_pose_buffer.release();
        bone_anim_channel_table_buffer.release();
        mesh_vertex_table_buffer.release();
        mesh_face_table_buffer.release();
        mesh_face_buffer.release();
        anim_key_frame_buffer.release();
        anim_frame_time_buffer.release();
        anim_bone_pos_channel_buffer.release();
        anim_bone_rot_channel_buffer.release();
        anim_bone_scl_channel_buffer.release();
        anim_duration_buffer.release();
        anim_tick_rate_buffer.release();
        anim_timing_index_buffer.release();
        armature_bone_buffer.release();
        armature_bone_reference_id_buffer.release();
        armature_bone_parent_id_buffer.release();
        entity_buffer.release();
        entity_flag_buffer.release();
        entity_root_hull_buffer.release();
        entity_model_id_buffer.release();
        entity_model_transform_buffer.release();
        entity_accel_buffer.release();
        entity_mass_buffer.release();
        entity_anim_index_buffer.release();
        entity_anim_elapsed_buffer.release();
        entity_anim_blend_buffer.release();
        entity_motion_state_buffer.release();
        entity_hull_table_buffer.release();
        entity_bone_table_buffer.release();

        debug();

        GPGPU.cl_release_buffer(delete_counter_ptr);
        GPGPU.cl_release_buffer(position_buffer_ptr);
        GPGPU.cl_release_buffer(delete_sizes_ptr);
    }

    private void debug()
    {
        long total = 0;
        total += hull_shift.debug_data();
        total += edge_shift.debug_data();
        total += point_shift.debug_data();
        total += hull_bone_shift.debug_data();
        total += armature_bone_shift.debug_data();
        total += delete_buffer_1.debug_data();
        total += delete_buffer_2.debug_data();
        total += delete_partial_buffer_1.debug_data();
        total += delete_partial_buffer_2.debug_data();
        total += edge_buffer.debug_data();
        total += edge_length_buffer.debug_data();
        total += edge_flag_buffer.debug_data();
        total += hull_b.debug_data();
        total += hull_scale_b.debug_data();
        total += hull_mesh_id_b.debug_data();
        total += hull_uv_offset_b.debug_data();
        total += hull_rotation_b.debug_data();
        total += hull_integrity_b.debug_data();
        total += hull_point_table_b.debug_data();
        total += hull_edge_table_b.debug_data();
        total += hull_flag_b.debug_data();
        total += hull_bone_table_b.debug_data();
        total += hull_entity_id_b.debug_data();
        total += hull_aabb_b.debug_data();
        total += hull_aabb_index_b.debug_data();
        total += hull_aabb_key_b.debug_data();
        total += hull_bone_b.debug_data();
        total += hull_bone_bind_pose_id_b.debug_data();
        total += hull_bone_inv_bind_pose_id_b.debug_data();
        total += hull_friction_b.debug_data();
        total += hull_restitution_b.debug_data();
        total += point_buffer.debug_data();
        total += point_anti_gravity_buffer.debug_data();
        total += point_vertex_reference_buffer.debug_data();
        total += point_hull_index_buffer.debug_data();
        total += point_flag_buffer.debug_data();

        total += point_hit_count_buffer.debug_data();

        total += point_bone_table_buffer.debug_data();
        total += vertex_reference_buffer.debug_data();
        total += vertex_weight_buffer.debug_data();
        total += vertex_texture_uv_buffer.debug_data();
        total += vertex_uv_table_buffer.debug_data();
        total += model_transform_buffer.debug_data();
        total += bone_reference_buffer.debug_data();
        total += bone_bind_pose_buffer.debug_data();
        total += bone_anim_channel_table_buffer.debug_data();
        total += mesh_vertex_table_buffer.debug_data();
        total += mesh_face_table_buffer.debug_data();
        total += mesh_face_buffer.debug_data();
        total += anim_key_frame_buffer.debug_data();
        total += anim_frame_time_buffer.debug_data();
        total += anim_bone_pos_channel_buffer.debug_data();
        total += anim_bone_rot_channel_buffer.debug_data();
        total += anim_bone_scl_channel_buffer.debug_data();
        total += anim_duration_buffer.debug_data();
        total += anim_tick_rate_buffer.debug_data();
        total += anim_timing_index_buffer.debug_data();
        total += armature_bone_buffer.debug_data();
        total += armature_bone_reference_id_buffer.debug_data();
        total += armature_bone_parent_id_buffer.debug_data();
        total += entity_buffer.debug_data();
        total += entity_flag_buffer.debug_data();
        total += entity_root_hull_buffer.debug_data();
        total += entity_model_id_buffer.debug_data();
        total += entity_model_transform_buffer.debug_data();
        total += entity_accel_buffer.debug_data();
        total += entity_mass_buffer.debug_data();
        total += entity_anim_index_buffer.debug_data();
        total += entity_anim_elapsed_buffer.debug_data();
        total += entity_anim_blend_buffer.debug_data();
        total += entity_motion_state_buffer.debug_data();
        total += entity_hull_table_buffer.debug_data();
        total += entity_bone_table_buffer.debug_data();

        //System.out.println("---------------------------");
        System.out.println("Core Memory Usage: MB " + ((float) total / 1024f / 1024f));
    }
}
