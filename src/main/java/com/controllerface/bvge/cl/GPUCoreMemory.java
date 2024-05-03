package com.controllerface.bvge.cl;

import com.controllerface.bvge.cl.kernels.*;
import com.controllerface.bvge.cl.programs.GPUCrud;
import com.controllerface.bvge.cl.programs.ScanDeletes;

import static com.controllerface.bvge.cl.CLUtils.*;

public class GPUCoreMemory
{
    private final GPUProgram gpu_crud = new GPUCrud();
    private final GPUProgram scan_deletes = new ScanDeletes();

    private final GPUKernel compact_armature_bones_k;
    private final GPUKernel compact_armatures_k;
    private final GPUKernel compact_bones_k;
    private final GPUKernel compact_edges_k;
    private final GPUKernel compact_hulls_k;
    private final GPUKernel compact_points_k;
    private final GPUKernel complete_deletes_multi_block_out_k;
    private final GPUKernel create_animation_timings_k;
    private final GPUKernel create_armature_bone_k;
    private final GPUKernel create_armature_k;
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

    // internal buffers
    /**
     * During the armature compaction process, these buffers are written to, and store the number of
     * positions that the corresponding values must shift left within their own buffers when the
     * buffer compaction occurs. Each index is aligned with the corresponding data type
     * that will be shifted. I.e. every bone in the bone buffer has a corresponding entry in the
     * bone shift buffer. Points, edges, and hulls work the same way.
     */
    private final ResizableBuffer bone_bind_shift;
    private final ResizableBuffer bone_shift;
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

    // external buffers

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

    /** float2
     * x: current x acceleration
     * y: current y acceleration
     */
    private final ResizableBuffer armature_accel_buffer;

    /** float
     * x: the last rendered timestamp
     */
    private final ResizableBuffer armature_anim_elapsed_buffer;

    /** short2
     * x: number of ticks moving downward
     * y: number of ticks moving upward
     */
    private final ResizableBuffer armature_anim_state_buffer;

    /** int
     * x: the currently running animation index
     */
    private final ResizableBuffer armature_anim_index_buffer;

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

    /** float4
     * x: current x position
     * y: current y position
     * z: previous x position
     * w: previous y position
     */
    private final ResizableBuffer armature_buffer;

    /** int
     * x: armature flags (bit-field)
     */
    private final ResizableBuffer armature_flag_buffer;

    /** int
     * x: root hull index of the aligned armature
     */
    private final ResizableBuffer armature_root_hull_buffer;

    /** int
     * x: model id of the aligned armature
     */
    private final ResizableBuffer armature_model_id_buffer;

    /** int
     * x: model transform index of the aligned armature
     */
    private final ResizableBuffer armature_model_transform_buffer;

    /** int2
     * x: start hull index
     * y: end hull index
     */
    private final ResizableBuffer armature_hull_table_buffer;

    /** int2
     * x: start bone anim index
     * y: end bone anim index
     */
    private final ResizableBuffer armature_bone_table_buffer;

    /** float
     * x: mass of the armature
     */
    private final ResizableBuffer armature_mass_buffer;

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

    /** float4
     * x: corner x position
     * y: corner y position
     * z: width
     * w: height
     */
    private final ResizableBuffer hull_aabb_buffer;

    /** int4
     * x: minimum x key index
     * y: maximum x key index
     * z: minimum y key index
     * w: maximum y key index
     */
    private final ResizableBuffer hull_aabb_index_buffer;

    /** int2
     * x: key bank offset
     * y: key bank size
     */
    private final ResizableBuffer hull_aabb_key_buffer;

    /** float16
     * s0-sF: Column-major, 4x4 transformation matrix, hull bone instance
     */
    private final ResizableBuffer hull_bone_buffer;

    /** int
     * x: bone bind pose index (model space)
     */
    private final ResizableBuffer hull_bind_pose_id_buffer;

    /** int
     * x: bone inverse bind pose index (mesh-space)
     */
    private final ResizableBuffer hull_inv_bind_pose_id_buffer;

    /** float2
     * x: current x position
     * y: current y position
     */
    private final ResizableBuffer hull_buffer;

    /** float2
     * x: scale x
     * y: scale y
     */
    private final ResizableBuffer hull_scale_buffer;

    /** int2
     * x: start point index
     * y: end point index
     */
    private final ResizableBuffer hull_point_table_buffer;

    /** int2
     * x: start edge index
     * y: end edge index
     */
    private final ResizableBuffer hull_edge_table_buffer;

    /** int
     * x: hull flags (bit-field)
     */
    private final ResizableBuffer hull_flag_buffer;

    /** int
     * x: armature id for aligned hull
     */
    private final ResizableBuffer hull_armature_id_buffer;

    /** int2
     * x: start bone
     * y: end bone
     */
    private final ResizableBuffer hull_bone_table_buffer;

    /** float
     * x: friction coefficient
     */
    private final ResizableBuffer hull_friction_buffer;

    /** float
     * x: restitution coefficient
     */
    private final ResizableBuffer hull_restitution_buffer;

    /** int
     * x: reference mesh id
     */
    private final ResizableBuffer hull_mesh_id_buffer;

    /** float2
     * x: initial reference angle
     * y: current rotation
     */
    private final ResizableBuffer hull_rotation_buffer;

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





    private final ResizableBuffer mirror_armature_buffer;
    private final ResizableBuffer mirror_armature_model_id_buffer;
    private final ResizableBuffer mirror_armature_root_hull_buffer;
    private final ResizableBuffer mirror_edge_buffer;
    private final ResizableBuffer mirror_edge_flag_buffer;
    private final ResizableBuffer mirror_hull_buffer;
    private final ResizableBuffer mirror_hull_aabb_buffer;
    private final ResizableBuffer mirror_hull_flag_buffer;
    private final ResizableBuffer mirror_hull_mesh_id_buffer;
    private final ResizableBuffer mirror_hull_point_table_buffer;
    private final ResizableBuffer mirror_hull_rotation_buffer;
    private final ResizableBuffer mirror_hull_scale_buffer;
    private final ResizableBuffer mirror_point_buffer;
    private final ResizableBuffer mirror_point_anti_gravity_buffer;
    private final ResizableBuffer mirror_point_hit_count_buffer;
    private final ResizableBuffer mirror_point_vertex_reference_buffer;







    private final long delete_counter_ptr;
    private final long position_buffer_ptr;
    private final long delete_sizes_ptr;

    private int hull_index            = 0;
    private int point_index           = 0;
    private int edge_index            = 0;
    private int vertex_ref_index      = 0;
    private int bone_bind_index       = 0;
    private int bone_ref_index        = 0;
    private int bone_index            = 0;
    private int model_transform_index = 0;
    private int armature_bone_index   = 0;
    private int armature_index        = 0;
    private int mesh_index            = 0;
    private int face_index            = 0;
    private int uv_index              = 0;
    private int keyframe_index        = 0;
    private int bone_channel_index    = 0;
    private int animation_index       = 0;

    private int last_hull_index       = 0;
    private int last_point_index      = 0;
    private int last_edge_index       = 0;
    private int last_armature_index   = 0;

    public GPUCoreMemory()
    {
        delete_counter_ptr      = GPGPU.cl_new_int_arg_buffer(new int[]{ 0 });
        position_buffer_ptr     = GPGPU.cl_new_pinned_buffer(CLSize.cl_float2);
        delete_sizes_ptr        = GPGPU.cl_new_pinned_buffer(CLSize.cl_int * 6);

        hull_shift                      = new TransientBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_int, 10_000L);
        edge_shift                      = new TransientBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_int, 24_000L);
        point_shift                     = new TransientBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_int, 50_000L);
        bone_shift                      = new TransientBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_int, 10_000L);
        bone_bind_shift                 = new TransientBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_int, 24_000L);
        delete_buffer_1                 = new TransientBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_int2, 10_000L);
        delete_buffer_2                 = new TransientBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_int4, 20_000L);
        delete_partial_buffer_1         = new TransientBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_int2, 10_000L);
        delete_partial_buffer_2         = new TransientBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_int4, 20_000L);

        anim_bone_pos_channel_buffer    = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_int2);
        anim_bone_rot_channel_buffer    = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_int2);
        anim_bone_scl_channel_buffer    = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_int2);
        anim_frame_time_buffer          = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_float);
        anim_key_frame_buffer           = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_float4);
        anim_duration_buffer            = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_float);
        anim_tick_rate_buffer           = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_float);
        anim_timing_index_buffer        = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_int);
        armature_accel_buffer           = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_float2, 10_000L);
        armature_anim_elapsed_buffer    = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_float, 10_000L);
        armature_anim_state_buffer      = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_short2, 10_000L);
        armature_anim_index_buffer      = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_int, 10_000L);
        armature_bone_buffer            = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_float16);
        armature_bone_reference_id_buffer = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_int);
        armature_bone_parent_id_buffer  = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_int);
        armature_buffer                 = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_float4, 10_000L);
        armature_flag_buffer            = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_int, 10_000L);
        armature_root_hull_buffer       = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_int, 10_000L);
        armature_model_id_buffer        = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_int, 10_000L);
        armature_model_transform_buffer = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_int, 10_000L);

        armature_hull_table_buffer      = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_int2, 10_000L);
        armature_bone_table_buffer      = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_int2, 10_000L);

        armature_mass_buffer            = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_float, 10_000L);
        bone_anim_channel_table_buffer  = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_int2);
        bone_bind_pose_buffer           = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_float16);
        bone_reference_buffer           = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_float16);
        edge_buffer                     = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_int2, 24_000L);
        edge_flag_buffer                = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_int, 24_000L);
        edge_length_buffer              = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_float, 24_000L);
        hull_aabb_buffer                = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_float4, 10_000L);
        hull_aabb_index_buffer          = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_int4, 10_000L);
        hull_aabb_key_buffer            = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_int2, 10_000L);
        hull_bone_buffer                = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_float16, 10_000L);
        hull_bind_pose_id_buffer        = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_int, 10_000L);
        hull_inv_bind_pose_id_buffer    = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_int, 10_000L);
        hull_buffer                     = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_float2, 10_000L);
        hull_scale_buffer               = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_float2, 10_000L);
        hull_point_table_buffer         = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_int2, 10_000L);
        hull_edge_table_buffer          = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_int2, 10_000L);
        hull_flag_buffer                = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_int, 10_000L);
        hull_bone_table_buffer          = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_int2, 10_000L);
        hull_armature_id_buffer         = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_int, 10_000L);
        hull_friction_buffer            = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_float, 10_000L);
        hull_restitution_buffer         = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_float, 10_000L);
        hull_mesh_id_buffer             = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_int, 10_000L);
        hull_rotation_buffer            = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_float2, 10_000L);
        mesh_face_buffer                = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_int4);
        mesh_vertex_table_buffer        = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_int2);
        mesh_face_table_buffer          = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_int2);
        model_transform_buffer          = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_float16);
        point_anti_gravity_buffer       = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_float, 50_000L);
        point_bone_table_buffer         = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_int4, 50_000L);
        point_buffer                    = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_float4, 50_000L);
        point_vertex_reference_buffer   = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_int, 50_000L);
        point_hull_index_buffer         = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_int, 50_000L);
        point_flag_buffer               = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_int, 50_000L);

        point_hit_count_buffer          = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_ushort, 50_000L);

        vertex_reference_buffer         = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_float2);
        vertex_texture_uv_buffer        = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_float2);
        vertex_uv_table_buffer          = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_int2);
        vertex_weight_buffer            = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_float4);


        // mirrors:

        mirror_armature_buffer                 = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_float4, 10_000L);
        mirror_armature_model_id_buffer        = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_int, 10_000L);
        mirror_armature_root_hull_buffer       = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_int, 10_000L);
        mirror_edge_buffer                     = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_int2, 24_000L);
        mirror_edge_flag_buffer                = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_int, 24_000L);
        mirror_hull_buffer                     = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_float2, 10_000L);
        mirror_hull_aabb_buffer                = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_float4, 10_000L);
        mirror_hull_flag_buffer                = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_int, 10_000L);
        mirror_hull_mesh_id_buffer             = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_int, 10_000L);
        mirror_hull_point_table_buffer         = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_int2, 10_000L);
        mirror_hull_rotation_buffer            = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_float2, 10_000L);
        mirror_hull_scale_buffer               = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_float2, 10_000L);
        mirror_point_hit_count_buffer          = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_ushort, 50_000L);
        mirror_point_anti_gravity_buffer       = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_float, 50_000L);
        mirror_point_buffer                    = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_float4, 50_000L);
        mirror_point_vertex_reference_buffer   = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_int, 50_000L);

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

        long create_armature_k_ptr = gpu_crud.kernel_ptr(Kernel.create_armature);
        create_armature_k = new CreateArmature_k(GPGPU.cl_cmd_queue_ptr, create_armature_k_ptr)
            .buf_arg(CreateArmature_k.Args.armatures, armature_buffer)
            .buf_arg(CreateArmature_k.Args.armature_root_hulls, armature_root_hull_buffer)
            .buf_arg(CreateArmature_k.Args.armature_model_indices, armature_model_id_buffer)
            .buf_arg(CreateArmature_k.Args.armature_model_transforms, armature_model_transform_buffer)
            .buf_arg(CreateArmature_k.Args.armature_flags, armature_flag_buffer)
            .buf_arg(CreateArmature_k.Args.armature_hull_tables, armature_hull_table_buffer)
            .buf_arg(CreateArmature_k.Args.armature_bone_tables, armature_bone_table_buffer)
            .buf_arg(CreateArmature_k.Args.armature_masses, armature_mass_buffer)
            .buf_arg(CreateArmature_k.Args.armature_animation_indices, armature_anim_index_buffer)
            .buf_arg(CreateArmature_k.Args.armature_animation_elapsed, armature_anim_elapsed_buffer)
            .buf_arg(CreateArmature_k.Args.armature_animation_states, armature_anim_state_buffer);

        long create_bone_k_ptr = gpu_crud.kernel_ptr(Kernel.create_hull_bone);
        create_bone_k = new CreateHullBone_k(GPGPU.cl_cmd_queue_ptr, create_bone_k_ptr)
            .buf_arg(CreateHullBone_k.Args.bones, hull_bone_buffer)
            .buf_arg(CreateHullBone_k.Args.hull_bind_pose_indicies, hull_bind_pose_id_buffer)
            .buf_arg(CreateHullBone_k.Args.hull_inv_bind_pose_indicies, hull_inv_bind_pose_id_buffer);

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
            .buf_arg(CreateHull_k.Args.hulls, hull_buffer)
            .buf_arg(CreateHull_k.Args.hull_scales, hull_scale_buffer)
            .buf_arg(CreateHull_k.Args.hull_rotations, hull_rotation_buffer)
            .buf_arg(CreateHull_k.Args.hull_frictions, hull_friction_buffer)
            .buf_arg(CreateHull_k.Args.hull_restitutions, hull_restitution_buffer)
            .buf_arg(CreateHull_k.Args.hull_point_tables, hull_point_table_buffer)
            .buf_arg(CreateHull_k.Args.hull_edge_tables, hull_edge_table_buffer)
            .buf_arg(CreateHull_k.Args.hull_bone_tables, hull_bone_table_buffer)
            .buf_arg(CreateHull_k.Args.hull_armature_ids, hull_armature_id_buffer)
            .buf_arg(CreateHull_k.Args.hull_flags, hull_flag_buffer)
            .buf_arg(CreateHull_k.Args.hull_mesh_ids, hull_mesh_id_buffer);

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
            .buf_arg(ReadPosition_k.Args.armatures, armature_buffer);

        // update methods

        long update_accel_k_ptr = gpu_crud.kernel_ptr(Kernel.update_accel);
        update_accel_k = new UpdateAccel_k(GPGPU.cl_cmd_queue_ptr, update_accel_k_ptr)
            .buf_arg(UpdateAccel_k.Args.armature_accel, armature_accel_buffer);

        long set_bone_channel_table_k_ptr = gpu_crud.kernel_ptr(Kernel.set_bone_channel_table);
        set_bone_channel_table_k = new SetBoneChannelTable_k(GPGPU.cl_cmd_queue_ptr, set_bone_channel_table_k_ptr)
            .buf_arg(SetBoneChannelTable_k.Args.bone_channel_tables, bone_anim_channel_table_buffer);

        // delete methods

        long locate_out_of_bounds_k_ptr = scan_deletes.kernel_ptr(Kernel.locate_out_of_bounds);
        locate_out_of_bounds_k = new LocateOutOfBounds_k(GPGPU.cl_cmd_queue_ptr, locate_out_of_bounds_k_ptr)
            .buf_arg(LocateOutOfBounds_k.Args.hull_tables, armature_hull_table_buffer)
            .buf_arg(LocateOutOfBounds_k.Args.hull_flags, hull_flag_buffer)
            .buf_arg(LocateOutOfBounds_k.Args.armature_flags, armature_flag_buffer);

        long scan_deletes_single_block_out_k_ptr = scan_deletes.kernel_ptr(Kernel.scan_deletes_single_block_out);
        scan_deletes_single_block_out_k = new ScanDeletesSingleBlockOut_k(GPGPU.cl_cmd_queue_ptr, scan_deletes_single_block_out_k_ptr)
            .ptr_arg(ScanDeletesSingleBlockOut_k.Args.sz, delete_sizes_ptr)
            .buf_arg(ScanDeletesSingleBlockOut_k.Args.armature_flags, armature_flag_buffer)
            .buf_arg(ScanDeletesSingleBlockOut_k.Args.hull_tables, armature_hull_table_buffer)
            .buf_arg(ScanDeletesSingleBlockOut_k.Args.bone_tables, armature_bone_table_buffer)
            .buf_arg(ScanDeletesSingleBlockOut_k.Args.point_tables, hull_point_table_buffer)
            .buf_arg(ScanDeletesSingleBlockOut_k.Args.edge_tables, hull_edge_table_buffer)
            .buf_arg(ScanDeletesSingleBlockOut_k.Args.hull_bone_tables, hull_bone_table_buffer);

        long scan_deletes_multi_block_out_k_ptr = scan_deletes.kernel_ptr(Kernel.scan_deletes_multi_block_out);
        scan_deletes_multi_block_out_k = new ScanDeletesMultiBlockOut_k(GPGPU.cl_cmd_queue_ptr, scan_deletes_multi_block_out_k_ptr)
            .buf_arg(ScanDeletesMultiBlockOut_k.Args.part1, delete_partial_buffer_1)
            .buf_arg(ScanDeletesMultiBlockOut_k.Args.part2, delete_partial_buffer_2)
            .buf_arg(ScanDeletesMultiBlockOut_k.Args.armature_flags, armature_flag_buffer)
            .buf_arg(ScanDeletesMultiBlockOut_k.Args.hull_tables, armature_hull_table_buffer)
            .buf_arg(ScanDeletesMultiBlockOut_k.Args.bone_tables, armature_bone_table_buffer)
            .buf_arg(ScanDeletesMultiBlockOut_k.Args.point_tables, hull_point_table_buffer)
            .buf_arg(ScanDeletesMultiBlockOut_k.Args.edge_tables, hull_edge_table_buffer)
            .buf_arg(ScanDeletesMultiBlockOut_k.Args.hull_bone_tables, hull_bone_table_buffer);

        long complete_deletes_multi_block_out_k_ptr = scan_deletes.kernel_ptr(Kernel.complete_deletes_multi_block_out);
        complete_deletes_multi_block_out_k = new CompleteDeletesMultiBlockOut_k(GPGPU.cl_cmd_queue_ptr, complete_deletes_multi_block_out_k_ptr)
            .ptr_arg(CompleteDeletesMultiBlockOut_k.Args.sz, delete_sizes_ptr)
            .buf_arg(CompleteDeletesMultiBlockOut_k.Args.part1, delete_partial_buffer_1)
            .buf_arg(CompleteDeletesMultiBlockOut_k.Args.part2, delete_partial_buffer_2)
            .buf_arg(CompleteDeletesMultiBlockOut_k.Args.armature_flags, armature_flag_buffer)
            .buf_arg(CompleteDeletesMultiBlockOut_k.Args.hull_tables, armature_hull_table_buffer)
            .buf_arg(CompleteDeletesMultiBlockOut_k.Args.bone_tables, armature_bone_table_buffer)
            .buf_arg(CompleteDeletesMultiBlockOut_k.Args.point_tables, hull_point_table_buffer)
            .buf_arg(CompleteDeletesMultiBlockOut_k.Args.edge_tables, hull_edge_table_buffer)
            .buf_arg(CompleteDeletesMultiBlockOut_k.Args.hull_bone_tables, hull_bone_table_buffer);

        long compact_armatures_k_ptr = scan_deletes.kernel_ptr(Kernel.compact_armatures);
        compact_armatures_k = new CompactArmatures_k(GPGPU.cl_cmd_queue_ptr, compact_armatures_k_ptr)
            .buf_arg(CompactArmatures_k.Args.armatures, armature_buffer)
            .buf_arg(CompactArmatures_k.Args.armature_masses, armature_mass_buffer)
            .buf_arg(CompactArmatures_k.Args.armature_root_hulls, armature_root_hull_buffer)
            .buf_arg(CompactArmatures_k.Args.armature_model_indices, armature_model_id_buffer)
            .buf_arg(CompactArmatures_k.Args.armature_model_transforms, armature_model_transform_buffer)
            .buf_arg(CompactArmatures_k.Args.armature_flags, armature_flag_buffer)
            .buf_arg(CompactArmatures_k.Args.armature_animation_indices, armature_anim_index_buffer)
            .buf_arg(CompactArmatures_k.Args.armature_animation_elapsed, armature_anim_elapsed_buffer)
            .buf_arg(CompactArmatures_k.Args.armature_animation_states, armature_anim_state_buffer)
            .buf_arg(CompactArmatures_k.Args.armature_hull_tables, armature_hull_table_buffer)
            .buf_arg(CompactArmatures_k.Args.armature_bone_tables, armature_bone_table_buffer)
            .buf_arg(CompactArmatures_k.Args.hull_bone_tables, hull_bone_table_buffer)
            .buf_arg(CompactArmatures_k.Args.hull_armature_ids, hull_armature_id_buffer)
            .buf_arg(CompactArmatures_k.Args.hull_point_tables, hull_point_table_buffer)
            .buf_arg(CompactArmatures_k.Args.hull_edge_tables, hull_edge_table_buffer)
            .buf_arg(CompactArmatures_k.Args.points, point_buffer)
            .buf_arg(CompactArmatures_k.Args.point_hull_indices, point_hull_index_buffer)
            .buf_arg(CompactArmatures_k.Args.point_bone_tables, point_bone_table_buffer)
            .buf_arg(CompactArmatures_k.Args.armature_bone_parent_ids, armature_bone_parent_id_buffer)
            .buf_arg(CompactArmatures_k.Args.hull_bind_pose_indices, hull_bind_pose_id_buffer)
            .buf_arg(CompactArmatures_k.Args.edges, edge_buffer)
            .buf_arg(CompactArmatures_k.Args.bone_shift, bone_shift)
            .buf_arg(CompactArmatures_k.Args.point_shift, point_shift)
            .buf_arg(CompactArmatures_k.Args.edge_shift, edge_shift)
            .buf_arg(CompactArmatures_k.Args.hull_shift, hull_shift)
            .buf_arg(CompactArmatures_k.Args.bone_bind_shift, bone_bind_shift);

        long compact_hulls_k_ptr = scan_deletes.kernel_ptr(Kernel.compact_hulls);
        compact_hulls_k = new CompactHulls_k(GPGPU.cl_cmd_queue_ptr, compact_hulls_k_ptr)
            .buf_arg(CompactHulls_k.Args.hull_shift, hull_shift)
            .buf_arg(CompactHulls_k.Args.hulls, hull_buffer)
            .buf_arg(CompactHulls_k.Args.hull_scales, hull_scale_buffer)
            .buf_arg(CompactHulls_k.Args.hull_mesh_ids, hull_mesh_id_buffer)
            .buf_arg(CompactHulls_k.Args.hull_rotations, hull_rotation_buffer)
            .buf_arg(CompactHulls_k.Args.hull_frictions, hull_friction_buffer)
            .buf_arg(CompactHulls_k.Args.hull_restitutions, hull_restitution_buffer)
            .buf_arg(CompactHulls_k.Args.bone_tables, hull_bone_table_buffer)
            .buf_arg(CompactHulls_k.Args.armature_ids, hull_armature_id_buffer)
            .buf_arg(CompactHulls_k.Args.hull_flags, hull_flag_buffer)
            .buf_arg(CompactHulls_k.Args.hull_point_tables, hull_point_table_buffer)
            .buf_arg(CompactHulls_k.Args.hull_edge_tables, hull_edge_table_buffer)
            .buf_arg(CompactHulls_k.Args.bounds, hull_aabb_buffer)
            .buf_arg(CompactHulls_k.Args.bounds_index_data, hull_aabb_index_buffer)
            .buf_arg(CompactHulls_k.Args.bounds_bank_data, hull_aabb_key_buffer);

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

        long compact_bones_k_ptr = scan_deletes.kernel_ptr(Kernel.compact_bones);
        compact_bones_k = new CompactBones_k(GPGPU.cl_cmd_queue_ptr, compact_bones_k_ptr)
            .buf_arg(CompactBones_k.Args.bone_shift, bone_shift)
            .buf_arg(CompactBones_k.Args.bone_instances, hull_bone_buffer)
            .buf_arg(CompactBones_k.Args.hull_bind_pose_indicies, hull_bind_pose_id_buffer)
            .buf_arg(CompactBones_k.Args.hull_inv_bind_pose_indicies, hull_inv_bind_pose_id_buffer);

        long compact_armature_bones_k_ptr = scan_deletes.kernel_ptr(Kernel.compact_armature_bones);
        compact_armature_bones_k = new CompactArmatureBones_k(GPGPU.cl_cmd_queue_ptr, compact_armature_bones_k_ptr)
            .buf_arg(CompactArmatureBones_k.Args.armature_bone_shift, bone_bind_shift)
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
            case ARMATURE                      -> armature_buffer;
            case ARMATURE_ACCEL                -> armature_accel_buffer;
            case ARMATURE_ANIM_ELAPSED         -> armature_anim_elapsed_buffer;
            case ARMATURE_ANIM_STATE           -> armature_anim_state_buffer;
            case ARMATURE_ANIM_INDEX           -> armature_anim_index_buffer;
            case ARMATURE_BONE                 -> armature_bone_buffer;
            case ARMATURE_BONE_REFERENCE_ID    -> armature_bone_reference_id_buffer;
            case ARMATURE_BONE_PARENT_ID       -> armature_bone_parent_id_buffer;
            case ARMATURE_FLAG                 -> armature_flag_buffer;
            case ARMATURE_BONE_TABLE           -> armature_bone_table_buffer;
            case ARMATURE_HULL_TABLE           -> armature_hull_table_buffer;
            case ARMATURE_MASS                 -> armature_mass_buffer;
            case ARMATURE_MODEL_ID             -> armature_model_id_buffer;
            case ARMATURE_ROOT_HULL            -> armature_root_hull_buffer;
            case ARMATURE_TRANSFORM_ID         -> armature_model_transform_buffer;
            case BONE_ANIM_TABLE               -> bone_anim_channel_table_buffer;
            case BONE_BIND_POSE                -> bone_bind_pose_buffer;
            case BONE_REFERENCE                -> bone_reference_buffer;
            case EDGE                          -> edge_buffer;
            case EDGE_FLAG                     -> edge_flag_buffer;
            case EDGE_LENGTH                   -> edge_length_buffer;
            case HULL                          -> hull_buffer;
            case HULL_SCALE                    -> hull_scale_buffer;
            case HULL_AABB                     -> hull_aabb_buffer;
            case HULL_AABB_INDEX               -> hull_aabb_index_buffer;
            case HULL_AABB_KEY_TABLE           -> hull_aabb_key_buffer;
            case HULL_BONE                     -> hull_bone_buffer;
            case HULL_ARMATURE_ID              -> hull_armature_id_buffer;
            case HULL_BONE_TABLE               -> hull_bone_table_buffer;
            case HULL_BONE_BIND_POSE           -> hull_bind_pose_id_buffer;
            case HULL_BONE_INV_BIND_POSE       -> hull_inv_bind_pose_id_buffer;
            case HULL_POINT_TABLE              -> hull_point_table_buffer;
            case HULL_EDGE_TABLE               -> hull_edge_table_buffer;
            case HULL_FLAG                     -> hull_flag_buffer;
            case HULL_FRICTION                 -> hull_friction_buffer;
            case HULL_RESTITUTION              -> hull_restitution_buffer;
            case HULL_MESH_ID                  -> hull_mesh_id_buffer;
            case HULL_ROTATION                 -> hull_rotation_buffer;
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
            case MIRROR_ARMATURE               -> mirror_armature_buffer;
            case MIRROR_POINT                  -> mirror_point_buffer;
            case MIRROR_ARMATURE_MODEL_ID      -> mirror_armature_model_id_buffer;
            case MIRROR_ARMATURE_ROOT_HULL     -> mirror_armature_root_hull_buffer;
            case MIRROR_EDGE_FLAG              -> mirror_edge_flag_buffer;
            case MIRROR_HULL_AABB              -> mirror_hull_aabb_buffer;
            case MIRROR_HULL_FLAG              -> mirror_hull_flag_buffer;
            case MIRROR_HULL_MESH_ID           -> mirror_hull_mesh_id_buffer;
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
        mirror_armature_buffer.mirror_buffer(armature_buffer);
        mirror_armature_model_id_buffer.mirror_buffer(armature_model_id_buffer);
        mirror_armature_root_hull_buffer.mirror_buffer(armature_root_hull_buffer);
        mirror_edge_buffer.mirror_buffer(edge_buffer);
        mirror_edge_flag_buffer.mirror_buffer(edge_flag_buffer);
        mirror_hull_buffer.mirror_buffer(hull_buffer);
        mirror_hull_aabb_buffer.mirror_buffer(hull_aabb_buffer);
        mirror_hull_flag_buffer.mirror_buffer(hull_flag_buffer);
        mirror_hull_mesh_id_buffer.mirror_buffer(hull_mesh_id_buffer);
        mirror_hull_point_table_buffer.mirror_buffer(hull_point_table_buffer);
        mirror_hull_rotation_buffer.mirror_buffer(hull_rotation_buffer);
        mirror_hull_scale_buffer.mirror_buffer(hull_scale_buffer);
        mirror_point_hit_count_buffer.mirror_buffer(point_hit_count_buffer);
        mirror_point_anti_gravity_buffer.mirror_buffer(point_anti_gravity_buffer);
        mirror_point_buffer.mirror_buffer(point_buffer);
        mirror_point_vertex_reference_buffer.mirror_buffer(point_vertex_reference_buffer);

        last_edge_index     = edge_index;
        last_armature_index = armature_index;
        last_hull_index     = hull_index;
        last_point_index    = point_index;
    }

    // index methods

    public int next_mesh()
    {
        return mesh_index;
    }

    public int next_armature()
    {
        return armature_index;
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

    public int next_bone()
    {
        return bone_index;
    }

    public int last_point()
    {
        return last_point_index;
    }

    public int last_armature()
    {
        return last_armature_index;
    }

    public int last_hull()
    {
        return last_hull_index;
    }

    public int last_edge()
    {
        return last_edge_index;
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
                        int armature_id,
                        int flags)
    {
        int capacity = hull_index + 1;
        hull_buffer.ensure_capacity(capacity);
        hull_scale_buffer.ensure_capacity(capacity);
        hull_mesh_id_buffer.ensure_capacity(capacity);
        hull_rotation_buffer.ensure_capacity(capacity);
        hull_point_table_buffer.ensure_capacity(capacity);
        hull_edge_table_buffer.ensure_capacity(capacity);
        hull_flag_buffer.ensure_capacity(capacity);
        hull_bone_table_buffer.ensure_capacity(capacity);
        hull_armature_id_buffer.ensure_capacity(capacity);
        hull_friction_buffer.ensure_capacity(capacity);
        hull_restitution_buffer.ensure_capacity(capacity);
        hull_aabb_buffer.ensure_capacity(capacity);
        hull_aabb_index_buffer.ensure_capacity(capacity);
        hull_aabb_key_buffer.ensure_capacity(capacity);

        create_hull_k
            .set_arg(CreateHull_k.Args.target, hull_index)
            .set_arg(CreateHull_k.Args.new_hull, position)
            .set_arg(CreateHull_k.Args.new_hull_scale, scale)
            .set_arg(CreateHull_k.Args.new_rotation, rotation)
            .set_arg(CreateHull_k.Args.new_friction, friction)
            .set_arg(CreateHull_k.Args.new_restitution, restitution)
            .set_arg(CreateHull_k.Args.new_point_table, point_table)
            .set_arg(CreateHull_k.Args.new_edge_table, edge_table)
            .set_arg(CreateHull_k.Args.new_bone_table, bone_table)
            .set_arg(CreateHull_k.Args.new_armature_id, armature_id)
            .set_arg(CreateHull_k.Args.new_flags, flags)
            .set_arg(CreateHull_k.Args.new_hull_mesh_id, mesh_id)
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

    public int new_armature(float x, float y,
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
        int capacity = armature_index + 1;
        armature_buffer.ensure_capacity(capacity);
        armature_flag_buffer.ensure_capacity(capacity);
        armature_root_hull_buffer.ensure_capacity(capacity);
        armature_model_id_buffer.ensure_capacity(capacity);
        armature_model_transform_buffer.ensure_capacity(capacity);
        armature_accel_buffer.ensure_capacity(capacity);
        armature_mass_buffer.ensure_capacity(capacity);
        armature_anim_index_buffer.ensure_capacity(capacity);
        armature_anim_elapsed_buffer.ensure_capacity(capacity);
        armature_anim_state_buffer.ensure_capacity(capacity);
        armature_hull_table_buffer.ensure_capacity(capacity);
        armature_bone_table_buffer.ensure_capacity(capacity);

        create_armature_k
            .set_arg(CreateArmature_k.Args.target, armature_index)
            .set_arg(CreateArmature_k.Args.new_armature, arg_float4(x, y, x, y))
            .set_arg(CreateArmature_k.Args.new_armature_root_hull, root_hull)
            .set_arg(CreateArmature_k.Args.new_armature_model_id, model_id)
            .set_arg(CreateArmature_k.Args.new_armature_model_transform, model_transform_id)
            .set_arg(CreateArmature_k.Args.new_armature_flags, flags)
            .set_arg(CreateArmature_k.Args.new_armature_hull_table, hull_table)
            .set_arg(CreateArmature_k.Args.new_armature_bone_table, bone_table)
            .set_arg(CreateArmature_k.Args.new_armature_mass, mass)
            .set_arg(CreateArmature_k.Args.new_armature_animation_index, anim_index)
            .set_arg(CreateArmature_k.Args.new_armature_animation_time, 0.0f)
            .set_arg(CreateArmature_k.Args.new_armature_animation_state, arg_short2((short) 0, (short) 0))
            .call(GPGPU.global_single_size);

        return armature_index++;
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
        int capacity = bone_index + 1;
        hull_bone_buffer.ensure_capacity(capacity);
        hull_bind_pose_id_buffer.ensure_capacity(capacity);
        hull_inv_bind_pose_id_buffer.ensure_capacity(capacity);

        create_bone_k
            .set_arg(CreateHullBone_k.Args.target, bone_index)
            .set_arg(CreateHullBone_k.Args.new_bone, bone_data)
            .set_arg(CreateHullBone_k.Args.new_hull_bind_pose_id, bind_pose_id)
            .set_arg(CreateHullBone_k.Args.new_hull_inv_bind_pose_id, inv_bind_pose_id)
            .call(GPGPU.global_single_size);

        return bone_index++;
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

    public void update_accel(int armature_index, float acc_x, float acc_y)
    {
        update_accel_k
            .set_arg(UpdateAccel_k.Args.target, armature_index)
            .set_arg(UpdateAccel_k.Args.new_value, arg_float2(acc_x, acc_y))
            .call(GPGPU.global_single_size);
    }

    public float[] read_position(int armature_index)
    {
        GPGPU.cl_zero_buffer(GPGPU.cl_cmd_queue_ptr, position_buffer_ptr, CLSize.cl_float2);

        read_position_k
            .ptr_arg(ReadPosition_k.Args.output, position_buffer_ptr)
            .set_arg(ReadPosition_k.Args.target, armature_index)
            .call(GPGPU.global_single_size);

        return GPGPU.cl_read_pinned_float_buffer(GPGPU.cl_cmd_queue_ptr, position_buffer_ptr, CLSize.cl_float2, 2);
    }

    public void delete_and_compact()
    {
        GPGPU.cl_zero_buffer(GPGPU.cl_cmd_queue_ptr, delete_counter_ptr, CLSize.cl_int);

        locate_out_of_bounds_k
            .ptr_arg(LocateOutOfBounds_k.Args.counter, delete_counter_ptr)
            .call(arg_long(armature_index));

        delete_buffer_1.ensure_capacity(armature_index);
        delete_buffer_2.ensure_capacity(armature_index);

        int[] shift_counts = scan_deletes(delete_buffer_1.pointer(), delete_buffer_2.pointer(), armature_index);

        if (shift_counts[4] == 0)
        {
            return;
        }

        hull_shift.ensure_capacity(hull_index);
        edge_shift.ensure_capacity(edge_index);
        point_shift.ensure_capacity(point_index);
        bone_shift.ensure_capacity(bone_index);
        bone_bind_shift.ensure_capacity(armature_bone_index);

        hull_shift.clear();
        edge_shift.clear();
        point_shift.clear();
        bone_shift.clear();
        bone_bind_shift.clear();

        compact_armatures_k
            .ptr_arg(CompactArmatures_k.Args.buffer_in_1, delete_buffer_1.pointer())
            .ptr_arg(CompactArmatures_k.Args.buffer_in_2, delete_buffer_2.pointer());

        linearize_kernel(compact_armatures_k, armature_index);
        linearize_kernel(compact_bones_k, bone_index);
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
        bone_index          -= (shift_counts[1]);
        point_index         -= (shift_counts[2]);
        hull_index          -= (shift_counts[3]);
        armature_index      -= (shift_counts[4]);
        armature_bone_index -= (shift_counts[5]);
    }

    // todo: implement armature rotations and update this
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
        bone_shift.release();
        bone_bind_shift.release();
        delete_buffer_1.release();
        delete_buffer_2.release();
        delete_partial_buffer_1.release();
        delete_partial_buffer_2.release();

        edge_buffer.release();
        edge_length_buffer.release();
        edge_flag_buffer.release();
        hull_buffer.release();
        hull_scale_buffer.release();
        hull_mesh_id_buffer.release();
        hull_rotation_buffer.release();
        hull_point_table_buffer.release();
        hull_edge_table_buffer.release();
        hull_flag_buffer.release();
        hull_bone_table_buffer.release();
        hull_armature_id_buffer.release();
        hull_aabb_buffer.release();
        hull_aabb_index_buffer.release();
        hull_aabb_key_buffer.release();
        hull_bone_buffer.release();
        hull_bind_pose_id_buffer.release();
        hull_inv_bind_pose_id_buffer.release();
        hull_friction_buffer.release();
        hull_restitution_buffer.release();
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
        armature_buffer.release();
        armature_flag_buffer.release();
        armature_root_hull_buffer.release();
        armature_model_id_buffer.release();
        armature_model_transform_buffer.release();
        armature_accel_buffer.release();
        armature_mass_buffer.release();
        armature_anim_index_buffer.release();
        armature_anim_elapsed_buffer.release();
        armature_anim_state_buffer.release();
        armature_hull_table_buffer.release();
        armature_bone_table_buffer.release();

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
        total += bone_shift.debug_data();
        total += bone_bind_shift.debug_data();
        total += delete_buffer_1.debug_data();
        total += delete_buffer_2.debug_data();
        total += delete_partial_buffer_1.debug_data();
        total += delete_partial_buffer_2.debug_data();
        total += edge_buffer.debug_data();
        total += edge_length_buffer.debug_data();
        total += edge_flag_buffer.debug_data();
        total += hull_buffer.debug_data();
        total += hull_scale_buffer.debug_data();
        total += hull_mesh_id_buffer.debug_data();
        total += hull_rotation_buffer.debug_data();
        total += hull_point_table_buffer.debug_data();
        total += hull_edge_table_buffer.debug_data();
        total += hull_flag_buffer.debug_data();
        total += hull_bone_table_buffer.debug_data();
        total += hull_armature_id_buffer.debug_data();
        total += hull_aabb_buffer.debug_data();
        total += hull_aabb_index_buffer.debug_data();
        total += hull_aabb_key_buffer.debug_data();
        total += hull_bone_buffer.debug_data();
        total += hull_bind_pose_id_buffer.debug_data();
        total += hull_inv_bind_pose_id_buffer.debug_data();
        total += hull_friction_buffer.debug_data();
        total += hull_restitution_buffer.debug_data();
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
        total += armature_buffer.debug_data();
        total += armature_flag_buffer.debug_data();
        total += armature_root_hull_buffer.debug_data();
        total += armature_model_id_buffer.debug_data();
        total += armature_model_transform_buffer.debug_data();
        total += armature_accel_buffer.debug_data();
        total += armature_mass_buffer.debug_data();
        total += armature_anim_index_buffer.debug_data();
        total += armature_anim_elapsed_buffer.debug_data();
        total += armature_anim_state_buffer.debug_data();
        total += armature_hull_table_buffer.debug_data();
        total += armature_bone_table_buffer.debug_data();

        //System.out.println("---------------------------");
        System.out.println("Core Memory Usage: MB " + ((float) total / 1024f / 1024f));
    }
}
