package com.controllerface.bvge.cl;

import com.controllerface.bvge.cl.kernels.*;
import com.controllerface.bvge.cl.programs.GPUCrud;
import com.controllerface.bvge.cl.programs.ScanDeletes;

import static com.controllerface.bvge.cl.CLUtils.*;

public class GPUCoreMemory
{
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

    private final GPUProgram gpu_crud = new GPUCrud();
    private final GPUProgram scan_deletes = new ScanDeletes();

    private final GPUKernel create_animation_timings_k;
    private final GPUKernel create_armature_k;
    private final GPUKernel create_armature_bone_k;
    private final GPUKernel create_bone_k;
    private final GPUKernel create_bone_bind_pose_k;
    private final GPUKernel create_bone_channel_k;
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
    private final GPUKernel read_position_k;
    private final GPUKernel update_accel_k;
    private final GPUKernel set_bone_channel_table_k;
    private final GPUKernel locate_out_of_bounds_k;
    private final GPUKernel scan_deletes_single_block_out_k;
    private final GPUKernel scan_deletes_multi_block_out_k;
    private final GPUKernel complete_deletes_multi_block_out_k;
    private final GPUKernel compact_armatures_k;
    private final GPUKernel compact_hulls_k;
    private final GPUKernel compact_edges_k;
    private final GPUKernel compact_points_k;
    private final GPUKernel compact_bones_k;
    private final GPUKernel compact_armature_bones_k;

    // internally used buffers
    private final ResizableBuffer hull_shift;
    private final ResizableBuffer edge_shift;
    private final ResizableBuffer point_shift;
    private final ResizableBuffer bone_shift;
    private final ResizableBuffer bone_bind_shift;
    private final ResizableBuffer delete_buffer_1;
    private final ResizableBuffer delete_buffer_2;
    private final ResizableBuffer delete_partial_buffer_1;
    private final ResizableBuffer delete_partial_buffer_2;
    private final ResizableBuffer bone_bind_parent_buffer; // todo: may not be needed, consider removing

    // externally used buffers
    private final ResizableBuffer edge_buffer;
    private final ResizableBuffer edge_length_buffer;
    private final ResizableBuffer edge_flag_buffer;
    private final ResizableBuffer hull_buffer;
    private final ResizableBuffer hull_mesh_id_buffer;
    private final ResizableBuffer hull_rotation_buffer;
    private final ResizableBuffer hull_element_table_buffer;
    private final ResizableBuffer hull_flag_buffer;
    private final ResizableBuffer hull_aabb_buffer;
    private final ResizableBuffer hull_aabb_index_buffer;
    private final ResizableBuffer hull_aabb_key_buffer;
    private final ResizableBuffer hull_bone_buffer;
    private final ResizableBuffer hull_bone_table_buffer;
    private final ResizableBuffer point_buffer;
    private final ResizableBuffer point_anti_gravity_buffer;
    private final ResizableBuffer point_vertex_table_buffer;
    private final ResizableBuffer point_bone_table_buffer;
    private final ResizableBuffer vertex_reference_buffer;
    private final ResizableBuffer vertex_weight_buffer;
    private final ResizableBuffer vertex_texture_uv_buffer;
    private final ResizableBuffer vertex_uv_table_buffer;
    private final ResizableBuffer model_transform_buffer;
    private final ResizableBuffer bone_reference_buffer;
    private final ResizableBuffer bone_bind_pose_buffer;

    private final ResizableBuffer mesh_reference_buffer;
    private final ResizableBuffer mesh_face_buffer;


    private final long delete_counter_ptr;
    private final long position_buffer_ptr;
    private final long delete_sizes_ptr;

    public GPUCoreMemory()
    {
        delete_counter_ptr = GPGPU.cl_new_int_arg_buffer(new int[]{ 0 });
        position_buffer_ptr = GPGPU.cl_new_pinned_buffer(CLSize.cl_float2);
        delete_sizes_ptr = GPGPU.cl_new_pinned_buffer(CLSize.cl_int * 6);

        hull_shift = new TransientBuffer(CLSize.cl_int);
        edge_shift = new TransientBuffer(CLSize.cl_int);
        point_shift = new TransientBuffer(CLSize.cl_int);
        bone_shift = new TransientBuffer(CLSize.cl_int);
        bone_bind_shift = new TransientBuffer(CLSize.cl_int);
        delete_buffer_1 = new TransientBuffer(CLSize.cl_int2);
        delete_buffer_2 = new TransientBuffer(CLSize.cl_int4);
        delete_partial_buffer_1 = new TransientBuffer(CLSize.cl_int2);
        delete_partial_buffer_2 = new TransientBuffer(CLSize.cl_int4);
        edge_buffer = new PersistentBuffer(CLSize.cl_int2);
        edge_length_buffer = new PersistentBuffer(CLSize.cl_float);
        edge_flag_buffer = new PersistentBuffer(CLSize.cl_int);
        hull_buffer = new PersistentBuffer(CLSize.cl_float4);
        hull_mesh_id_buffer = new PersistentBuffer(CLSize.cl_int);
        hull_rotation_buffer = new PersistentBuffer(CLSize.cl_float2);
        hull_element_table_buffer = new PersistentBuffer(CLSize.cl_int4);
        hull_flag_buffer = new PersistentBuffer(CLSize.cl_int4);
        hull_aabb_buffer = new PersistentBuffer(CLSize.cl_float4);
        hull_aabb_index_buffer = new PersistentBuffer(CLSize.cl_int4);
        hull_aabb_key_buffer = new PersistentBuffer(CLSize.cl_int2);
        hull_bone_buffer = new PersistentBuffer(CLSize.cl_float16);
        hull_bone_table_buffer = new PersistentBuffer(CLSize.cl_int2);
        point_buffer = new PersistentBuffer(CLSize.cl_float4, 50_000L);
        point_anti_gravity_buffer = new PersistentBuffer(CLSize.cl_float, 50_000L);
        point_vertex_table_buffer = new PersistentBuffer(CLSize.cl_int4, 50_000L);
        point_bone_table_buffer = new PersistentBuffer(CLSize.cl_int4, 50_000L);
        vertex_reference_buffer = new PersistentBuffer(CLSize.cl_float2);
        vertex_weight_buffer = new PersistentBuffer(CLSize.cl_float4);
        vertex_texture_uv_buffer = new PersistentBuffer(CLSize.cl_float2);
        vertex_uv_table_buffer = new PersistentBuffer(CLSize.cl_int2);
        model_transform_buffer = new PersistentBuffer(CLSize.cl_float16);
        bone_reference_buffer = new PersistentBuffer(CLSize.cl_float16);
        bone_bind_pose_buffer = new PersistentBuffer(CLSize.cl_float16);
        bone_bind_parent_buffer = new PersistentBuffer(CLSize.cl_int);

        mesh_reference_buffer = new PersistentBuffer(CLSize.cl_int4);
        mesh_face_buffer = new PersistentBuffer(CLSize.cl_int4);

        gpu_crud.init();
        scan_deletes.init();

        // create methods

        long create_point_k_ptr = gpu_crud.kernel_ptr(Kernel.create_point);
        create_point_k = new CreatePoint_k(GPGPU.command_queue_ptr, create_point_k_ptr)
            .buf_arg(CreatePoint_k.Args.points, point_buffer)
            .buf_arg(CreatePoint_k.Args.vertex_tables, point_vertex_table_buffer)
            .buf_arg(CreatePoint_k.Args.bone_tables, point_bone_table_buffer);

        long create_texture_uv_ptr = gpu_crud.kernel_ptr(Kernel.create_texture_uv);
        create_texture_uv_k = new CreateTextureUV_k(GPGPU.command_queue_ptr, create_texture_uv_ptr)
            .buf_arg(CreateTextureUV_k.Args.texture_uvs, vertex_texture_uv_buffer);

        long create_edge_k_ptr = gpu_crud.kernel_ptr(Kernel.create_edge);
        create_edge_k = new CreateEdge_k(GPGPU.command_queue_ptr, create_edge_k_ptr)
            .buf_arg(CreateEdge_k.Args.edges, edge_buffer)
            .buf_arg(CreateEdge_k.Args.edge_lengths, edge_length_buffer)
            .buf_arg(CreateEdge_k.Args.edge_flags, edge_flag_buffer);

        long create_keyframe_k_ptr = gpu_crud.kernel_ptr(Kernel.create_keyframe);
        create_keyframe_k = new CreateKeyFrame_k(GPGPU.command_queue_ptr, create_keyframe_k_ptr)
            .ptr_arg(CreateKeyFrame_k.Args.key_frames, GPGPU.Buffer.animation_key_frames.pointer)
            .ptr_arg(CreateKeyFrame_k.Args.frame_times, GPGPU.Buffer.animation_frame_times.pointer);

        long create_vertex_reference_k_ptr = gpu_crud.kernel_ptr(Kernel.create_vertex_reference);
        create_vertex_reference_k = new CreateVertexRef_k(GPGPU.command_queue_ptr, create_vertex_reference_k_ptr)
            .buf_arg(CreateVertexRef_k.Args.vertex_references, vertex_reference_buffer)
            .buf_arg(CreateVertexRef_k.Args.vertex_weights, vertex_weight_buffer)
            .buf_arg(CreateVertexRef_k.Args.uv_tables, vertex_uv_table_buffer);

        long create_bone_bind_pose_k_ptr = gpu_crud.kernel_ptr(Kernel.create_bone_bind_pose);
        create_bone_bind_pose_k = new CreateBoneBindPose_k(GPGPU.command_queue_ptr, create_bone_bind_pose_k_ptr)
            .buf_arg(CreateBoneBindPose_k.Args.bone_bind_poses, bone_bind_pose_buffer)
            .buf_arg(CreateBoneBindPose_k.Args.bone_bind_parents, bone_bind_parent_buffer);

        long create_bone_reference_k_ptr = gpu_crud.kernel_ptr(Kernel.create_bone_reference);
        create_bone_reference_k = new CreateBoneRef_k(GPGPU.command_queue_ptr, create_bone_reference_k_ptr)
            .buf_arg(CreateBoneRef_k.Args.bone_references, bone_reference_buffer);

        long create_bone_channel_k_ptr = gpu_crud.kernel_ptr(Kernel.create_bone_channel);
        create_bone_channel_k = new CreateBoneChannel_k(GPGPU.command_queue_ptr, create_bone_channel_k_ptr)
            .ptr_arg(CreateBoneChannel_k.Args.animation_timing_indices, GPGPU.Buffer.animation_timing_indices.pointer)
            .ptr_arg(CreateBoneChannel_k.Args.bone_pos_channel_tables, GPGPU.Buffer.animation_bone_pos_channel_tables.pointer)
            .ptr_arg(CreateBoneChannel_k.Args.bone_rot_channel_tables, GPGPU.Buffer.animation_bone_rot_channel_tables.pointer)
            .ptr_arg(CreateBoneChannel_k.Args.bone_scl_channel_tables, GPGPU.Buffer.animation_bone_scl_channel_tables.pointer);

        long create_armature_k_ptr = gpu_crud.kernel_ptr(Kernel.create_armature);
        create_armature_k = new CreateArmature_k(GPGPU.command_queue_ptr, create_armature_k_ptr)
            .ptr_arg(CreateArmature_k.Args.armatures, GPGPU.Buffer.armatures.pointer)
            .ptr_arg(CreateArmature_k.Args.armature_flags, GPGPU.Buffer.armature_flags.pointer)
            .ptr_arg(CreateArmature_k.Args.hull_tables, GPGPU.Buffer.armature_hull_table.pointer)
            .ptr_arg(CreateArmature_k.Args.armature_masses, GPGPU.Buffer.armature_mass.pointer)
            .ptr_arg(CreateArmature_k.Args.armature_animation_indices, GPGPU.Buffer.armature_animation_indices.pointer)
            .ptr_arg(CreateArmature_k.Args.armature_animation_elapsed, GPGPU.Buffer.armature_animation_elapsed.pointer);

        long create_bone_k_ptr = gpu_crud.kernel_ptr(Kernel.create_bone);
        create_bone_k = new CreateBone_k(GPGPU.command_queue_ptr, create_bone_k_ptr)
            .buf_arg(CreateBone_k.Args.bones, hull_bone_buffer)
            .buf_arg(CreateBone_k.Args.bone_index_tables, hull_bone_table_buffer);

        long create_armature_bone_k_ptr = gpu_crud.kernel_ptr(Kernel.create_armature_bone);
        create_armature_bone_k = new CreateArmatureBone_k(GPGPU.command_queue_ptr, create_armature_bone_k_ptr)
            .ptr_arg(CreateArmatureBone_k.Args.armature_bones, GPGPU.Buffer.armature_bones.pointer)
            .ptr_arg(CreateArmatureBone_k.Args.bone_bind_tables, GPGPU.Buffer.armature_bone_tables.pointer);

        long create_model_transform_k_ptr = gpu_crud.kernel_ptr(Kernel.create_model_transform);
        create_model_transform_k = new CreateModelTransform_k(GPGPU.command_queue_ptr, create_model_transform_k_ptr)
            .buf_arg(CreateModelTransform_k.Args.model_transforms, model_transform_buffer);

        long create_hull_k_ptr = gpu_crud.kernel_ptr(Kernel.create_hull);
        create_hull_k = new CreateHull_k(GPGPU.command_queue_ptr, create_hull_k_ptr)
            .buf_arg(CreateHull_k.Args.hulls, hull_buffer)
            .buf_arg(CreateHull_k.Args.hull_rotations, hull_rotation_buffer)
            .buf_arg(CreateHull_k.Args.element_tables, hull_element_table_buffer)
            .buf_arg(CreateHull_k.Args.hull_flags, hull_flag_buffer)
            .buf_arg(CreateHull_k.Args.hull_mesh_ids, hull_mesh_id_buffer);

        long create_mesh_reference_k_ptr = gpu_crud.kernel_ptr(Kernel.create_mesh_reference);
        create_mesh_reference_k = new CreateMeshReference_k(GPGPU.command_queue_ptr, create_mesh_reference_k_ptr)
            .buf_arg(CreateMeshReference_k.Args.mesh_ref_tables, mesh_reference_buffer);

        long create_mesh_face_k_ptr = gpu_crud.kernel_ptr(Kernel.create_mesh_face);
        create_mesh_face_k = new CreateMeshFace_k(GPGPU.command_queue_ptr, create_mesh_face_k_ptr)
            .buf_arg(CreateMeshFace_k.Args.mesh_faces, mesh_face_buffer);

        long create_animation_timings_k_ptr = gpu_crud.kernel_ptr(Kernel.create_animation_timings);
        create_animation_timings_k = new CreateAnimationTimings_k(GPGPU.command_queue_ptr, create_animation_timings_k_ptr)
            .ptr_arg(CreateAnimationTimings_k.Args.animation_timings, GPGPU.Buffer.animation_timings.pointer);

        // read methods

        long read_position_k_ptr = gpu_crud.kernel_ptr(Kernel.read_position);
        read_position_k = new ReadPosition_k(GPGPU.command_queue_ptr, read_position_k_ptr)
            .ptr_arg(ReadPosition_k.Args.armatures, GPGPU.Buffer.armatures.pointer);

        // update methods

        long update_accel_k_ptr = gpu_crud.kernel_ptr(Kernel.update_accel);
        update_accel_k = new UpdateAccel_k(GPGPU.command_queue_ptr, update_accel_k_ptr)
            .ptr_arg(UpdateAccel_k.Args.armature_accel, GPGPU.Buffer.armature_accel.pointer);

        long set_bone_channel_table_k_ptr = gpu_crud.kernel_ptr(Kernel.set_bone_channel_table);
        set_bone_channel_table_k = new SetBoneChannelTable_k(GPGPU.command_queue_ptr, set_bone_channel_table_k_ptr)
            .ptr_arg(SetBoneChannelTable_k.Args.bone_channel_tables, GPGPU.Buffer.animation_bone_channel_tables.pointer);

        // delete methods

        long locate_out_of_bounds_k_ptr = scan_deletes.kernel_ptr(Kernel.locate_out_of_bounds);
        locate_out_of_bounds_k = new LocateOutOfBounds_k(GPGPU.command_queue_ptr, locate_out_of_bounds_k_ptr)
            .ptr_arg(LocateOutOfBounds_k.Args.hull_tables, GPGPU.Buffer.armature_hull_table.pointer)
            .buf_arg(LocateOutOfBounds_k.Args.hull_flags, hull_flag_buffer)
            .ptr_arg(LocateOutOfBounds_k.Args.armature_flags, GPGPU.Buffer.armature_flags.pointer);

        long scan_deletes_single_block_out_k_ptr = scan_deletes.kernel_ptr(Kernel.scan_deletes_single_block_out);
        scan_deletes_single_block_out_k = new ScanDeletesSingleBlockOut_k(GPGPU.command_queue_ptr, scan_deletes_single_block_out_k_ptr)
            .ptr_arg(ScanDeletesSingleBlockOut_k.Args.sz, delete_sizes_ptr)
            .ptr_arg(ScanDeletesSingleBlockOut_k.Args.armature_flags, GPGPU.Buffer.armature_flags.pointer)
            .ptr_arg(ScanDeletesSingleBlockOut_k.Args.hull_tables, GPGPU.Buffer.armature_hull_table.pointer)
            .buf_arg(ScanDeletesSingleBlockOut_k.Args.element_tables, hull_element_table_buffer)
            .buf_arg(ScanDeletesSingleBlockOut_k.Args.hull_flags, hull_flag_buffer);

        long scan_deletes_multi_block_out_k_ptr = scan_deletes.kernel_ptr(Kernel.scan_deletes_multi_block_out);
        scan_deletes_multi_block_out_k = new ScanDeletesMultiBlockOut_k(GPGPU.command_queue_ptr, scan_deletes_multi_block_out_k_ptr)
            .buf_arg(ScanDeletesMultiBlockOut_k.Args.part1, delete_partial_buffer_1)
            .buf_arg(ScanDeletesMultiBlockOut_k.Args.part2, delete_partial_buffer_2)
            .ptr_arg(ScanDeletesMultiBlockOut_k.Args.armature_flags, GPGPU.Buffer.armature_flags.pointer)
            .ptr_arg(ScanDeletesMultiBlockOut_k.Args.hull_tables, GPGPU.Buffer.armature_hull_table.pointer)
            .buf_arg(ScanDeletesMultiBlockOut_k.Args.element_tables, hull_element_table_buffer)
            .buf_arg(ScanDeletesMultiBlockOut_k.Args.hull_flags, hull_flag_buffer);

        long complete_deletes_multi_block_out_k_ptr = scan_deletes.kernel_ptr(Kernel.complete_deletes_multi_block_out);
        complete_deletes_multi_block_out_k = new CompleteDeletesMultiBlockOut_k(GPGPU.command_queue_ptr, complete_deletes_multi_block_out_k_ptr)
            .ptr_arg(CompleteDeletesMultiBlockOut_k.Args.sz, delete_sizes_ptr)
            .buf_arg(CompleteDeletesMultiBlockOut_k.Args.part1, delete_partial_buffer_1)
            .buf_arg(CompleteDeletesMultiBlockOut_k.Args.part2, delete_partial_buffer_2)
            .ptr_arg(CompleteDeletesMultiBlockOut_k.Args.armature_flags, GPGPU.Buffer.armature_flags.pointer)
            .ptr_arg(CompleteDeletesMultiBlockOut_k.Args.hull_tables, GPGPU.Buffer.armature_hull_table.pointer)
            .buf_arg(CompleteDeletesMultiBlockOut_k.Args.element_tables, hull_element_table_buffer)
            .buf_arg(CompleteDeletesMultiBlockOut_k.Args.hull_flags, hull_flag_buffer);

        long compact_armatures_k_ptr = scan_deletes.kernel_ptr(Kernel.compact_armatures);
        compact_armatures_k = new CompactArmatures_k(GPGPU.command_queue_ptr, compact_armatures_k_ptr)
            .ptr_arg(CompactArmatures_k.Args.armatures, GPGPU.Buffer.armatures.pointer)
            .ptr_arg(CompactArmatures_k.Args.armature_accel, GPGPU.Buffer.armature_accel.pointer)
            .ptr_arg(CompactArmatures_k.Args.armature_flags, GPGPU.Buffer.armature_flags.pointer)
            .ptr_arg(CompactArmatures_k.Args.armature_animation_indices, GPGPU.Buffer.armature_animation_indices.pointer)
            .ptr_arg(CompactArmatures_k.Args.armature_animation_elapsed, GPGPU.Buffer.armature_animation_elapsed.pointer)
            .ptr_arg(CompactArmatures_k.Args.hull_tables, GPGPU.Buffer.armature_hull_table.pointer)
            .buf_arg(CompactArmatures_k.Args.hulls, hull_buffer)
            .buf_arg(CompactArmatures_k.Args.hull_flags, hull_flag_buffer)
            .buf_arg(CompactArmatures_k.Args.element_tables, hull_element_table_buffer)
            .buf_arg(CompactArmatures_k.Args.points, point_buffer)
            .buf_arg(CompactArmatures_k.Args.vertex_tables, point_vertex_table_buffer)
            .buf_arg(CompactArmatures_k.Args.bone_tables, point_bone_table_buffer)
            .ptr_arg(CompactArmatures_k.Args.bone_bind_tables, GPGPU.Buffer.armature_bone_tables.pointer)
            .buf_arg(CompactArmatures_k.Args.bone_index_tables, hull_bone_table_buffer)
            .buf_arg(CompactArmatures_k.Args.edges, edge_buffer)
            .buf_arg(CompactArmatures_k.Args.bone_shift, bone_shift)
            .buf_arg(CompactArmatures_k.Args.point_shift, point_shift)
            .buf_arg(CompactArmatures_k.Args.edge_shift, edge_shift)
            .buf_arg(CompactArmatures_k.Args.hull_shift, hull_shift)
            .buf_arg(CompactArmatures_k.Args.bone_bind_shift, bone_bind_shift);

        long compact_hulls_k_ptr = scan_deletes.kernel_ptr(Kernel.compact_hulls);
        compact_hulls_k = new CompactHulls_k(GPGPU.command_queue_ptr, compact_hulls_k_ptr)
            .buf_arg(CompactHulls_k.Args.hull_shift, hull_shift)
            .buf_arg(CompactHulls_k.Args.hulls, hull_buffer)
            .buf_arg(CompactHulls_k.Args.hull_mesh_ids, hull_mesh_id_buffer)
            .buf_arg(CompactHulls_k.Args.hull_rotations, hull_rotation_buffer)
            .buf_arg(CompactHulls_k.Args.hull_flags, hull_flag_buffer)
            .buf_arg(CompactHulls_k.Args.element_tables, hull_element_table_buffer)
            .buf_arg(CompactHulls_k.Args.bounds, hull_aabb_buffer)
            .buf_arg(CompactHulls_k.Args.bounds_index_data, hull_aabb_index_buffer)
            .buf_arg(CompactHulls_k.Args.bounds_bank_data, hull_aabb_key_buffer);

        long compact_edges_k_ptr = scan_deletes.kernel_ptr(Kernel.compact_edges);
        compact_edges_k = new CompactEdges_k(GPGPU.command_queue_ptr, compact_edges_k_ptr)
            .buf_arg(CompactEdges_k.Args.edge_shift, edge_shift)
            .buf_arg(CompactEdges_k.Args.edges, edge_buffer)
            .buf_arg(CompactEdges_k.Args.edge_lengths, edge_length_buffer)
            .buf_arg(CompactEdges_k.Args.edge_flags, edge_flag_buffer);

        long compact_points_k_ptr = scan_deletes.kernel_ptr(Kernel.compact_points);
        compact_points_k = new CompactPoints_k(GPGPU.command_queue_ptr, compact_points_k_ptr)
            .buf_arg(CompactPoints_k.Args.point_shift, point_shift)
            .buf_arg(CompactPoints_k.Args.points, point_buffer)
            .buf_arg(CompactPoints_k.Args.anti_gravity, point_anti_gravity_buffer)
            .buf_arg(CompactPoints_k.Args.vertex_tables, point_vertex_table_buffer)
            .buf_arg(CompactPoints_k.Args.bone_tables, point_bone_table_buffer);

        long compact_bones_k_ptr = scan_deletes.kernel_ptr(Kernel.compact_bones);
        compact_bones_k = new CompactBones_k(GPGPU.command_queue_ptr, compact_bones_k_ptr)
            .buf_arg(CompactBones_k.Args.bone_shift, bone_shift)
            .buf_arg(CompactBones_k.Args.bone_instances, hull_bone_buffer)
            .buf_arg(CompactBones_k.Args.bone_index_tables, hull_bone_table_buffer);

        long compact_armature_bones_k_ptr = scan_deletes.kernel_ptr(Kernel.compact_armature_bones);
        compact_armature_bones_k = new CompactArmatureBones_k(GPGPU.command_queue_ptr, compact_armature_bones_k_ptr)
            .buf_arg(CompactArmatureBones_k.Args.armature_bone_shift, bone_bind_shift)
            .ptr_arg(CompactArmatureBones_k.Args.armature_bones, GPGPU.Buffer.armature_bones.pointer)
            .ptr_arg(CompactArmatureBones_k.Args.armature_bone_tables, GPGPU.Buffer.armature_bone_tables.pointer);
    }

    public ResizableBuffer buffer(BufferType bufferType)
    {
        return switch (bufferType)
        {
            case EDGE -> edge_buffer;
            case EDGE_FLAG -> edge_flag_buffer;
            case EDGE_LENGTH -> edge_length_buffer;
            case HULL -> hull_buffer;
            case HULL_MESH_ID -> hull_mesh_id_buffer;
            case HULL_ROTATION -> hull_rotation_buffer;
            case HULL_ELEMENT_TABLE -> hull_element_table_buffer;
            case HULL_FLAG -> hull_flag_buffer;
            case HULL_AABB -> hull_aabb_buffer;
            case HULL_AABB_INDEX -> hull_aabb_index_buffer;
            case HULL_AABB_KEY_TABLE -> hull_aabb_key_buffer;
            case HULL_BONE -> hull_bone_buffer;
            case HULL_BONE_TABLE -> hull_bone_table_buffer;
            case POINT -> point_buffer;
            case POINT_ANTI_GRAV -> point_anti_gravity_buffer;
            case POINT_VERTEX_TABLE -> point_vertex_table_buffer;
            case POINT_BONE_TABLE -> point_bone_table_buffer;
            case VERTEX_REFERENCE -> vertex_reference_buffer;
            case VERTEX_WEIGHT -> vertex_weight_buffer;
            case VERTEX_TEXTURE_UV -> vertex_texture_uv_buffer;
            case VERTEX_UV_TABLE -> vertex_uv_table_buffer;
            case MODEL_TRANSFORM -> model_transform_buffer;
            case BONE_REFERENCE -> bone_reference_buffer;
            case BONE_BIND_POSE -> bone_bind_pose_buffer;
            case MESH_REFERENCE -> mesh_reference_buffer;
            case MESH_FACE -> mesh_face_buffer;
        };
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

    public int new_animation_timings(double[] timings)
    {
        create_animation_timings_k
            .set_arg(CreateAnimationTimings_k.Args.target, animation_index)
            .set_arg(CreateAnimationTimings_k.Args.new_animation_timing, timings)
            .call(GPGPU.global_single_size);

        return animation_index++;
    }

    public int new_bone_channel(int anim_timing_index, int[] pos_table, int[] rot_table, int[] scl_table)
    {
        create_bone_channel_k
            .set_arg(CreateBoneChannel_k.Args.target, bone_channel_index)
            .set_arg(CreateBoneChannel_k.Args.new_animation_timing_index, anim_timing_index)
            .set_arg(CreateBoneChannel_k.Args.new_bone_pos_channel_table, pos_table)
            .set_arg(CreateBoneChannel_k.Args.new_bone_rot_channel_table, rot_table)
            .set_arg(CreateBoneChannel_k.Args.new_bone_scl_channel_table, scl_table)
            .call(GPGPU.global_single_size);

        return bone_channel_index++;
    }

    public int new_keyframe(float[] frame, double time)
    {
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

    public int new_point(float[] position, int[] vertex_table, int[] bone_ids)
    {
        int capacity = point_index + 1;
        point_buffer.ensure_capacity(capacity);
        point_anti_gravity_buffer.ensure_capacity(capacity);
        point_vertex_table_buffer.ensure_capacity(capacity);
        point_bone_table_buffer.ensure_capacity(capacity);

        var new_point = new float[]{position[0], position[1], position[0], position[1]};
        create_point_k
            .set_arg(CreatePoint_k.Args.target, point_index)
            .set_arg(CreatePoint_k.Args.new_point, new_point)
            .set_arg(CreatePoint_k.Args.new_vertex_table, vertex_table)
            .set_arg(CreatePoint_k.Args.new_bone_table, bone_ids)
            .call(GPGPU.global_single_size);

        return point_index++;
    }

    public int new_hull(int mesh_id, float[] transform, float[] rotation, int[] table, int[] flags)
    {
        int capacity = hull_index + 1;
        hull_buffer.ensure_capacity(capacity);
        hull_mesh_id_buffer.ensure_capacity(capacity);
        hull_rotation_buffer.ensure_capacity(capacity);
        hull_element_table_buffer.ensure_capacity(capacity);
        hull_flag_buffer.ensure_capacity(capacity);
        hull_aabb_buffer.ensure_capacity(capacity);
        hull_aabb_index_buffer.ensure_capacity(capacity);
        hull_aabb_key_buffer.ensure_capacity(capacity);

        create_hull_k
            .set_arg(CreateHull_k.Args.target, hull_index)
            .set_arg(CreateHull_k.Args.new_hull, transform)
            .set_arg(CreateHull_k.Args.new_rotation, rotation)
            .set_arg(CreateHull_k.Args.new_table, table)
            .set_arg(CreateHull_k.Args.new_flags, flags)
            .set_arg(CreateHull_k.Args.new_hull_mesh_id, mesh_id)
            .call(GPGPU.global_single_size);

        return hull_index++;
    }

    public int new_mesh_reference(int[] mesh_ref_table)
    {
        int capacity = mesh_index + 1;
        mesh_reference_buffer.ensure_capacity(capacity);

        create_mesh_reference_k
            .set_arg(CreateMeshReference_k.Args.target, mesh_index)
            .set_arg(CreateMeshReference_k.Args.new_mesh_ref_table, mesh_ref_table)
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

    public int new_armature(float x, float y, int[] table, int[] flags, float mass, int anim_index, double anim_time)
    {
        create_armature_k
            .set_arg(CreateArmature_k.Args.target, armature_index)
            .set_arg(CreateArmature_k.Args.new_armature, arg_float4(x, y, x, y))
            .set_arg(CreateArmature_k.Args.new_armature_flags, flags)
            .set_arg(CreateArmature_k.Args.new_hull_table, table)
            .set_arg(CreateArmature_k.Args.new_armature_mass, mass)
            .set_arg(CreateArmature_k.Args.new_armature_animation_index, anim_index)
            .set_arg(CreateArmature_k.Args.new_armature_animation_time, anim_time)
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

    public int new_bone_bind_pose(int bind_parent, float[] bone_data)
    {
        int capacity = bone_bind_index + 1;
        bone_bind_pose_buffer.ensure_capacity(capacity);
        bone_bind_parent_buffer.ensure_capacity(capacity);

        create_bone_bind_pose_k
            .set_arg(CreateBoneBindPose_k.Args.target,bone_bind_index)
            .set_arg(CreateBoneBindPose_k.Args.new_bone_bind_pose, bone_data)
            .set_arg(CreateBoneBindPose_k.Args.bone_bind_parent, bind_parent)
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

    public int new_bone(int[] bone_table, float[] bone_data)
    {
        int capacity = bone_index + 1;
        hull_bone_buffer.ensure_capacity(capacity);
        hull_bone_table_buffer.ensure_capacity(capacity);

        create_bone_k
            .set_arg(CreateBone_k.Args.target, bone_index)
            .set_arg(CreateBone_k.Args.new_bone, bone_data)
            .set_arg(CreateBone_k.Args.new_bone_table, bone_table)
            .call(GPGPU.global_single_size);

        return bone_index++;
    }

    public int new_armature_bone(int[] bone_bind_table, float[] bone_data)
    {
        create_armature_bone_k
            .set_arg(CreateArmatureBone_k.Args.target, armature_bone_index)
            .set_arg(CreateArmatureBone_k.Args.new_armature_bone, bone_data)
            .set_arg(CreateArmatureBone_k.Args.new_bone_bind_table, bone_bind_table)
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

    public void set_bone_channel_table(int bone_channel_index, int[] channel_table)
    {
        set_bone_channel_table_k
            .set_arg(SetBoneChannelTable_k.Args.target, bone_channel_index)
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
        GPGPU.cl_zero_buffer(position_buffer_ptr, CLSize.cl_float2);

        read_position_k
            .ptr_arg(ReadPosition_k.Args.output, position_buffer_ptr)
            .set_arg(ReadPosition_k.Args.target, armature_index)
            .call(GPGPU.global_single_size);

        return GPGPU.cl_read_pinned_float_buffer(position_buffer_ptr, CLSize.cl_float2, 2);
    }

    public void delete_and_compact()
    {
        GPGPU.cl_zero_buffer(delete_counter_ptr, CLSize.cl_int);

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

        GPGPU.cl_zero_buffer(delete_sizes_ptr, CLSize.cl_int * 6);

        scan_deletes_single_block_out_k
            .ptr_arg(ScanDeletesSingleBlockOut_k.Args.output, o1_data_ptr)
            .ptr_arg(ScanDeletesSingleBlockOut_k.Args.output2, o2_data_ptr)
            .loc_arg(ScanDeletesSingleBlockOut_k.Args.buffer, local_buffer_size)
            .loc_arg(ScanDeletesSingleBlockOut_k.Args.buffer2, local_buffer_size2)
            .set_arg(ScanDeletesSingleBlockOut_k.Args.n, n)
            .call(GPGPU.local_work_default, GPGPU.local_work_default);

        return GPGPU.cl_read_pinned_int_buffer(delete_sizes_ptr, CLSize.cl_int * 6, 6);
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

        GPGPU.cl_zero_buffer(delete_sizes_ptr, CLSize.cl_int * 6);

        complete_deletes_multi_block_out_k
            .ptr_arg(CompleteDeletesMultiBlockOut_k.Args.output1, o1_data_ptr)
            .ptr_arg(CompleteDeletesMultiBlockOut_k.Args.output2, o2_data_ptr)
            .loc_arg(CompleteDeletesMultiBlockOut_k.Args.buffer1, local_buffer_size)
            .loc_arg(CompleteDeletesMultiBlockOut_k.Args.buffer2, local_buffer_size2)
            .set_arg(CompleteDeletesMultiBlockOut_k.Args.n, n)
            .call(global_work_size, GPGPU.local_work_default);

        return GPGPU.cl_read_pinned_int_buffer(delete_sizes_ptr, CLSize.cl_int * 6, 6);
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
        System.out.println("--- shutting down --- ");

        System.out.println("hulls      : " + hull_index);
        System.out.println("points     : " + point_index);
        System.out.println("edges      : " + edge_index);
        System.out.println("bones      : " + bone_index);
        System.out.println("armatures  : " + armature_index);

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
        hull_mesh_id_buffer.release();
        hull_rotation_buffer.release();
        hull_element_table_buffer.release();
        hull_flag_buffer.release();
        hull_aabb_buffer.release();
        hull_aabb_index_buffer.release();
        hull_aabb_key_buffer.release();
        hull_bone_buffer.release();
        hull_bone_table_buffer.release();

        point_buffer.release();
        point_anti_gravity_buffer.release();
        point_vertex_table_buffer.release();
        point_bone_table_buffer.release();

        GPGPU.cl_release_buffer(delete_counter_ptr);
        GPGPU.cl_release_buffer(position_buffer_ptr);
        GPGPU.cl_release_buffer(delete_sizes_ptr);
    }
}
