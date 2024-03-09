package com.controllerface.bvge.gpu;

import com.controllerface.bvge.cl.*;
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

    private GPUKernel create_animation_timings_k;
    private GPUKernel create_armature_k;
    private GPUKernel create_armature_bone_k;
    private GPUKernel create_bone_k;
    private GPUKernel create_bone_bind_pose_k;
    private GPUKernel create_bone_channel_k;
    private GPUKernel create_bone_reference_k;
    private GPUKernel create_edge_k;
    private GPUKernel create_hull_k;
    private GPUKernel create_keyframe_k;
    private GPUKernel create_mesh_face_k;
    private GPUKernel create_mesh_reference_k;
    private GPUKernel create_model_transform_k;
    private GPUKernel create_point_k;
    private GPUKernel create_texture_uv_k;
    private GPUKernel create_vertex_reference_k;
    private GPUKernel read_position_k;
    private GPUKernel update_accel_k;
    private GPUKernel set_bone_channel_table_k;
    private GPUKernel locate_out_of_bounds_k;
    private GPUKernel scan_deletes_single_block_out_k;
    private GPUKernel scan_deletes_multi_block_out_k;
    private GPUKernel complete_deletes_multi_block_out_k;
    private GPUKernel compact_armatures_k;
    private GPUKernel compact_hulls_k;
    private GPUKernel compact_edges_k;
    private GPUKernel compact_points_k;
    private GPUKernel compact_bones_k;
    private GPUKernel compact_armature_bones_k;

    public GPUCoreMemory()
    {
        init();
    }

    private void init()
    {
        gpu_crud.init();
        scan_deletes.init();

        long create_point_k_ptr = gpu_crud.kernel_ptr(GPU.Kernel.create_point);
        create_point_k = new CreatePoint_k(GPU.command_queue_ptr, create_point_k_ptr)
            .mem_arg(CreatePoint_k.Args.points, GPU.Buffer.points.memory)
            .mem_arg(CreatePoint_k.Args.vertex_tables, GPU.Buffer.point_vertex_tables.memory)
            .mem_arg(CreatePoint_k.Args.bone_tables, GPU.Buffer.point_bone_tables.memory);

        long create_texture_uv_ptr = gpu_crud.kernel_ptr(GPU.Kernel.create_texture_uv);
        create_texture_uv_k = new CreateTextureUV_k(GPU.command_queue_ptr, create_texture_uv_ptr)
            .mem_arg(CreateTextureUV_k.Args.texture_uvs, GPU.Buffer.texture_uvs.memory);

        long create_edge_k_ptr = gpu_crud.kernel_ptr(GPU.Kernel.create_edge);
        create_edge_k = new CreateEdge_k(GPU.command_queue_ptr, create_edge_k_ptr)
            .mem_arg(CreateEdge_k.Args.edges, GPU.Buffer.edges.memory);

        long create_keyframe_k_ptr = gpu_crud.kernel_ptr(GPU.Kernel.create_keyframe);
        create_keyframe_k = new CreateKeyFrame_k(GPU.command_queue_ptr, create_keyframe_k_ptr)
            .mem_arg(CreateKeyFrame_k.Args.key_frames, GPU.Buffer.key_frames.memory)
            .mem_arg(CreateKeyFrame_k.Args.frame_times, GPU.Buffer.frame_times.memory);

        long create_vertex_reference_k_ptr = gpu_crud.kernel_ptr(GPU.Kernel.create_vertex_reference);
        create_vertex_reference_k = new CreateVertexRef_k(GPU.command_queue_ptr, create_vertex_reference_k_ptr)
            .mem_arg(CreateVertexRef_k.Args.vertex_references, GPU.Buffer.vertex_references.memory)
            .mem_arg(CreateVertexRef_k.Args.vertex_weights, GPU.Buffer.vertex_weights.memory)
            .mem_arg(CreateVertexRef_k.Args.uv_tables, GPU.Buffer.uv_tables.memory);

        long create_bone_bind_pose_k_ptr = gpu_crud.kernel_ptr(GPU.Kernel.create_bone_bind_pose);
        create_bone_bind_pose_k = new CreateBoneBindPose_k(GPU.command_queue_ptr, create_bone_bind_pose_k_ptr)
            .mem_arg(CreateBoneBindPose_k.Args.bone_bind_poses, GPU.Buffer.bone_bind_poses.memory)
            .mem_arg(CreateBoneBindPose_k.Args.bone_bind_parents, GPU.Buffer.bone_bind_parents.memory);

        long create_bone_reference_k_ptr = gpu_crud.kernel_ptr(GPU.Kernel.create_bone_reference);
        create_bone_reference_k = new CreateBoneRef_k(GPU.command_queue_ptr, create_bone_reference_k_ptr)
            .mem_arg(CreateBoneRef_k.Args.bone_references, GPU.Buffer.bone_references.memory);

        long create_bone_channel_k_ptr = gpu_crud.kernel_ptr(GPU.Kernel.create_bone_channel);
        create_bone_channel_k = new CreateBoneChannel_k(GPU.command_queue_ptr, create_bone_channel_k_ptr)
            .mem_arg(CreateBoneChannel_k.Args.animation_timing_indices, GPU.Buffer.animation_timing_indices.memory)
            .mem_arg(CreateBoneChannel_k.Args.bone_pos_channel_tables, GPU.Buffer.bone_pos_channel_tables.memory)
            .mem_arg(CreateBoneChannel_k.Args.bone_rot_channel_tables, GPU.Buffer.bone_rot_channel_tables.memory)
            .mem_arg(CreateBoneChannel_k.Args.bone_scl_channel_tables, GPU.Buffer.bone_scl_channel_tables.memory);

        long create_armature_k_ptr = gpu_crud.kernel_ptr(GPU.Kernel.create_armature);
        create_armature_k = new CreateArmature_k(GPU.command_queue_ptr, create_armature_k_ptr)
            .mem_arg(CreateArmature_k.Args.armatures, GPU.Buffer.armatures.memory)
            .mem_arg(CreateArmature_k.Args.armature_flags, GPU.Buffer.armature_flags.memory)
            .mem_arg(CreateArmature_k.Args.hull_tables, GPU.Buffer.armature_hull_table.memory)
            .mem_arg(CreateArmature_k.Args.armature_masses, GPU.Buffer.armature_mass.memory)
            .mem_arg(CreateArmature_k.Args.armature_animation_indices, GPU.Buffer.armature_animation_indices.memory)
            .mem_arg(CreateArmature_k.Args.armature_animation_elapsed, GPU.Buffer.armature_animation_elapsed.memory);

        long create_bone_k_ptr = gpu_crud.kernel_ptr(GPU.Kernel.create_bone);
        create_bone_k = new CreateBone_k(GPU.command_queue_ptr, create_bone_k_ptr)
            .mem_arg(CreateBone_k.Args.bones, GPU.Buffer.bone_instances.memory)
            .mem_arg(CreateBone_k.Args.bone_index_tables, GPU.Buffer.bone_index_tables.memory);

        long create_armature_bone_k_ptr = gpu_crud.kernel_ptr(GPU.Kernel.create_armature_bone);
        create_armature_bone_k = new CreateArmatureBone_k(GPU.command_queue_ptr, create_armature_bone_k_ptr)
            .mem_arg(CreateArmatureBone_k.Args.armature_bones, GPU.Buffer.armatures_bones.memory)
            .mem_arg(CreateArmatureBone_k.Args.bone_bind_tables, GPU.Buffer.bone_bind_tables.memory);

        long create_model_transform_k_ptr = gpu_crud.kernel_ptr(GPU.Kernel.create_model_transform);
        create_model_transform_k = new CreateModelTransform_k(GPU.command_queue_ptr, create_model_transform_k_ptr)
            .mem_arg(CreateModelTransform_k.Args.model_transforms, GPU.Buffer.model_transforms.memory);

        long create_hull_k_ptr = gpu_crud.kernel_ptr(GPU.Kernel.create_hull);
        create_hull_k = new CreateHull_k(GPU.command_queue_ptr, create_hull_k_ptr)
            .mem_arg(CreateHull_k.Args.hulls, GPU.Buffer.hulls.memory)
            .mem_arg(CreateHull_k.Args.hull_rotations, GPU.Buffer.hull_rotation.memory)
            .mem_arg(CreateHull_k.Args.element_tables, GPU.Buffer.hull_element_tables.memory)
            .mem_arg(CreateHull_k.Args.hull_flags, GPU.Buffer.hull_flags.memory)
            .mem_arg(CreateHull_k.Args.hull_mesh_ids, GPU.Buffer.hull_mesh_ids.memory);

        long create_mesh_reference_k_ptr = gpu_crud.kernel_ptr(GPU.Kernel.create_mesh_reference);
        create_mesh_reference_k = new CreateMeshReference_k(GPU.command_queue_ptr, create_mesh_reference_k_ptr)
            .mem_arg(CreateMeshReference_k.Args.mesh_ref_tables, GPU.Buffer.mesh_references.memory);

        long create_mesh_face_k_ptr = gpu_crud.kernel_ptr(GPU.Kernel.create_mesh_face);
        create_mesh_face_k = new CreateMeshFace_k(GPU.command_queue_ptr, create_mesh_face_k_ptr)
            .mem_arg(CreateMeshFace_k.Args.mesh_faces, GPU.Buffer.mesh_faces.memory);

        long create_animation_timings_k_ptr = gpu_crud.kernel_ptr(GPU.Kernel.create_animation_timings);
        create_animation_timings_k = new CreateAnimationTimings_k(GPU.command_queue_ptr, create_animation_timings_k_ptr)
            .mem_arg(CreateAnimationTimings_k.Args.animation_timings, GPU.Buffer.animation_timings.memory);

        long read_position_k_ptr = gpu_crud.kernel_ptr(GPU.Kernel.read_position);
        read_position_k = new ReadPosition_k(GPU.command_queue_ptr, read_position_k_ptr)
            .mem_arg(ReadPosition_k.Args.armatures, GPU.Buffer.armatures.memory);

        long update_accel_k_ptr = gpu_crud.kernel_ptr(GPU.Kernel.update_accel);
        update_accel_k = new UpdateAccel_k(GPU.command_queue_ptr, update_accel_k_ptr)
            .mem_arg(UpdateAccel_k.Args.armature_accel, GPU.Buffer.armature_accel.memory);

        long set_bone_channel_table_k_ptr = gpu_crud.kernel_ptr(GPU.Kernel.set_bone_channel_table);
        set_bone_channel_table_k = new SetBoneChannelTable_k(GPU.command_queue_ptr, set_bone_channel_table_k_ptr)
            .mem_arg(SetBoneChannelTable_k.Args.bone_channel_tables, GPU.Buffer.bone_channel_tables.memory);

        long locate_out_of_bounds_k_ptr = scan_deletes.kernel_ptr(GPU.Kernel.locate_out_of_bounds);
        locate_out_of_bounds_k = new LocateOutOfBounds_k(GPU.command_queue_ptr, locate_out_of_bounds_k_ptr)
            .mem_arg(LocateOutOfBounds_k.Args.hull_tables, GPU.Buffer.armature_hull_table.memory)
            .mem_arg(LocateOutOfBounds_k.Args.hull_flags, GPU.Buffer.hull_flags.memory)
            .mem_arg(LocateOutOfBounds_k.Args.armature_flags, GPU.Buffer.armature_flags.memory);

        long scan_deletes_single_block_out_k_ptr = scan_deletes.kernel_ptr(GPU.Kernel.scan_deletes_single_block_out);
        scan_deletes_single_block_out_k = new ScanDeletesSingleBlockOut_k(GPU.command_queue_ptr, scan_deletes_single_block_out_k_ptr)
            .mem_arg(ScanDeletesSingleBlockOut_k.Args.armature_flags, GPU.Buffer.armature_flags.memory)
            .mem_arg(ScanDeletesSingleBlockOut_k.Args.hull_tables, GPU.Buffer.armature_hull_table.memory)
            .mem_arg(ScanDeletesSingleBlockOut_k.Args.element_tables, GPU.Buffer.hull_element_tables.memory)
            .mem_arg(ScanDeletesSingleBlockOut_k.Args.hull_flags, GPU.Buffer.hull_flags.memory);

        long scan_deletes_multi_block_out_k_ptr = scan_deletes.kernel_ptr(GPU.Kernel.scan_deletes_multi_block_out);
        scan_deletes_multi_block_out_k = new ScanDeletesMultiBlockOut_k(GPU.command_queue_ptr, scan_deletes_multi_block_out_k_ptr)
            .mem_arg(ScanDeletesMultiBlockOut_k.Args.armature_flags, GPU.Buffer.armature_flags.memory)
            .mem_arg(ScanDeletesMultiBlockOut_k.Args.hull_tables, GPU.Buffer.armature_hull_table.memory)
            .mem_arg(ScanDeletesMultiBlockOut_k.Args.element_tables, GPU.Buffer.hull_element_tables.memory)
            .mem_arg(ScanDeletesMultiBlockOut_k.Args.hull_flags, GPU.Buffer.hull_flags.memory);

        long complete_deletes_multi_block_out_k_ptr = scan_deletes.kernel_ptr(GPU.Kernel.complete_deletes_multi_block_out);
        complete_deletes_multi_block_out_k = new CompleteDeletesMultiBlockOut_k(GPU.command_queue_ptr, complete_deletes_multi_block_out_k_ptr)
            .mem_arg(CompleteDeletesMultiBlockOut_k.Args.armature_flags, GPU.Buffer.armature_flags.memory)
            .mem_arg(CompleteDeletesMultiBlockOut_k.Args.hull_tables, GPU.Buffer.armature_hull_table.memory)
            .mem_arg(CompleteDeletesMultiBlockOut_k.Args.element_tables, GPU.Buffer.hull_element_tables.memory)
            .mem_arg(CompleteDeletesMultiBlockOut_k.Args.hull_flags, GPU.Buffer.hull_flags.memory);

        // post-delete buffer compaction

        long compact_armatures_k_ptr = scan_deletes.kernel_ptr(GPU.Kernel.compact_armatures);
        compact_armatures_k = new CompactArmatures_k(GPU.command_queue_ptr, compact_armatures_k_ptr)
            .mem_arg(CompactArmatures_k.Args.armatures, GPU.Buffer.armatures.memory)
            .mem_arg(CompactArmatures_k.Args.armature_accel, GPU.Buffer.armature_accel.memory)
            .mem_arg(CompactArmatures_k.Args.armature_flags, GPU.Buffer.armature_flags.memory)
            .mem_arg(CompactArmatures_k.Args.armature_animation_indices, GPU.Buffer.armature_animation_indices.memory)
            .mem_arg(CompactArmatures_k.Args.armature_animation_elapsed, GPU.Buffer.armature_animation_elapsed.memory)
            .mem_arg(CompactArmatures_k.Args.hull_tables, GPU.Buffer.armature_hull_table.memory)
            .mem_arg(CompactArmatures_k.Args.hulls, GPU.Buffer.hulls.memory)
            .mem_arg(CompactArmatures_k.Args.hull_flags, GPU.Buffer.hull_flags.memory)
            .mem_arg(CompactArmatures_k.Args.element_tables, GPU.Buffer.hull_element_tables.memory)
            .mem_arg(CompactArmatures_k.Args.points, GPU.Buffer.points.memory)
            .mem_arg(CompactArmatures_k.Args.vertex_tables, GPU.Buffer.point_vertex_tables.memory)
            .mem_arg(CompactArmatures_k.Args.bone_tables, GPU.Buffer.point_bone_tables.memory)
            .mem_arg(CompactArmatures_k.Args.bone_bind_tables, GPU.Buffer.bone_bind_tables.memory)
            .mem_arg(CompactArmatures_k.Args.bone_index_tables, GPU.Buffer.bone_index_tables.memory)
            .mem_arg(CompactArmatures_k.Args.edges, GPU.Buffer.edges.memory)
            .mem_arg(CompactArmatures_k.Args.bone_shift, GPU.Buffer.bone_shift.memory)
            .mem_arg(CompactArmatures_k.Args.point_shift, GPU.Buffer.point_shift.memory)
            .mem_arg(CompactArmatures_k.Args.edge_shift, GPU.Buffer.edge_shift.memory)
            .mem_arg(CompactArmatures_k.Args.hull_shift, GPU.Buffer.hull_shift.memory)
            .mem_arg(CompactArmatures_k.Args.bone_bind_shift, GPU.Buffer.bone_bind_shift.memory);

        long compact_hulls_k_ptr = scan_deletes.kernel_ptr(GPU.Kernel.compact_hulls);
        compact_hulls_k = new CompactHulls_k(GPU.command_queue_ptr, compact_hulls_k_ptr)
            .mem_arg(CompactHulls_k.Args.hull_shift, GPU.Buffer.hull_shift.memory)
            .mem_arg(CompactHulls_k.Args.hulls, GPU.Buffer.hulls.memory)
            .mem_arg(CompactHulls_k.Args.hull_mesh_ids, GPU.Buffer.hull_mesh_ids.memory)
            .mem_arg(CompactHulls_k.Args.hull_rotations, GPU.Buffer.hull_rotation.memory)
            .mem_arg(CompactHulls_k.Args.hull_flags, GPU.Buffer.hull_flags.memory)
            .mem_arg(CompactHulls_k.Args.element_tables, GPU.Buffer.hull_element_tables.memory)
            .mem_arg(CompactHulls_k.Args.bounds, GPU.Buffer.aabb.memory)
            .mem_arg(CompactHulls_k.Args.bounds_index_data, GPU.Buffer.aabb_index.memory)
            .mem_arg(CompactHulls_k.Args.bounds_bank_data, GPU.Buffer.aabb_key_table.memory);

        long compact_edges_k_ptr = scan_deletes.kernel_ptr(GPU.Kernel.compact_edges);
        compact_edges_k = new CompactEdges_k(GPU.command_queue_ptr, compact_edges_k_ptr)
            .mem_arg(CompactEdges_k.Args.edge_shift, GPU.Buffer.edge_shift.memory)
            .mem_arg(CompactEdges_k.Args.edges, GPU.Buffer.edges.memory);

        long compact_points_k_ptr = scan_deletes.kernel_ptr(GPU.Kernel.compact_points);
        compact_points_k = new CompactPoints_k(GPU.command_queue_ptr, compact_points_k_ptr)
            .mem_arg(CompactPoints_k.Args.point_shift, GPU.Buffer.point_shift.memory)
            .mem_arg(CompactPoints_k.Args.points, GPU.Buffer.points.memory)
            .mem_arg(CompactPoints_k.Args.anti_gravity, GPU.Buffer.point_anti_gravity.memory)
            .mem_arg(CompactPoints_k.Args.vertex_tables, GPU.Buffer.point_vertex_tables.memory)
            .mem_arg(CompactPoints_k.Args.bone_tables, GPU.Buffer.point_bone_tables.memory);

        long compact_bones_k_ptr = scan_deletes.kernel_ptr(GPU.Kernel.compact_bones);
        compact_bones_k = new CompactBones_k(GPU.command_queue_ptr, compact_bones_k_ptr)
            .mem_arg(CompactBones_k.Args.bone_shift, GPU.Buffer.bone_shift.memory)
            .mem_arg(CompactBones_k.Args.bone_instances, GPU.Buffer.bone_instances.memory)
            .mem_arg(CompactBones_k.Args.bone_index_tables, GPU.Buffer.bone_index_tables.memory);

        long compact_armature_bones_k_ptr = scan_deletes.kernel_ptr(GPU.Kernel.compact_armature_bones);
        compact_armature_bones_k = new CompactArmatureBones_k(GPU.command_queue_ptr, compact_armature_bones_k_ptr)
            .mem_arg(CompactArmatureBones_k.Args.armature_bone_shift, GPU.Buffer.bone_bind_shift.memory)
            .mem_arg(CompactArmatureBones_k.Args.armature_bones, GPU.Buffer.armatures_bones.memory)
            .mem_arg(CompactArmatureBones_k.Args.armature_bone_tables, GPU.Buffer.bone_bind_tables.memory);
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

    public int next_armature_bone()
    {
        return armature_bone_index;
    }


    // creation methods

    public int new_animation_timings(double[] timings)
    {
        create_animation_timings_k
            .set_arg(CreateAnimationTimings_k.Args.target, animation_index)
            .set_arg(CreateAnimationTimings_k.Args.new_animation_timing, timings)
            .call(GPU.global_single_size);

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
            .call(GPU.global_single_size);

        return bone_channel_index++;
    }

    public int new_keyframe(float[] frame, double time)
    {
        create_keyframe_k
            .set_arg(CreateKeyFrame_k.Args.target, keyframe_index)
            .set_arg(CreateKeyFrame_k.Args.new_keyframe, frame)
            .set_arg(CreateKeyFrame_k.Args.new_frame_time, time)
            .call(GPU.global_single_size);

        return keyframe_index++;
    }

    public int new_texture_uv(float u, float v)
    {
        create_texture_uv_k
            .set_arg(CreateTextureUV_k.Args.target, uv_index)
            .set_arg(CreateTextureUV_k.Args.new_texture_uv, arg_float2(u, v))
            .call(GPU.global_single_size);

        return uv_index++;
    }

    public int new_edge(int p1, int p2, float l, int flags)
    {
        create_edge_k
            .set_arg(CreateEdge_k.Args.target, edge_index)
            .set_arg(CreateEdge_k.Args.new_edge, arg_float4(p1, p2, l, flags))
            .call(GPU.global_single_size);

        return edge_index++;
    }

    public int new_point(float[] position, int[] vertex_table, int[] bone_ids)
    {
        var new_point = new float[]{position[0], position[1], position[0], position[1]};
        create_point_k
            .set_arg(CreatePoint_k.Args.target, point_index)
            .set_arg(CreatePoint_k.Args.new_point, new_point)
            .set_arg(CreatePoint_k.Args.new_vertex_table, vertex_table)
            .set_arg(CreatePoint_k.Args.new_bone_table, bone_ids)
            .call(GPU.global_single_size);

        return point_index++;
    }

    public int new_hull(int mesh_id, float[] transform, float[] rotation, int[] table, int[] flags)
    {
        create_hull_k
            .set_arg(CreateHull_k.Args.target, hull_index)
            .set_arg(CreateHull_k.Args.new_hull, transform)
            .set_arg(CreateHull_k.Args.new_rotation, rotation)
            .set_arg(CreateHull_k.Args.new_table, table)
            .set_arg(CreateHull_k.Args.new_flags, flags)
            .set_arg(CreateHull_k.Args.new_hull_mesh_id, mesh_id)
            .call(GPU.global_single_size);

        return hull_index++;
    }

    public int new_mesh_reference(int[] mesh_ref_table)
    {
        create_mesh_reference_k
            .set_arg(CreateMeshReference_k.Args.target, mesh_index)
            .set_arg(CreateMeshReference_k.Args.new_mesh_ref_table, mesh_ref_table)
            .call(GPU.global_single_size);

        return mesh_index++;
    }

    public int new_mesh_face(int[] face)
    {
        create_mesh_face_k
            .set_arg(CreateMeshFace_k.Args.target, face_index)
            .set_arg(CreateMeshFace_k.Args.new_mesh_face, face)
            .call(GPU.global_single_size);

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
            .call(GPU.global_single_size);

        return armature_index++;
    }

    public int new_vertex_reference(float x, float y, float[] weights, int[] uv_table)
    {
        create_vertex_reference_k
            .set_arg(CreateVertexRef_k.Args.target, vertex_ref_index)
            .set_arg(CreateVertexRef_k.Args.new_vertex_reference, arg_float2(x, y))
            .set_arg(CreateVertexRef_k.Args.new_vertex_weights, weights)
            .set_arg(CreateVertexRef_k.Args.new_uv_table, uv_table)
            .call(GPU.global_single_size);

        return vertex_ref_index++;
    }

    public int new_bone_bind_pose(int bind_parent, float[] bone_data)
    {
        create_bone_bind_pose_k
            .set_arg(CreateBoneBindPose_k.Args.target,bone_bind_index)
            .set_arg(CreateBoneBindPose_k.Args.new_bone_bind_pose, bone_data)
            .set_arg(CreateBoneBindPose_k.Args.bone_bind_parent, bind_parent)
            .call(GPU.global_single_size);

        return bone_bind_index++;
    }

    public int new_bone_reference(float[] bone_data)
    {
        create_bone_reference_k
            .set_arg(CreateBoneRef_k.Args.target, bone_ref_index)
            .set_arg(CreateBoneRef_k.Args.new_bone_reference, bone_data)
            .call(GPU.global_single_size);

        return bone_ref_index++;
    }

    public int new_bone(int[] bone_table, float[] bone_data)
    {
        create_bone_k
            .set_arg(CreateBone_k.Args.target, bone_index)
            .set_arg(CreateBone_k.Args.new_bone, bone_data)
            .set_arg(CreateBone_k.Args.new_bone_table, bone_table)
            .call(GPU.global_single_size);

        return bone_index++;
    }

    public int new_armature_bone(int[] bone_bind_table, float[] bone_data)
    {
        create_armature_bone_k
            .set_arg(CreateArmatureBone_k.Args.target, armature_bone_index)
            .set_arg(CreateArmatureBone_k.Args.new_armature_bone, bone_data)
            .set_arg(CreateArmatureBone_k.Args.new_bone_bind_table, bone_bind_table)
            .call(GPU.global_single_size);

        return armature_bone_index++;
    }

    public int new_model_transform(float[] transform_data)
    {
        create_model_transform_k
            .set_arg(CreateModelTransform_k.Args.target, model_transform_index)
            .set_arg(CreateModelTransform_k.Args.new_model_transform, transform_data)
            .call(GPU.global_single_size);

        return model_transform_index++;
    }

    public void set_bone_channel_table(int bone_channel_index, int[] channel_table)
    {
        set_bone_channel_table_k
            .set_arg(SetBoneChannelTable_k.Args.target, bone_channel_index)
            .set_arg(SetBoneChannelTable_k.Args.new_bone_channel_table, channel_table)
            .call(GPU.global_single_size);
    }

    public void update_accel(int armature_index, float acc_x, float acc_y)
    {
        update_accel_k
            .set_arg(UpdateAccel_k.Args.target, armature_index)
            .set_arg(UpdateAccel_k.Args.new_value, arg_float2(acc_x, acc_y))
            .call(GPU.global_single_size);
    }

    public float[] read_position(int armature_index)
    {
        var result_data = GPU.cl_new_pinned_buffer(CLSize.cl_float2);
        GPU.cl_zero_buffer(result_data, CLSize.cl_float2);

        read_position_k
            .ptr_arg(ReadPosition_k.Args.output, result_data)
            .set_arg(ReadPosition_k.Args.target, armature_index)
            .call(GPU.global_single_size);

        float[] result = GPU.cl_read_pinned_float_buffer(result_data, CLSize.cl_float2, 2);
        GPU.release_buffer(result_data);
        return result;
    }

    public void locate_out_of_bounds()
    {
        int armature_count = next_armature();

        int[] counter = new int[]{ 0 };
        var counter_ptr = GPU.cl_new_int_arg_buffer(counter);

        locate_out_of_bounds_k
            .ptr_arg(LocateOutOfBounds_k.Args.counter, counter_ptr)
            .call(arg_long(armature_count));

        GPU.release_buffer(counter_ptr);
    }

    public void delete_and_compact()
    {
        int armature_count = GPU.core_memory.next_armature();
        long output_buf_size = (long) CLSize.cl_int2 * armature_count;
        long output_buf_size2 = (long) CLSize.cl_int4 * armature_count;

        var output_buf_data = GPU.cl_new_buffer(output_buf_size);
        var output_buf_data2 = GPU.cl_new_buffer(output_buf_size2);

        var del_buffer_1 = new GPUMemory(output_buf_data);
        var del_buffer_2 = new GPUMemory(output_buf_data2);

        int[] shift_counts = scan_deletes(del_buffer_1.pointer(), del_buffer_2.pointer(), armature_count);

        if (shift_counts[4] == 0)
        {
            del_buffer_1.release();
            del_buffer_2.release();
            return;
        }

        // shift buffers are cleared before compacting to clean out any data from the last tick
        GPU.Buffer.hull_shift.clear();
        GPU.Buffer.edge_shift.clear();
        GPU.Buffer.point_shift.clear();
        GPU.Buffer.bone_shift.clear();
        GPU.Buffer.bone_bind_shift.clear();

        // as armatures are compacted, the shift buffers for the other components are updated
        compact_armatures_k
            .ptr_arg(CompactArmatures_k.Args.buffer_in, del_buffer_1.pointer())
            .ptr_arg(CompactArmatures_k.Args.buffer_in_2, del_buffer_2.pointer());

        linearize_kernel(compact_armatures_k, armature_count);
        linearize_kernel(compact_bones_k, next_bone());
        linearize_kernel(compact_points_k, next_point());
        linearize_kernel(compact_edges_k, next_edge());
        linearize_kernel(compact_hulls_k, next_hull());
        linearize_kernel(compact_armature_bones_k, next_armature_bone());

        compact_buffers(shift_counts[0], shift_counts[1], shift_counts[2],
            shift_counts[3], shift_counts[4], shift_counts[5]);

        del_buffer_1.release();
        del_buffer_2.release();
    }

    public static void linearize_kernel(GPUKernel kernel, int object_count)
    {
        int offset = 0;
        for (long remaining = object_count; remaining > 0; remaining -= GPU.max_work_group_size)
        {
            int count = (int) Math.min(GPU.max_work_group_size, remaining);
            var sz = count == GPU.max_work_group_size
                ? GPU.local_work_default
                : arg_long(count);
            kernel.call(sz, sz, arg_long(offset));
            offset += count;
        }
    }

    public int[] scan_deletes(long o1_data_ptr, long o2_data_ptr, int n)
    {
        int k = GPU.work_group_count(n);
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
        long local_buffer_size = CLSize.cl_int2 * GPU.max_scan_block_size;
        long local_buffer_size2 = CLSize.cl_int4 * GPU.max_scan_block_size;

        var size_data = GPU.cl_new_pinned_buffer(CLSize.cl_int * 6);
        GPU.cl_zero_buffer(size_data, CLSize.cl_int * 6);

        scan_deletes_single_block_out_k
            .ptr_arg(ScanDeletesSingleBlockOut_k.Args.output, o1_data_ptr)
            .ptr_arg(ScanDeletesSingleBlockOut_k.Args.output2, o2_data_ptr)
            .ptr_arg(ScanDeletesSingleBlockOut_k.Args.sz, size_data)
            .loc_arg(ScanDeletesSingleBlockOut_k.Args.buffer, local_buffer_size)
            .loc_arg(ScanDeletesSingleBlockOut_k.Args.buffer2, local_buffer_size2)
            .set_arg(ScanDeletesSingleBlockOut_k.Args.n, n)
            .call(GPU.local_work_default, GPU.local_work_default);

        int[] sz = GPU.cl_read_pinned_int_buffer(size_data, CLSize.cl_int * 6, 6);
        GPU.release_buffer(size_data);

        return sz;
    }

    private int[] scan_multi_block_deletes_out(long o1_data_ptr, long o2_data_ptr, int n, int k)
    {
        long local_buffer_size = CLSize.cl_int2 * GPU.max_scan_block_size;
        long local_buffer_size2 = CLSize.cl_int4 * GPU.max_scan_block_size;

        long gx = k * GPU.max_scan_block_size;
        long[] global_work_size = arg_long(gx);
        int part_size = k * 2;

        long part_buf_size = ((long) CLSize.cl_int2 * ((long) part_size));
        long part_buf_size2 = ((long) CLSize.cl_int4 * ((long) part_size));

        var p_data = GPU.cl_new_buffer(part_buf_size);
        var p_data2 = GPU.cl_new_buffer(part_buf_size2);

        scan_deletes_multi_block_out_k
            .ptr_arg(ScanDeletesMultiBlockOut_k.Args.output, o1_data_ptr)
            .ptr_arg(ScanDeletesMultiBlockOut_k.Args.output2, o2_data_ptr)
            .loc_arg(ScanDeletesMultiBlockOut_k.Args.buffer, local_buffer_size)
            .loc_arg(ScanDeletesMultiBlockOut_k.Args.buffer2, local_buffer_size2)
            .ptr_arg(ScanDeletesMultiBlockOut_k.Args.part, p_data)
            .ptr_arg(ScanDeletesMultiBlockOut_k.Args.part2, p_data2)
            .set_arg(ScanDeletesMultiBlockOut_k.Args.n, n)
            .call(global_work_size, GPU.local_work_default);

        // note the partial buffers are scanned and updated in-place
        GPU.scan_int2(p_data, part_size);
        GPU.scan_int4(p_data2, part_size);

        var size_data = GPU.cl_new_pinned_buffer(CLSize.cl_int * 6);
        GPU.cl_zero_buffer(size_data, CLSize.cl_int * 6);

        complete_deletes_multi_block_out_k
            .ptr_arg(CompleteDeletesMultiBlockOut_k.Args.output, o1_data_ptr)
            .ptr_arg(CompleteDeletesMultiBlockOut_k.Args.output2, o2_data_ptr)
            .ptr_arg(CompleteDeletesMultiBlockOut_k.Args.sz, size_data)
            .loc_arg(CompleteDeletesMultiBlockOut_k.Args.buffer, local_buffer_size)
            .loc_arg(CompleteDeletesMultiBlockOut_k.Args.buffer2, local_buffer_size2)
            .ptr_arg(CompleteDeletesMultiBlockOut_k.Args.part, p_data)
            .ptr_arg(CompleteDeletesMultiBlockOut_k.Args.part2, p_data2)
            .set_arg(CompleteDeletesMultiBlockOut_k.Args.n, n)
            .call(global_work_size, GPU.local_work_default);

        GPU.release_buffer(p_data);
        GPU.release_buffer(p_data2);

        int[] sz = GPU.cl_read_pinned_int_buffer(size_data, CLSize.cl_int * 6, 6);
        GPU.release_buffer(size_data);

        return sz;
    }

    private void compact_buffers(int edge_shift,
                                 int bone_shift,
                                 int point_shift,
                                 int hull_shift,
                                 int armature_shift,
                                 int armature_bone_shift)
    {
        edge_index          -= (edge_shift);
        bone_index          -= (bone_shift);
        point_index         -= (point_shift);
        hull_index          -= (hull_shift);
        armature_index      -= (armature_shift);
        armature_bone_index -= (armature_bone_shift);
    }
}
