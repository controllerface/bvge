package com.controllerface.bvge.cl;

import com.controllerface.bvge.cl.buffers.*;
import com.controllerface.bvge.cl.kernels.*;
import com.controllerface.bvge.cl.kernels.crud.*;
import com.controllerface.bvge.cl.programs.GPUCrud;
import com.controllerface.bvge.cl.programs.GPUProgram;
import com.controllerface.bvge.cl.programs.ScanDeletes;
import com.controllerface.bvge.game.world.sectors.SectorContainer;
import com.controllerface.bvge.game.world.sectors.*;
import com.controllerface.bvge.geometry.ModelRegistry;
import com.controllerface.bvge.physics.PhysicsEntityBatch;
import com.controllerface.bvge.physics.PhysicsObjects;
import com.controllerface.bvge.util.Constants;
import com.controllerface.bvge.window.Window;

import java.util.concurrent.BrokenBarrierException;
import java.util.concurrent.CyclicBarrier;

import static com.controllerface.bvge.cl.CLSize.*;
import static com.controllerface.bvge.cl.CLUtils.*;
import static com.controllerface.bvge.cl.buffers.BufferType.*;
import static com.controllerface.bvge.geometry.ModelRegistry.*;
import static org.lwjgl.opencl.CL10.clFinish;

public class GPUCoreMemory implements SectorContainer
{
    private static final long ENTITY_INIT   = 10_000L;
    private static final long HULL_INIT     = 10_000L;
    private static final long EDGE_INIT     = 24_000L;
    private static final long POINT_INIT    = 50_000L;
    private static final long DELETE_1_INIT = 10_000L;
    private static final long DELETE_2_INIT = 20_000L;

    private static final int DELETE_COUNTERS = 6;
    private static final int EGRESS_COUNTERS = 8;
    private static final int DELETE_COUNTERS_SIZE = cl_int * DELETE_COUNTERS;
    private static final int EGRESS_COUNTERS_SIZE = cl_int * EGRESS_COUNTERS;

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
    private final GPUKernel k_create_bone_bind_pose;
    private final GPUKernel k_create_bone_channel;
    private final GPUKernel k_create_bone_reference;
    private final GPUKernel k_create_keyframe;
    private final GPUKernel k_create_mesh_face;
    private final GPUKernel k_create_mesh_reference;
    private final GPUKernel k_create_model_transform;
    private final GPUKernel k_create_texture_uv;
    private final GPUKernel k_create_vertex_reference;
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

    private final long ptr_delete_counter;
    private final long ptr_position_buffer;
    private final long ptr_delete_sizes;
    private final long ptr_egress_sizes;

    private int vertex_ref_index      = 0;
    private int bone_bind_index       = 0;
    private int bone_ref_index        = 0;
    private int model_transform_index = 0;
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

    private final int[] next_egress_counts = new int[8];
    private final int[] last_egress_counts = new int[8];
    private final OrderedSectorInput sector_ingress_buffer;
    private final DoubleBuffer<UnorderedSectorOutput> sector_egress_buffer;
    private final DoubleBuffer<BrokenObjectBuffer> broken_egress_buffer;
    private final DoubleBuffer<CollectedObjectBuffer> object_egress_buffer;
    private final SectorGroup sector_group;
    private final SectorInput sector_input;
    private final MirrorGroup mirror_group;
    private final ReferenceGroup reference_group;

    /**
     * This barrier is used to facilitate co-operation between the sector loading thread and the main loop.
     * Each iteration, the sector loader waits on this barrier once it is done loading sectors, and then the
     * main loop does the same, tripping the barrier, which it then immediately resets.
     */
    private final CyclicBarrier world_barrier = new CyclicBarrier(4);

    private final GPUScanVectorInt2 gpu_int2_scan;
    private final GPUScanVectorInt4 gpu_int4_scan;

    public GPUCoreMemory()
    {
        ptr_delete_counter  = GPGPU.cl_new_int_arg_buffer(new int[]{ 0 });
        ptr_position_buffer = GPGPU.cl_new_pinned_buffer(cl_float2);
        ptr_delete_sizes    = GPGPU.cl_new_pinned_buffer(DELETE_COUNTERS_SIZE);
        ptr_egress_sizes    = GPGPU.cl_new_pinned_buffer(EGRESS_COUNTERS_SIZE);

        gpu_int2_scan = new GPUScanVectorInt2(GPGPU.ptr_compute_queue);
        gpu_int4_scan = new GPUScanVectorInt4(GPGPU.ptr_compute_queue);

        // transients
        b_hull_shift                 = new TransientBuffer(GPGPU.ptr_compute_queue, cl_int, HULL_INIT);
        b_edge_shift                 = new TransientBuffer(GPGPU.ptr_compute_queue, cl_int, EDGE_INIT);
        b_point_shift                = new TransientBuffer(GPGPU.ptr_compute_queue, cl_int, POINT_INIT);
        b_hull_bone_shift            = new TransientBuffer(GPGPU.ptr_compute_queue, cl_int, HULL_INIT);
        b_armature_bone_shift        = new TransientBuffer(GPGPU.ptr_compute_queue, cl_int, ENTITY_INIT);
        b_delete_1                   = new TransientBuffer(GPGPU.ptr_compute_queue, cl_int2, DELETE_1_INIT);
        b_delete_2                   = new TransientBuffer(GPGPU.ptr_compute_queue, cl_int4, DELETE_2_INIT);
        b_delete_partial_1           = new TransientBuffer(GPGPU.ptr_compute_queue, cl_int2, DELETE_1_INIT);
        b_delete_partial_2           = new TransientBuffer(GPGPU.ptr_compute_queue, cl_int4, DELETE_2_INIT);

        p_gpu_crud.init();
        p_scan_deletes.init();

        this.sector_group = new SectorGroup("Live Sectors", GPGPU.ptr_compute_queue, ENTITY_INIT, HULL_INIT, EDGE_INIT, POINT_INIT);
        this.sector_input = new SectorInput(GPGPU.ptr_compute_queue, this.p_gpu_crud, this.sector_group);
        this.mirror_group = new MirrorGroup("Render Mirror", GPGPU.ptr_compute_queue, ENTITY_INIT, HULL_INIT, EDGE_INIT, POINT_INIT);
        this.reference_group = new ReferenceGroup("Reference Data", GPGPU.ptr_compute_queue);


        long k_ptr_create_texture_uv = p_gpu_crud.kernel_ptr(Kernel.create_texture_uv);
        k_create_texture_uv = new CreateTextureUV_k(GPGPU.ptr_compute_queue, k_ptr_create_texture_uv)
            .buf_arg(CreateTextureUV_k.Args.texture_uvs, reference_group.get_buffer(VERTEX_TEXTURE_UV));

        long k_ptr_create_keyframe = p_gpu_crud.kernel_ptr(Kernel.create_keyframe);
        k_create_keyframe = new CreateKeyFrame_k(GPGPU.ptr_compute_queue, k_ptr_create_keyframe)
            .buf_arg(CreateKeyFrame_k.Args.key_frames, reference_group.get_buffer(ANIM_KEY_FRAME))
            .buf_arg(CreateKeyFrame_k.Args.frame_times, reference_group.get_buffer(ANIM_FRAME_TIME));

        long k_ptr_create_vertex_reference = p_gpu_crud.kernel_ptr(Kernel.create_vertex_reference);
        k_create_vertex_reference = new CreateVertexRef_k(GPGPU.ptr_compute_queue, k_ptr_create_vertex_reference)
            .buf_arg(CreateVertexRef_k.Args.vertex_references, reference_group.get_buffer(VERTEX_REFERENCE))
            .buf_arg(CreateVertexRef_k.Args.vertex_weights, reference_group.get_buffer(VERTEX_WEIGHT))
            .buf_arg(CreateVertexRef_k.Args.uv_tables, reference_group.get_buffer(VERTEX_UV_TABLE));

        long k_ptr_create_bone_bind_pose = p_gpu_crud.kernel_ptr(Kernel.create_bone_bind_pose);
        k_create_bone_bind_pose = new CreateBoneBindPose_k(GPGPU.ptr_compute_queue, k_ptr_create_bone_bind_pose)
            .buf_arg(CreateBoneBindPose_k.Args.bone_bind_poses, reference_group.get_buffer(BONE_BIND_POSE));

        long k_ptr_create_bone_reference = p_gpu_crud.kernel_ptr(Kernel.create_bone_reference);
        k_create_bone_reference = new CreateBoneRef_k(GPGPU.ptr_compute_queue, k_ptr_create_bone_reference)
            .buf_arg(CreateBoneRef_k.Args.bone_references, reference_group.get_buffer(BONE_REFERENCE));

        long k_ptr_create_bone_channel = p_gpu_crud.kernel_ptr(Kernel.create_bone_channel);
        k_create_bone_channel = new CreateBoneChannel_k(GPGPU.ptr_compute_queue, k_ptr_create_bone_channel)
            .buf_arg(CreateBoneChannel_k.Args.animation_timing_indices, reference_group.get_buffer(ANIM_TIMING_INDEX))
            .buf_arg(CreateBoneChannel_k.Args.bone_pos_channel_tables, reference_group.get_buffer(ANIM_POS_CHANNEL))
            .buf_arg(CreateBoneChannel_k.Args.bone_rot_channel_tables, reference_group.get_buffer(ANIM_ROT_CHANNEL))
            .buf_arg(CreateBoneChannel_k.Args.bone_scl_channel_tables, reference_group.get_buffer(ANIM_SCL_CHANNEL));

        long k_ptr_create_model_transform = p_gpu_crud.kernel_ptr(Kernel.create_model_transform);
        k_create_model_transform = new CreateModelTransform_k(GPGPU.ptr_compute_queue, k_ptr_create_model_transform)
            .buf_arg(CreateModelTransform_k.Args.model_transforms, reference_group.get_buffer(MODEL_TRANSFORM));

        long k_ptr_create_mesh_reference = p_gpu_crud.kernel_ptr(Kernel.create_mesh_reference);
        k_create_mesh_reference = new CreateMeshReference_k(GPGPU.ptr_compute_queue, k_ptr_create_mesh_reference)
            .buf_arg(CreateMeshReference_k.Args.mesh_vertex_tables, reference_group.get_buffer(MESH_VERTEX_TABLE))
            .buf_arg(CreateMeshReference_k.Args.mesh_face_tables, reference_group.get_buffer(MESH_FACE_TABLE));

        long k_ptr_create_mesh_face = p_gpu_crud.kernel_ptr(Kernel.create_mesh_face);
        k_create_mesh_face = new CreateMeshFace_k(GPGPU.ptr_compute_queue, k_ptr_create_mesh_face)
            .buf_arg(CreateMeshFace_k.Args.mesh_faces, reference_group.get_buffer(MESH_FACE));

        long k_ptr_create_animation_timings = p_gpu_crud.kernel_ptr(Kernel.create_animation_timings);
        k_create_animation_timings = new CreateAnimationTimings_k(GPGPU.ptr_compute_queue, k_ptr_create_animation_timings)
            .buf_arg(CreateAnimationTimings_k.Args.animation_durations, reference_group.get_buffer(ANIM_DURATION))
            .buf_arg(CreateAnimationTimings_k.Args.animation_tick_rates, reference_group.get_buffer(ANIM_TICK_RATE));

        // read methods

        long k_ptr_read_position = p_gpu_crud.kernel_ptr(Kernel.read_position);
        k_read_position = new ReadPosition_k(GPGPU.ptr_compute_queue, k_ptr_read_position)
            .buf_arg(ReadPosition_k.Args.entities, sector_group.get_buffer(ENTITY));

        // update methods

        long k_ptr_update_accel = p_gpu_crud.kernel_ptr(Kernel.update_accel);
        k_update_accel = new UpdateAccel_k(GPGPU.ptr_compute_queue, k_ptr_update_accel)
            .buf_arg(UpdateAccel_k.Args.entity_accel, sector_group.get_buffer(ENTITY_ACCEL));

        long k_ptr_update_mouse_position = p_gpu_crud.kernel_ptr(Kernel.update_mouse_position);
        k_update_mouse_position = new UpdateMousePosition_k(GPGPU.ptr_compute_queue, k_ptr_update_mouse_position)
            .buf_arg(UpdateMousePosition_k.Args.entity_root_hulls, sector_group.get_buffer(ENTITY_ROOT_HULL))
            .buf_arg(UpdateMousePosition_k.Args.hull_point_tables, sector_group.get_buffer(HULL_POINT_TABLE))
            .buf_arg(UpdateMousePosition_k.Args.points, sector_group.get_buffer(POINT));

        long k_ptr_set_bone_channel_table = p_gpu_crud.kernel_ptr(Kernel.set_bone_channel_table);
        k_set_bone_channel_table = new SetBoneChannelTable_k(GPGPU.ptr_compute_queue, k_ptr_set_bone_channel_table)
            .buf_arg(SetBoneChannelTable_k.Args.bone_channel_tables, reference_group.get_buffer(BONE_ANIM_CHANNEL_TABLE));

        // delete methods

        long k_ptr_scan_deletes_single_block_out = p_scan_deletes.kernel_ptr(Kernel.scan_deletes_single_block_out);
        k_scan_deletes_single_block_out = new ScanDeletesSingleBlockOut_k(GPGPU.ptr_compute_queue, k_ptr_scan_deletes_single_block_out)
            .ptr_arg(ScanDeletesSingleBlockOut_k.Args.sz, ptr_delete_sizes)
            .buf_arg(ScanDeletesSingleBlockOut_k.Args.entity_flags, sector_group.get_buffer(ENTITY_FLAG))
            .buf_arg(ScanDeletesSingleBlockOut_k.Args.hull_tables, sector_group.get_buffer(ENTITY_HULL_TABLE))
            .buf_arg(ScanDeletesSingleBlockOut_k.Args.bone_tables, sector_group.get_buffer(ENTITY_BONE_TABLE))
            .buf_arg(ScanDeletesSingleBlockOut_k.Args.point_tables, sector_group.get_buffer(HULL_POINT_TABLE))
            .buf_arg(ScanDeletesSingleBlockOut_k.Args.edge_tables, sector_group.get_buffer(HULL_EDGE_TABLE))
            .buf_arg(ScanDeletesSingleBlockOut_k.Args.hull_bone_tables, sector_group.get_buffer(HULL_BONE_TABLE));

        long k_ptr_scan_deletes_multi_block_out = p_scan_deletes.kernel_ptr(Kernel.scan_deletes_multi_block_out);
        k_scan_deletes_multi_block_out = new ScanDeletesMultiBlockOut_k(GPGPU.ptr_compute_queue, k_ptr_scan_deletes_multi_block_out)
            .buf_arg(ScanDeletesMultiBlockOut_k.Args.part1, b_delete_partial_1)
            .buf_arg(ScanDeletesMultiBlockOut_k.Args.part2, b_delete_partial_2)
            .buf_arg(ScanDeletesMultiBlockOut_k.Args.entity_flags, sector_group.get_buffer(ENTITY_FLAG))
            .buf_arg(ScanDeletesMultiBlockOut_k.Args.hull_tables, sector_group.get_buffer(ENTITY_HULL_TABLE))
            .buf_arg(ScanDeletesMultiBlockOut_k.Args.bone_tables, sector_group.get_buffer(ENTITY_BONE_TABLE))
            .buf_arg(ScanDeletesMultiBlockOut_k.Args.point_tables, sector_group.get_buffer(HULL_POINT_TABLE))
            .buf_arg(ScanDeletesMultiBlockOut_k.Args.edge_tables, sector_group.get_buffer(HULL_EDGE_TABLE))
            .buf_arg(ScanDeletesMultiBlockOut_k.Args.hull_bone_tables, sector_group.get_buffer(HULL_BONE_TABLE));

        long k_ptr_complete_deletes_multi_block_out = p_scan_deletes.kernel_ptr(Kernel.complete_deletes_multi_block_out);
        k_complete_deletes_multi_block_out = new CompleteDeletesMultiBlockOut_k(GPGPU.ptr_compute_queue, k_ptr_complete_deletes_multi_block_out)
            .ptr_arg(CompleteDeletesMultiBlockOut_k.Args.sz, ptr_delete_sizes)
            .buf_arg(CompleteDeletesMultiBlockOut_k.Args.part1, b_delete_partial_1)
            .buf_arg(CompleteDeletesMultiBlockOut_k.Args.part2, b_delete_partial_2)
            .buf_arg(CompleteDeletesMultiBlockOut_k.Args.entity_flags, sector_group.get_buffer(ENTITY_FLAG))
            .buf_arg(CompleteDeletesMultiBlockOut_k.Args.hull_tables, sector_group.get_buffer(ENTITY_HULL_TABLE))
            .buf_arg(CompleteDeletesMultiBlockOut_k.Args.bone_tables, sector_group.get_buffer(ENTITY_BONE_TABLE))
            .buf_arg(CompleteDeletesMultiBlockOut_k.Args.point_tables, sector_group.get_buffer(HULL_POINT_TABLE))
            .buf_arg(CompleteDeletesMultiBlockOut_k.Args.edge_tables, sector_group.get_buffer(HULL_EDGE_TABLE))
            .buf_arg(CompleteDeletesMultiBlockOut_k.Args.hull_bone_tables, sector_group.get_buffer(HULL_BONE_TABLE));

        long k_ptr_compact_entities = p_scan_deletes.kernel_ptr(Kernel.compact_entities);
        k_compact_entities = new CompactEntities_k(GPGPU.ptr_compute_queue, k_ptr_compact_entities)
            .buf_arg(CompactEntities_k.Args.entities, sector_group.get_buffer(ENTITY))
            .buf_arg(CompactEntities_k.Args.entity_masses, sector_group.get_buffer(ENTITY_MASS))
            .buf_arg(CompactEntities_k.Args.entity_root_hulls, sector_group.get_buffer(ENTITY_ROOT_HULL))
            .buf_arg(CompactEntities_k.Args.entity_model_indices, sector_group.get_buffer(ENTITY_MODEL_ID))
            .buf_arg(CompactEntities_k.Args.entity_model_transforms, sector_group.get_buffer(ENTITY_TRANSFORM_ID))
            .buf_arg(CompactEntities_k.Args.entity_types, sector_group.get_buffer(ENTITY_TYPE))
            .buf_arg(CompactEntities_k.Args.entity_flags, sector_group.get_buffer(ENTITY_FLAG))
            .buf_arg(CompactEntities_k.Args.entity_animation_indices, sector_group.get_buffer(ENTITY_ANIM_INDEX))
            .buf_arg(CompactEntities_k.Args.entity_animation_elapsed, sector_group.get_buffer(ENTITY_ANIM_ELAPSED))
            .buf_arg(CompactEntities_k.Args.entity_animation_blend, sector_group.get_buffer(ENTITY_ANIM_BLEND))
            .buf_arg(CompactEntities_k.Args.entity_motion_states, sector_group.get_buffer(ENTITY_MOTION_STATE))
            .buf_arg(CompactEntities_k.Args.entity_entity_hull_tables, sector_group.get_buffer(ENTITY_HULL_TABLE))
            .buf_arg(CompactEntities_k.Args.entity_bone_tables, sector_group.get_buffer(ENTITY_BONE_TABLE))
            .buf_arg(CompactEntities_k.Args.hull_bone_tables, sector_group.get_buffer(HULL_BONE_TABLE))
            .buf_arg(CompactEntities_k.Args.hull_entity_ids, sector_group.get_buffer(HULL_ENTITY_ID))
            .buf_arg(CompactEntities_k.Args.hull_point_tables, sector_group.get_buffer(HULL_POINT_TABLE))
            .buf_arg(CompactEntities_k.Args.hull_edge_tables, sector_group.get_buffer(HULL_EDGE_TABLE))
            .buf_arg(CompactEntities_k.Args.points, sector_group.get_buffer(POINT))
            .buf_arg(CompactEntities_k.Args.point_hull_indices, sector_group.get_buffer(POINT_HULL_INDEX))
            .buf_arg(CompactEntities_k.Args.point_bone_tables, sector_group.get_buffer(POINT_BONE_TABLE))
            .buf_arg(CompactEntities_k.Args.entity_bone_parent_ids, sector_group.get_buffer(ENTITY_BONE_PARENT_ID))
            .buf_arg(CompactEntities_k.Args.hull_bind_pose_indices, sector_group.get_buffer(HULL_BONE_BIND_POSE))
            .buf_arg(CompactEntities_k.Args.edges, sector_group.get_buffer(EDGE))
            .buf_arg(CompactEntities_k.Args.hull_bone_shift, b_hull_bone_shift)
            .buf_arg(CompactEntities_k.Args.point_shift, b_point_shift)
            .buf_arg(CompactEntities_k.Args.edge_shift, b_edge_shift)
            .buf_arg(CompactEntities_k.Args.hull_shift, b_hull_shift)
            .buf_arg(CompactEntities_k.Args.entity_bone_shift, b_armature_bone_shift);

        long k_ptr_compact_hulls = p_scan_deletes.kernel_ptr(Kernel.compact_hulls);
        k_compact_hulls = new CompactHulls_k(GPGPU.ptr_compute_queue, k_ptr_compact_hulls)
            .buf_arg(CompactHulls_k.Args.hull_shift, b_hull_shift)
            .buf_arg(CompactHulls_k.Args.hulls, sector_group.get_buffer(HULL))
            .buf_arg(CompactHulls_k.Args.hull_scales, sector_group.get_buffer(HULL_SCALE))
            .buf_arg(CompactHulls_k.Args.hull_mesh_ids, sector_group.get_buffer(HULL_MESH_ID))
            .buf_arg(CompactHulls_k.Args.hull_uv_offsets, sector_group.get_buffer(HULL_UV_OFFSET))
            .buf_arg(CompactHulls_k.Args.hull_rotations, sector_group.get_buffer(HULL_ROTATION))
            .buf_arg(CompactHulls_k.Args.hull_frictions, sector_group.get_buffer(HULL_FRICTION))
            .buf_arg(CompactHulls_k.Args.hull_restitutions, sector_group.get_buffer(HULL_RESTITUTION))
            .buf_arg(CompactHulls_k.Args.hull_integrity, sector_group.get_buffer(HULL_INTEGRITY))
            .buf_arg(CompactHulls_k.Args.hull_bone_tables, sector_group.get_buffer(HULL_BONE_TABLE))
            .buf_arg(CompactHulls_k.Args.hull_entity_ids, sector_group.get_buffer(HULL_ENTITY_ID))
            .buf_arg(CompactHulls_k.Args.hull_flags, sector_group.get_buffer(HULL_FLAG))
            .buf_arg(CompactHulls_k.Args.hull_point_tables, sector_group.get_buffer(HULL_POINT_TABLE))
            .buf_arg(CompactHulls_k.Args.hull_edge_tables, sector_group.get_buffer(HULL_EDGE_TABLE))
            .buf_arg(CompactHulls_k.Args.bounds, sector_group.get_buffer(HULL_AABB))
            .buf_arg(CompactHulls_k.Args.bounds_index_data, sector_group.get_buffer(HULL_AABB_INDEX))
            .buf_arg(CompactHulls_k.Args.bounds_bank_data, sector_group.get_buffer(HULL_AABB_KEY_TABLE));

        long k_ptr_compact_edges = p_scan_deletes.kernel_ptr(Kernel.compact_edges);
        k_compact_edges = new CompactEdges_k(GPGPU.ptr_compute_queue, k_ptr_compact_edges)
            .buf_arg(CompactEdges_k.Args.edge_shift, b_edge_shift)
            .buf_arg(CompactEdges_k.Args.edges, sector_group.get_buffer(EDGE))
            .buf_arg(CompactEdges_k.Args.edge_lengths, sector_group.get_buffer(EDGE_LENGTH))
            .buf_arg(CompactEdges_k.Args.edge_flags, sector_group.get_buffer(EDGE_FLAG));

        long k_ptr_compact_points = p_scan_deletes.kernel_ptr(Kernel.compact_points);
        k_compact_points = new CompactPoints_k(GPGPU.ptr_compute_queue, k_ptr_compact_points)
            .buf_arg(CompactPoints_k.Args.point_shift, b_point_shift)
            .buf_arg(CompactPoints_k.Args.points, sector_group.get_buffer(POINT))
            .buf_arg(CompactPoints_k.Args.anti_gravity, sector_group.get_buffer(POINT_ANTI_GRAV))
            .buf_arg(CompactPoints_k.Args.point_vertex_references, sector_group.get_buffer(POINT_VERTEX_REFERENCE))
            .buf_arg(CompactPoints_k.Args.point_hull_indices, sector_group.get_buffer(POINT_HULL_INDEX))
            .buf_arg(CompactPoints_k.Args.point_flags, sector_group.get_buffer(POINT_FLAG))
            .buf_arg(CompactPoints_k.Args.point_hit_counts, sector_group.get_buffer(POINT_HIT_COUNT))
            .buf_arg(CompactPoints_k.Args.bone_tables, sector_group.get_buffer(POINT_BONE_TABLE));

        long k_ptr_compact_hull_bones = p_scan_deletes.kernel_ptr(Kernel.compact_hull_bones);
        k_compact_hull_bones = new CompactHullBones_k(GPGPU.ptr_compute_queue, k_ptr_compact_hull_bones)
            .buf_arg(CompactHullBones_k.Args.hull_bone_shift, b_hull_bone_shift)
            .buf_arg(CompactHullBones_k.Args.bone_instances, sector_group.get_buffer(HULL_BONE))
            .buf_arg(CompactHullBones_k.Args.hull_bind_pose_indicies, sector_group.get_buffer(HULL_BONE_BIND_POSE))
            .buf_arg(CompactHullBones_k.Args.hull_inv_bind_pose_indicies, sector_group.get_buffer(HULL_BONE_INV_BIND_POSE));

        long k_ptr_compact_armature_bones = p_scan_deletes.kernel_ptr(Kernel.compact_armature_bones);
        k_compact_armature_bones = new CompactArmatureBones_k(GPGPU.ptr_compute_queue, k_ptr_compact_armature_bones)
            .buf_arg(CompactArmatureBones_k.Args.armature_bone_shift, b_armature_bone_shift)
            .buf_arg(CompactArmatureBones_k.Args.armature_bones, sector_group.get_buffer(ENTITY_BONE))
            .buf_arg(CompactArmatureBones_k.Args.armature_bone_reference_ids, sector_group.get_buffer(ENTITY_BONE_REFERENCE_ID))
            .buf_arg(CompactArmatureBones_k.Args.armature_bone_parent_ids, sector_group.get_buffer(ENTITY_BONE_PARENT_ID));

        long k_ptr_count_egress_candidates = p_gpu_crud.kernel_ptr(Kernel.count_egress_entities);
        k_count_egress_entities = new CountEgressEntities_k(GPGPU.ptr_compute_queue, k_ptr_count_egress_candidates)
            .buf_arg(CountEgressEntities_k.Args.entity_flags, sector_group.get_buffer(ENTITY_FLAG))
            .buf_arg(CountEgressEntities_k.Args.entity_hull_tables, sector_group.get_buffer(ENTITY_HULL_TABLE))
            .buf_arg(CountEgressEntities_k.Args.entity_bone_tables, sector_group.get_buffer(ENTITY_BONE_TABLE))
            .buf_arg(CountEgressEntities_k.Args.hull_flags, sector_group.get_buffer(HULL_FLAG))
            .buf_arg(CountEgressEntities_k.Args.hull_point_tables, sector_group.get_buffer(HULL_POINT_TABLE))
            .buf_arg(CountEgressEntities_k.Args.hull_edge_tables, sector_group.get_buffer(HULL_EDGE_TABLE))
            .buf_arg(CountEgressEntities_k.Args.hull_bone_tables, sector_group.get_buffer(HULL_BONE_TABLE))
            .ptr_arg(CountEgressEntities_k.Args.counters, ptr_egress_sizes);

        var sector_egress_a = new UnorderedSectorOutput("Sector Egress A",GPGPU.ptr_sector_queue, this);
        var sector_egress_b = new UnorderedSectorOutput("Sector Egress B", GPGPU.ptr_sector_queue, this);
        var broken_egress_a = new BrokenObjectBuffer("Broken Egress A", GPGPU.ptr_sector_queue, this);
        var broken_egress_b = new BrokenObjectBuffer("Broken Egress B", GPGPU.ptr_sector_queue, this);
        var object_egress_a = new CollectedObjectBuffer("Object Egress A", GPGPU.ptr_sector_queue, this);
        var object_egress_b = new CollectedObjectBuffer("Object Egress B", GPGPU.ptr_sector_queue, this);

        this.sector_ingress_buffer = new OrderedSectorInput(GPGPU.ptr_sector_queue, this);
        this.sector_egress_buffer  = new DoubleBuffer<>(sector_egress_a, sector_egress_b);
        this.broken_egress_buffer  = new DoubleBuffer<>(broken_egress_a, broken_egress_b);
        this.object_egress_buffer  = new DoubleBuffer<>(object_egress_a, object_egress_b);
    }

    public ResizableBuffer buffer(BufferType bufferType)
    {
        return switch (bufferType)
        {
            case BROKEN_POSITIONS,
                 BROKEN_UV_OFFSETS,
                 BROKEN_MODEL_IDS,
                 COLLECTED_UV_OFFSETS,
                 COLLECTED_FLAG,
                 COLLECTED_TYPE -> null;

            case ANIM_POS_CHANNEL,
                 ANIM_ROT_CHANNEL,
                 ANIM_SCL_CHANNEL,
                 VERTEX_REFERENCE,
                 VERTEX_UV_TABLE,
                 VERTEX_WEIGHT,
                 VERTEX_TEXTURE_UV,
                 MODEL_TRANSFORM,
                 ANIM_TIMING_INDEX,
                 MESH_VERTEX_TABLE,
                 MESH_FACE_TABLE,
                 MESH_FACE,
                 BONE_REFERENCE,
                 BONE_ANIM_CHANNEL_TABLE,
                 BONE_BIND_POSE,
                 ANIM_DURATION,
                 ANIM_TICK_RATE,
                 ANIM_KEY_FRAME,
                 ANIM_FRAME_TIME -> reference_group.get_buffer(bufferType);

            case MIRROR_POINT,
                 MIRROR_POINT_ANTI_GRAV,
                 MIRROR_POINT_HIT_COUNT,
                 MIRROR_POINT_VERTEX_REFERENCE,
                 MIRROR_EDGE,
                 MIRROR_EDGE_FLAG,
                 MIRROR_HULL,
                 MIRROR_HULL_AABB,
                 MIRROR_HULL_ENTITY_ID,
                 MIRROR_HULL_FLAG,
                 MIRROR_HULL_MESH_ID,
                 MIRROR_HULL_UV_OFFSET,
                 MIRROR_HULL_INTEGRITY,
                 MIRROR_HULL_POINT_TABLE,
                 MIRROR_HULL_ROTATION,
                 MIRROR_HULL_SCALE,
                 MIRROR_ENTITY,
                 MIRROR_ENTITY_FLAG,
                 MIRROR_ENTITY_MODEL_ID,
                 MIRROR_ENTITY_ROOT_HULL -> mirror_group.get_buffer(bufferType);

            case POINT,
                 POINT_HIT_COUNT,
                 POINT_FLAG,
                 POINT_HULL_INDEX,
                 POINT_VERTEX_REFERENCE,
                 POINT_BONE_TABLE,
                 POINT_ANTI_GRAV,
                 HULL,
                 HULL_ROTATION,
                 HULL_UV_OFFSET,
                 HULL_MESH_ID,
                 HULL_RESTITUTION,
                 HULL_INTEGRITY,
                 HULL_FRICTION,
                 HULL_FLAG,
                 HULL_EDGE_TABLE,
                 HULL_POINT_TABLE,
                 HULL_BONE_INV_BIND_POSE,
                 HULL_BONE_BIND_POSE,
                 HULL_BONE_TABLE,
                 HULL_ENTITY_ID,
                 HULL_BONE,
                 HULL_SCALE,
                 HULL_AABB,
                 HULL_AABB_INDEX,
                 HULL_AABB_KEY_TABLE,
                 EDGE,
                 EDGE_LENGTH,
                 EDGE_FLAG,
                 ENTITY,
                 ENTITY_TRANSFORM_ID,
                 ENTITY_ROOT_HULL,
                 ENTITY_MODEL_ID,
                 ENTITY_MASS,
                 ENTITY_HULL_TABLE,
                 ENTITY_BONE_TABLE,
                 ENTITY_TYPE,
                 ENTITY_FLAG,
                 ENTITY_BONE_PARENT_ID,
                 ENTITY_BONE_REFERENCE_ID,
                 ENTITY_BONE,
                 ENTITY_ANIM_INDEX,
                 ENTITY_MOTION_STATE,
                 ENTITY_ACCEL,
                 ENTITY_ANIM_BLEND,
                 ENTITY_ANIM_ELAPSED -> sector_group.get_buffer(bufferType);
        };
    }

    public void mirror_render_buffers()
    {
        mirror_group.mirror(sector_group);

        last_edge_index     = sector_input.edge_index();
        last_entity_index   = sector_input.entity_index();
        last_hull_index     = sector_input.hull_index();
        last_point_index    = sector_input.point_index();
    }

    // index methods

    public int next_mesh()
    {
        return mesh_index;
    }

    @Override
    public int next_entity()
    {
        return sector_input.entity_index();
    }

    @Override
    public int next_hull()
    {
        return sector_input.hull_index();
    }

    @Override
    public int next_point()
    {
        return sector_input.point_index();
    }

    @Override
    public int next_edge()
    {
        return sector_input.edge_index();
    }

    @Override
    public int next_hull_bone()
    {
        return sector_input.hull_bone_index();
    }

    @Override
    public int next_armature_bone()
    {
        return sector_input.entity_bone_index();
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

    public void flip_egress_buffers()
    {
        last_egress_counts[0] = next_egress_counts[0];
        last_egress_counts[1] = next_egress_counts[1];
        last_egress_counts[2] = next_egress_counts[2];
        last_egress_counts[3] = next_egress_counts[3];
        last_egress_counts[4] = next_egress_counts[4];
        last_egress_counts[5] = next_egress_counts[5];
        last_egress_counts[6] = next_egress_counts[6];
        last_egress_counts[7] = next_egress_counts[7];

        sector_egress_buffer.flip();
        broken_egress_buffer.flip();
        object_egress_buffer.flip();
    }

    public void release_world_barrier()
    {
        world_barrier.reset();
    }

    public void await_world_barrier()
    {
        if (world_barrier.isBroken()) return;
        try { world_barrier.await(); }
        catch (InterruptedException _) { }
        catch (BrokenBarrierException e)
        {
            if (!Window.get().is_closing()) throw new RuntimeException(e);
        }
    }

    public void load_entity_batch(PhysicsEntityBatch batch)
    {
        for (var entity : batch.entities)
        {
            PhysicsObjects.load_entity(sector_ingress_buffer, entity);
        }
        for (var block : batch.blocks)
        {
            if (block.dynamic())
            {
                PhysicsObjects.base_block(sector_ingress_buffer, block.x(), block.y(), block.size(), block.mass(), block.friction(), block.restitution(), block.flags(), block.material(), block.hits());
            }
            else
            {
                int flags = block.flags() | Constants.HullFlags.IS_STATIC._int;
                PhysicsObjects.base_block(sector_ingress_buffer, block.x(), block.y(), block.size(), block.mass(), block.friction(), block.restitution(), flags, block.material(), block.hits());
            }
        }
        for (var shard : batch.shards)
        {
            int id = shard.spike()
                ? ModelRegistry.BASE_SPIKE_INDEX
                : shard.flip()
                    ? L_SHARD_INDEX
                    : R_SHARD_INDEX;

            int shard_flags = shard.flags();

            PhysicsObjects.tri(sector_ingress_buffer, shard.x(), shard.y(), shard.size(), shard_flags, shard.mass(), shard.friction(), shard.restitution(), id, shard.material());
        }
        for (var liquid : batch.liquids)
        {
            PhysicsObjects.liquid_particle(sector_ingress_buffer, liquid.x(), liquid.y(), liquid.size(), liquid.mass(), liquid.friction(), liquid.restitution(), liquid.flags(), liquid.point_flags(), liquid.particle_fluid());
        }
    }

    public int[] last_egress_counts()
    {
        return last_egress_counts;
    }

    public void egress(int[] egress_counts)
    {
        next_egress_counts[0]  = egress_counts[0];
        next_egress_counts[1]  = egress_counts[1];
        next_egress_counts[2]  = egress_counts[2];
        next_egress_counts[3]  = egress_counts[3];
        next_egress_counts[4]  = egress_counts[4];
        next_egress_counts[5]  = egress_counts[5];
        next_egress_counts[6]  = egress_counts[6];
        next_egress_counts[7]  = egress_counts[7];

        int checksum = next_egress_counts[0]
            + next_egress_counts[6]
            + next_egress_counts[7];

        if (checksum == 0) return;

        clFinish(GPGPU.ptr_compute_queue);
        if (next_egress_counts[0] > 0)
        {
            sector_egress_buffer.front().egress(sector_input.entity_index(), next_egress_counts);
        }
        if (next_egress_counts[6] > 0)
        {
            broken_egress_buffer.front().egress(sector_input.entity_index(), next_egress_counts[6]);
        }
        if (next_egress_counts[7] > 0)
        {
            object_egress_buffer.front().egress(sector_input.entity_index(), next_egress_counts[6]);
        }
        clFinish(GPGPU.ptr_sector_queue);
    }

    public void unload_collected(CollectedObjectBuffer.Raw raw, int count)
    {
        raw.ensure_space(count);
        object_egress_buffer.back().unload(raw, count);
        clFinish(GPGPU.ptr_sector_queue);
    }

    public void unload_broken(BrokenObjectBuffer.Raw raw, int count)
    {
        raw.ensure_space(count);
        broken_egress_buffer.back().unload(raw, count);
        clFinish(GPGPU.ptr_sector_queue);
    }

    public void unload_sectors(UnorderedSectorGroup.Raw raw, int[] egress_counts)
    {
        raw.ensure_space(egress_counts);
        sector_egress_buffer.back().unload(raw, egress_counts);
        clFinish(GPGPU.ptr_sector_queue);
    }

    public void transfer_world_input()
    {
        int point_count         = sector_ingress_buffer.next_point();
        int edge_count          = sector_ingress_buffer.next_edge();
        int hull_count          = sector_ingress_buffer.next_hull();
        int entity_count        = sector_ingress_buffer.next_entity();
        int hull_bone_count     = sector_ingress_buffer.next_hull_bone();
        int armature_bone_count = sector_ingress_buffer.next_armature_bone();

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

        sector_group.ensure_capacity(point_capacity, edge_capacity, hull_capacity, entity_capacity, hull_bone_capacity, armature_bone_capacity);

        clFinish(GPGPU.ptr_compute_queue);
        sector_ingress_buffer.merge_into_parent(this);
        clFinish(GPGPU.ptr_sector_queue);

        sector_input.expand(point_count, edge_count, hull_count, entity_count, hull_bone_count, armature_bone_count);
    }

    public int new_animation_timings(float duration, float tick_rate)
    {
        int capacity = animation_index + 1;

        reference_group.ensure_animation_timings(capacity);

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
        reference_group.ensure_bone_channel(capacity);

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
        reference_group.ensure_keyframe(capacity);

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
        reference_group.ensure_vertex_texture_uv(capacity);

        k_create_texture_uv
            .set_arg(CreateTextureUV_k.Args.target, uv_index)
            .set_arg(CreateTextureUV_k.Args.new_texture_uv, arg_float2(u, v))
            .call(GPGPU.global_single_size);

        return uv_index++;
    }

    @Override
    public int new_edge(int p1, int p2, float l, int flags)
    {
        return sector_input.create_edge(p1, p2, l, flags);
    }

    @Override
    public int new_point(float[] position, int[] bone_ids, int vertex_index, int hull_index, int hit_count, int flags)
    {
        return sector_input.create_point(position, bone_ids, vertex_index, hull_index, hit_count, flags);
    }

    @Override
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
        return sector_input.create_hull(mesh_id, position, scale, rotation, point_table, edge_table, bone_table, friction, restitution, entity_id, uv_offset, flags);
    }

    public int new_mesh_reference(int[] vertex_table, int[] face_table)
    {
        int capacity = mesh_index + 1;
        reference_group.ensure_mesh(capacity);

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
        reference_group.ensure_mesh_face(capacity);

        k_create_mesh_face
            .set_arg(CreateMeshFace_k.Args.target, face_index)
            .set_arg(CreateMeshFace_k.Args.new_mesh_face, face)
            .call(GPGPU.global_single_size);

        return face_index++;
    }

    @Override
    public int new_entity(float x, float y, float z, float w,
                          int[] hull_table,
                          int[] bone_table,
                          float mass,
                          int anim_index,
                          float anim_time,
                          int root_hull,
                          int model_id,
                          int model_transform_id,
                          int type,
                          int flags)
    {
        return sector_input.create_entity(x, y, z, w, hull_table, bone_table, mass, anim_index, anim_time, root_hull, model_id, model_transform_id, type, flags);
    }

    public int new_vertex_reference(float x, float y, float[] weights, int[] uv_table)
    {
        int capacity = vertex_ref_index + 1;
        reference_group.ensure_vertex_reference(capacity);

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
        reference_group.ensure_bind_pose(capacity);

        k_create_bone_bind_pose
            .set_arg(CreateBoneBindPose_k.Args.target,bone_bind_index)
            .set_arg(CreateBoneBindPose_k.Args.new_bone_bind_pose, bone_data)
            .call(GPGPU.global_single_size);

        return bone_bind_index++;
    }


    public int new_bone_reference(float[] bone_data)
    {
        int capacity = bone_ref_index + 1;
        reference_group.ensure_bone_reference(capacity);

        k_create_bone_reference
            .set_arg(CreateBoneRef_k.Args.target, bone_ref_index)
            .set_arg(CreateBoneRef_k.Args.new_bone_reference, bone_data)
            .call(GPGPU.global_single_size);

        return bone_ref_index++;
    }

    @Override
    public int new_hull_bone(float[] bone_data, int bind_pose_id, int inv_bind_pose_id)
    {
        return sector_input.create_hull_bone(bone_data, bind_pose_id, inv_bind_pose_id);
    }

    @Override
    public int new_armature_bone(int bone_reference, int bone_parent_id, float[] bone_data)
    {
        return sector_input.create_armature_bone(bone_reference, bone_parent_id, bone_data);
    }

    public int new_model_transform(float[] transform_data)
    {
        int capacity = model_transform_index + 1;
        reference_group.ensure_model_transform(capacity);

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
        GPGPU.cl_zero_buffer(GPGPU.ptr_compute_queue, ptr_position_buffer, cl_float2);

        k_read_position
            .ptr_arg(ReadPosition_k.Args.output, ptr_position_buffer)
            .set_arg(ReadPosition_k.Args.target, entity_index)
            .call(GPGPU.global_single_size);

        return GPGPU.cl_read_pinned_float_buffer(GPGPU.ptr_compute_queue, ptr_position_buffer, cl_float, 2);
    }

    public int[] count_egress_entities()
    {
        GPGPU.cl_zero_buffer(GPGPU.ptr_compute_queue, ptr_egress_sizes, EGRESS_COUNTERS_SIZE);
        k_count_egress_entities.call(arg_long(sector_input.entity_index()));
        return GPGPU.cl_read_pinned_int_buffer(GPGPU.ptr_compute_queue, ptr_egress_sizes, cl_int, EGRESS_COUNTERS);
    }

    private int init_skip_count = 0;
    private int init_skip_max = 10;

    public void delete_and_compact()
    {
        if (init_skip_count < init_skip_max)
        {
            init_skip_count++;
            return;
        }

        b_delete_1.ensure_capacity(sector_input.entity_index());
        b_delete_2.ensure_capacity(sector_input.entity_index());

        int[] shift_counts = scan_deletes(b_delete_1.pointer(), b_delete_2.pointer(), sector_input.entity_index());

        if (shift_counts[4] == 0)
        {
            return;
        }

        b_hull_shift.ensure_capacity(sector_input.hull_index());
        b_edge_shift.ensure_capacity(sector_input.edge_index());
        b_point_shift.ensure_capacity(sector_input.point_index());
        b_hull_bone_shift.ensure_capacity(sector_input.hull_bone_index());
        b_armature_bone_shift.ensure_capacity(sector_input.entity_bone_index());

        b_hull_shift.clear();
        b_edge_shift.clear();
        b_point_shift.clear();
        b_hull_bone_shift.clear();
        b_armature_bone_shift.clear();

        k_compact_entities
            .ptr_arg(CompactEntities_k.Args.buffer_in_1, b_delete_1.pointer())
            .ptr_arg(CompactEntities_k.Args.buffer_in_2, b_delete_2.pointer());

        linearize_kernel(k_compact_entities, sector_input.entity_index());
        linearize_kernel(k_compact_hull_bones, sector_input.hull_bone_index());
        linearize_kernel(k_compact_points, sector_input.point_index());
        linearize_kernel(k_compact_edges, sector_input.edge_index());
        linearize_kernel(k_compact_hulls, sector_input.hull_index());
        linearize_kernel(k_compact_armature_bones, sector_input.entity_bone_index());

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
        long local_buffer_size = cl_int2 * GPGPU.max_scan_block_size;
        long local_buffer_size2 = cl_int4 * GPGPU.max_scan_block_size;

        GPGPU.cl_zero_buffer(GPGPU.ptr_compute_queue, ptr_delete_sizes, DELETE_COUNTERS_SIZE);

        k_scan_deletes_single_block_out
            .ptr_arg(ScanDeletesSingleBlockOut_k.Args.output, o1_data_ptr)
            .ptr_arg(ScanDeletesSingleBlockOut_k.Args.output2, o2_data_ptr)
            .loc_arg(ScanDeletesSingleBlockOut_k.Args.buffer, local_buffer_size)
            .loc_arg(ScanDeletesSingleBlockOut_k.Args.buffer2, local_buffer_size2)
            .set_arg(ScanDeletesSingleBlockOut_k.Args.n, n)
            .call(GPGPU.local_work_default, GPGPU.local_work_default);

        return GPGPU.cl_read_pinned_int_buffer(GPGPU.ptr_compute_queue, ptr_delete_sizes, cl_int, DELETE_COUNTERS);
    }

    private int[] scan_multi_block_deletes_out(long o1_data_ptr, long o2_data_ptr, int n, int k)
    {
        long local_buffer_size = cl_int2 * GPGPU.max_scan_block_size;
        long local_buffer_size2 = cl_int4 * GPGPU.max_scan_block_size;

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
        gpu_int2_scan.scan_int2(b_delete_partial_1.pointer(), part_size);
        gpu_int4_scan.scan_int4(b_delete_partial_2.pointer(), part_size);

        GPGPU.cl_zero_buffer(GPGPU.ptr_compute_queue, ptr_delete_sizes, DELETE_COUNTERS_SIZE);

        k_complete_deletes_multi_block_out
            .ptr_arg(CompleteDeletesMultiBlockOut_k.Args.output1, o1_data_ptr)
            .ptr_arg(CompleteDeletesMultiBlockOut_k.Args.output2, o2_data_ptr)
            .loc_arg(CompleteDeletesMultiBlockOut_k.Args.buffer1, local_buffer_size)
            .loc_arg(CompleteDeletesMultiBlockOut_k.Args.buffer2, local_buffer_size2)
            .set_arg(CompleteDeletesMultiBlockOut_k.Args.n, n)
            .call(global_work_size, GPGPU.local_work_default);

        return GPGPU.cl_read_pinned_int_buffer(GPGPU.ptr_compute_queue, ptr_delete_sizes, cl_int, DELETE_COUNTERS);
    }

    private void compact_buffers(int[] shift_counts)
    {
        sector_input.compact(shift_counts);
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
        world_barrier.reset();
        sector_group.destroy();
        sector_ingress_buffer.destroy();
        sector_egress_buffer.front().destroy();
        sector_egress_buffer.back().destroy();
        broken_egress_buffer.front().destroy();
        broken_egress_buffer.back().destroy();
        object_egress_buffer.front().destroy();
        object_egress_buffer.back().destroy();
        mirror_group.destroy();
        reference_group.destroy();

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

        debug();

        GPGPU.cl_release_buffer(ptr_delete_counter);
        GPGPU.cl_release_buffer(ptr_position_buffer);
        GPGPU.cl_release_buffer(ptr_delete_sizes);
        GPGPU.cl_release_buffer(ptr_egress_sizes);
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

        //System.out.println("---------------------------");
        System.out.println("Core Memory Usage: MB " + ((float) total / 1024f / 1024f));
    }
}
