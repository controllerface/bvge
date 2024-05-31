package com.controllerface.bvge.physics;

import com.controllerface.bvge.cl.*;
import com.controllerface.bvge.cl.kernels.*;
import com.controllerface.bvge.cl.programs.*;
import com.controllerface.bvge.ecs.ECS;
import com.controllerface.bvge.ecs.components.*;
import com.controllerface.bvge.ecs.systems.GameSystem;
import com.controllerface.bvge.editor.Editor;
import com.controllerface.bvge.util.Constants;
import com.controllerface.bvge.window.Window;

import java.nio.ByteBuffer;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.SynchronousQueue;

import static com.controllerface.bvge.cl.CLUtils.arg_long;
import static org.lwjgl.opencl.CL10.clFinish;

public class PhysicsSimulation extends GameSystem
{
    //#region Constants

    private static final float TARGET_FPS = 24.0f;
    private static final float TICK_RATE = 1.0f / TARGET_FPS;
    private static final int TARGET_SUB_STEPS = 16;
    private static final int MAX_SUB_STEPS = 16;
    private static final float FIXED_TIME_STEP = TICK_RATE / TARGET_SUB_STEPS;
    private static final int EDGE_STEPS = 8;

    // todo: gravity should not be a constant but calculated based on proximity next to planets and other large bodies
    private static final float GRAVITY_MAGNITUDE = 9.8f * 10;
    private static final float GRAVITY_X = 0;
    private static final float GRAVITY_Y = -GRAVITY_MAGNITUDE * TARGET_FPS;

    // todo: investigate if this should be variable as well. It may make sense to increase damping in some cases,
    //  and lower it in others, for example in space vs on a planet. It may also be useful to set the direction
    //  or make damping interact with the gravity vector in some way. Kernels already do this now, may be helpful
    //  to have this as a variable.
    private static final float MOTION_DAMPING = .990f;

    //#endregion

    //#region GPU Programs & Kernels

    private final GPUProgram control_entities = new ControlEntities();
    private final GPUProgram integrate = new Integrate();
    private final GPUProgram scan_key_bank = new ScanKeyBank();
    private final GPUProgram generate_keys = new GenerateKeys();
    private final GPUProgram build_key_map = new BuildKeyMap();
    private final GPUProgram locate_in_bounds = new LocateInBounds();
    private final GPUProgram scan_key_candidates = new ScanKeyCandidates();
    private final GPUProgram aabb_collide = new AabbCollide();
    private final GPUProgram sat_collide = new SatCollide();
    private final GPUProgram animate_hulls = new AnimateHulls();
    private final GPUProgram resolve_constraints = new ResolveConstraints();

    private final GPUKernel set_control_points_k;
    private final GPUKernel handle_movement_k;
    private final GPUKernel integrate_k;
    private final GPUKernel integrate_entities_k;
    private final GPUKernel scan_bounds_single_block_k;
    private final GPUKernel scan_bounds_multi_block_k;
    private final GPUKernel complete_bounds_multi_block_k;
    private final GPUKernel generate_keys_k;
    private final GPUKernel build_key_map_k;
    private final GPUKernel locate_in_bounds_k;
    private final GPUKernel count_candidates_k;
    private final GPUKernel scan_candidates_single_block_out_k;
    private final GPUKernel scan_candidates_multi_block_out_k;
    private final GPUKernel complete_candidates_multi_block_out_k;
    private final GPUKernel aabb_collide_k;
    private final GPUKernel finalize_candidates_k;
    private final GPUKernel sat_collide_k;
    private final GPUKernel sort_reactions_k;
    private final GPUKernel apply_reactions_k;
    private final GPUKernel move_entities_k;
    private final GPUKernel move_hulls_k;
    private final GPUKernel animate_entities_k;
    private final GPUKernel animate_bones_k;
    private final GPUKernel animate_points_k;
    private final GPUKernel resolve_constraints_k;

    //#endregion

    //#region Buffers & Counters

    private final long grid_buffer_size;
    private final ByteBuffer atomic_counter_ptr;
    private final long counts_data_ptr;
    private final long offsets_data_ptr;

    /**
     * int
     * x: count of collision reactions for each point in the current frame
     */
    public final ResizableBuffer point_reaction_counts;

    /**
     * int
     * x: offset into reaction buffer for each point in the current frame
     */
    public final ResizableBuffer point_reaction_offsets;

    /**
     * float8
     * s0: collision reaction x
     * s1: collision reaction y
     * s2: opposing vector x
     * s3: opposing vector y
     * s4: friction reaction x
     * s5: friction reaction y
     * s6: restitution reaction x
     * s7: restitution reaction y
     */
    public final ResizableBuffer reactions_in;

    /**
     * float8
     * s0: collision reaction x
     * s1: collision reaction y
     * s2: opposing vector x
     * s3: opposing vector y
     * s4: friction reaction x
     * s5: friction reaction y
     * s6: restitution reaction x
     * s7: restitution reaction y
     */
    public final ResizableBuffer reactions_out;

    /**
     * int
     * x: index of the point that reactions apply to
     */
    public final ResizableBuffer reaction_index;

    public final ResizableBuffer key_map;
    public final ResizableBuffer key_bank;
    public final ResizableBuffer in_bounds;
    public final ResizableBuffer candidates;
    public final ResizableBuffer candidate_counts;
    public final ResizableBuffer candidate_offsets;
    public final ResizableBuffer matches;
    public final ResizableBuffer matches_used;

    public final ResizableBuffer control_point_flags;
    public final ResizableBuffer control_point_indices;
    public final ResizableBuffer control_point_tick_budgets;
    public final ResizableBuffer control_point_linear_mag;
    public final ResizableBuffer control_point_jump_mag;

    private long candidate_count = 0;
    private long reaction_count = 0;
    private long candidate_buffer_size = 0;
    private long match_buffer_count = 0;
    private long candidate_buffer_count = 0;

    //#endregion

    //#region Simulation Data

    private final UniformGrid uniform_grid;

    private float time_accumulator = 0.0f;

    //#endregion

    //#region Thread & Sync

    private final BlockingQueue<Float> next_phys_time = new SynchronousQueue<>();
    private final BlockingQueue<Long> last_phys_time = new SynchronousQueue<>();
    private final Thread physics_simulation = Thread.ofVirtual().name("Physics-Simulation").start(() ->
    {
        try
        {
            last_phys_time.put(0L);
        }
        catch (InterruptedException e)
        {
            throw new RuntimeException(e);
        }

        while (!Thread.currentThread().isInterrupted())
        {
            try
            {
                var dt = next_phys_time.take();
                long start = System.nanoTime();
                simulate(dt);
                long end = System.nanoTime() - start;
                last_phys_time.put(end);
            }
            catch (InterruptedException e)
            {
                Thread.currentThread().interrupt();
            }
        }
    });


    //#endregion

    public PhysicsSimulation(ECS ecs, UniformGrid uniform_grid)
    {
        super(ecs);
        this.uniform_grid = uniform_grid;

        grid_buffer_size = (long) CLSize.cl_int * this.uniform_grid.directory_length;

        atomic_counter_ptr = GPGPU.cl_new_svm_int();
        counts_data_ptr = GPGPU.cl_new_buffer(grid_buffer_size);
        offsets_data_ptr = GPGPU.cl_new_buffer(grid_buffer_size);

        point_reaction_counts = new TransientBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_int, 500_000L);
        point_reaction_offsets = new TransientBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_int, 500_000L);
        reactions_in = new TransientBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_float8, 500_000L);
        reactions_out = new TransientBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_float8, 500_000L);
        reaction_index = new TransientBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_int, 500_000L);
        key_map = new TransientBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_int, 500_000L);
        key_bank = new TransientBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_int, 500_000L);
        in_bounds = new TransientBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_int, 500_000L);
        candidates = new TransientBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_int2, 500_000L);
        candidate_counts = new TransientBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_int2, 500_000L);
        candidate_offsets = new TransientBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_int, 500_000L);
        matches = new TransientBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_int, 500_000L);
        matches_used = new TransientBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_int, 500_000L);

        control_point_flags = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_int, 1);
        control_point_indices = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_int, 1);
        control_point_tick_budgets = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_int, 1);
        control_point_linear_mag = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_float, 1);
        control_point_jump_mag = new PersistentBuffer(GPGPU.cl_cmd_queue_ptr, CLSize.cl_float, 1);

        control_entities.init();
        integrate.init();
        scan_key_bank.init();
        generate_keys.init();
        build_key_map.init();
        locate_in_bounds.init();
        scan_key_candidates.init();
        aabb_collide.init();
        sat_collide.init();
        animate_hulls.init();
        resolve_constraints.init();

        long set_control_points_k_ptr = control_entities.kernel_ptr(Kernel.set_control_points);
        set_control_points_k = new SetControlPoints_k(GPGPU.cl_cmd_queue_ptr, set_control_points_k_ptr)
            .buf_arg(SetControlPoints_k.Args.flags, control_point_flags)
            .buf_arg(SetControlPoints_k.Args.indices, control_point_indices)
            .buf_arg(SetControlPoints_k.Args.linear_mag, control_point_linear_mag)
            .buf_arg(SetControlPoints_k.Args.jump_mag, control_point_jump_mag);

        long handle_movements_k_ptr = control_entities.kernel_ptr(Kernel.handle_movement);
        handle_movement_k = new HandleMovement_k(GPGPU.cl_cmd_queue_ptr, handle_movements_k_ptr)
            .buf_arg(HandleMovement_k.Args.entities, GPGPU.core_memory.buffer(BufferType.ENTITY))
            .buf_arg(HandleMovement_k.Args.entity_accel, GPGPU.core_memory.buffer(BufferType.ENTITY_ACCEL))
            .buf_arg(HandleMovement_k.Args.entity_motion_states, GPGPU.core_memory.buffer(BufferType.ENTITY_MOTION_STATE))
            .buf_arg(HandleMovement_k.Args.entity_flags, GPGPU.core_memory.buffer(BufferType.ENTITY_FLAG))
            .buf_arg(HandleMovement_k.Args.entity_animation_indices, GPGPU.core_memory.buffer(BufferType.ENTITY_ANIM_INDEX))
            .buf_arg(HandleMovement_k.Args.entity_animation_elapsed, GPGPU.core_memory.buffer(BufferType.ENTITY_ANIM_ELAPSED))
            .buf_arg(HandleMovement_k.Args.entity_animation_blend, GPGPU.core_memory.buffer(BufferType.ENTITY_ANIM_BLEND))
            .buf_arg(HandleMovement_k.Args.flags, control_point_flags)
            .buf_arg(HandleMovement_k.Args.indices, control_point_indices)
            .buf_arg(HandleMovement_k.Args.tick_budgets, control_point_tick_budgets)
            .buf_arg(HandleMovement_k.Args.linear_mag, control_point_linear_mag)
            .buf_arg(HandleMovement_k.Args.jump_mag, control_point_jump_mag);

        long integrate_k_ptr = integrate.kernel_ptr(Kernel.integrate);
        integrate_k = new Integrate_k(GPGPU.cl_cmd_queue_ptr, integrate_k_ptr)
            .buf_arg(Integrate_k.Args.hulls, GPGPU.core_memory.buffer(BufferType.HULL))
            .buf_arg(Integrate_k.Args.hull_scales, GPGPU.core_memory.buffer(BufferType.HULL_SCALE))
            .buf_arg(Integrate_k.Args.hull_point_tables, GPGPU.core_memory.buffer(BufferType.HULL_POINT_TABLE))
            .buf_arg(Integrate_k.Args.entity_accel, GPGPU.core_memory.buffer(BufferType.ENTITY_ACCEL))
            .buf_arg(Integrate_k.Args.hull_rotations, GPGPU.core_memory.buffer(BufferType.HULL_ROTATION))
            .buf_arg(Integrate_k.Args.points, GPGPU.core_memory.buffer(BufferType.POINT))
            .buf_arg(Integrate_k.Args.point_hit_counts, GPGPU.core_memory.buffer(BufferType.POINT_HIT_COUNT))
            .buf_arg(Integrate_k.Args.point_flags, GPGPU.core_memory.buffer(BufferType.POINT_FLAG))
            .buf_arg(Integrate_k.Args.bounds, GPGPU.core_memory.buffer(BufferType.HULL_AABB))
            .buf_arg(Integrate_k.Args.bounds_index_data, GPGPU.core_memory.buffer(BufferType.HULL_AABB_INDEX))
            .buf_arg(Integrate_k.Args.bounds_bank_data, GPGPU.core_memory.buffer(BufferType.HULL_AABB_KEY_TABLE))
            .buf_arg(Integrate_k.Args.hull_flags, GPGPU.core_memory.buffer(BufferType.HULL_FLAG))
            .buf_arg(Integrate_k.Args.hull_entity_ids, GPGPU.core_memory.buffer(BufferType.HULL_ENTITY_ID))
            .buf_arg(Integrate_k.Args.anti_gravity, GPGPU.core_memory.buffer(BufferType.POINT_ANTI_GRAV));

        long integrate_entities_k_ptr = integrate.kernel_ptr(Kernel.integrate_entities);
        integrate_entities_k = new IntegrateEntities_k(GPGPU.cl_cmd_queue_ptr, integrate_entities_k_ptr)
            .buf_arg(IntegrateEntities_k.Args.entities, GPGPU.core_memory.buffer(BufferType.ENTITY))
            .buf_arg(IntegrateEntities_k.Args.entity_flags, GPGPU.core_memory.buffer(BufferType.ENTITY_FLAG))
            .buf_arg(IntegrateEntities_k.Args.entity_root_hulls, GPGPU.core_memory.buffer(BufferType.ENTITY_ROOT_HULL))
            .buf_arg(IntegrateEntities_k.Args.entity_accel, GPGPU.core_memory.buffer(BufferType.ENTITY_ACCEL))
            .buf_arg(IntegrateEntities_k.Args.hull_flags, GPGPU.core_memory.buffer(BufferType.HULL_FLAG));

        long scan_bounds_single_block_k_ptr = scan_key_bank.kernel_ptr(Kernel.scan_bounds_single_block);
        scan_bounds_single_block_k = new ScanBoundsSingleBlock_k(GPGPU.cl_cmd_queue_ptr, scan_bounds_single_block_k_ptr);

        long scan_bounds_multi_block_k_ptr = scan_key_bank.kernel_ptr(Kernel.scan_bounds_multi_block);
        scan_bounds_multi_block_k = new ScanBoundsMultiBlock_k(GPGPU.cl_cmd_queue_ptr, scan_bounds_multi_block_k_ptr);

        long complete_bounds_multi_block_k_ptr = scan_key_bank.kernel_ptr(Kernel.complete_bounds_multi_block);
        complete_bounds_multi_block_k = new CompleteBoundsMultiBlock_k(GPGPU.cl_cmd_queue_ptr, complete_bounds_multi_block_k_ptr);

        long generate_keys_k_ptr = generate_keys.kernel_ptr(Kernel.generate_keys);
        generate_keys_k = new GenerateKeys_k(GPGPU.cl_cmd_queue_ptr, generate_keys_k_ptr)
            .buf_arg(GenerateKeys_k.Args.key_bank, key_bank)
            .buf_arg(GenerateKeys_k.Args.bounds_index_data, GPGPU.core_memory.buffer(BufferType.HULL_AABB_INDEX))
            .buf_arg(GenerateKeys_k.Args.bounds_bank_data, GPGPU.core_memory.buffer(BufferType.HULL_AABB_KEY_TABLE))
            .ptr_arg(GenerateKeys_k.Args.key_counts, counts_data_ptr)
            .set_arg(GenerateKeys_k.Args.x_subdivisions, uniform_grid.x_subdivisions)
            .set_arg(GenerateKeys_k.Args.key_count_length, uniform_grid.directory_length);

        long build_key_map_k_ptr = build_key_map.kernel_ptr(Kernel.build_key_map);
        build_key_map_k = new BuildKeyMap_k(GPGPU.cl_cmd_queue_ptr, build_key_map_k_ptr)
            .buf_arg(BuildKeyMap_k.Args.key_map, key_map)
            .buf_arg(BuildKeyMap_k.Args.bounds_index_data, GPGPU.core_memory.buffer(BufferType.HULL_AABB_INDEX))
            .buf_arg(BuildKeyMap_k.Args.bounds_bank_data, GPGPU.core_memory.buffer(BufferType.HULL_AABB_KEY_TABLE))
            .ptr_arg(BuildKeyMap_k.Args.key_offsets, offsets_data_ptr)
            .ptr_arg(BuildKeyMap_k.Args.key_counts, counts_data_ptr)
            .set_arg(BuildKeyMap_k.Args.x_subdivisions, uniform_grid.x_subdivisions)
            .set_arg(BuildKeyMap_k.Args.key_count_length, uniform_grid.directory_length);

        long locate_in_bounds_k_ptr = locate_in_bounds.kernel_ptr(Kernel.locate_in_bounds);
        locate_in_bounds_k = (new LocateInBounds_k(GPGPU.cl_cmd_queue_ptr, locate_in_bounds_k_ptr))
            .buf_arg(LocateInBounds_k.Args.in_bounds, in_bounds)
            .buf_arg(LocateInBounds_k.Args.bounds_bank_data, GPGPU.core_memory.buffer(BufferType.HULL_AABB_KEY_TABLE));

        long count_candidates_k_ptr = locate_in_bounds.kernel_ptr(Kernel.count_candidates);
        count_candidates_k = new CountCandidates_k(GPGPU.cl_cmd_queue_ptr, count_candidates_k_ptr)
            .buf_arg(CountCandidates_k.Args.candidates, candidate_counts)
            .buf_arg(CountCandidates_k.Args.key_bank, key_bank)
            .buf_arg(CountCandidates_k.Args.in_bounds, in_bounds)
            .buf_arg(CountCandidates_k.Args.bounds_bank_data, GPGPU.core_memory.buffer(BufferType.HULL_AABB_KEY_TABLE))
            .ptr_arg(CountCandidates_k.Args.key_counts, counts_data_ptr)
            .set_arg(CountCandidates_k.Args.x_subdivisions, uniform_grid.x_subdivisions)
            .set_arg(CountCandidates_k.Args.key_count_length, uniform_grid.directory_length);

        long scan_candidates_single_block_out_k_ptr = scan_key_candidates.kernel_ptr(Kernel.scan_candidates_single_block_out);
        scan_candidates_single_block_out_k = new ScanCandidatesSingleBlockOut_k(GPGPU.cl_cmd_queue_ptr, scan_candidates_single_block_out_k_ptr);

        long scan_candidates_multi_block_out_k_ptr = scan_key_candidates.kernel_ptr(Kernel.scan_candidates_multi_block_out);
        scan_candidates_multi_block_out_k = new ScanCandidatesMultiBlockOut_k(GPGPU.cl_cmd_queue_ptr, scan_candidates_multi_block_out_k_ptr);

        long complete_candidates_multi_block_out_k_ptr = scan_key_candidates.kernel_ptr(Kernel.complete_candidates_multi_block_out);
        complete_candidates_multi_block_out_k = new CompleteCandidatesMultiBlockOut_k(GPGPU.cl_cmd_queue_ptr, complete_candidates_multi_block_out_k_ptr);

        long aabb_collide_k_ptr = aabb_collide.kernel_ptr(Kernel.aabb_collide);
        aabb_collide_k = new AABBCollide_k(GPGPU.cl_cmd_queue_ptr, aabb_collide_k_ptr)
            .buf_arg(AABBCollide_k.Args.used, matches_used)
            .buf_arg(AABBCollide_k.Args.matches, matches)
            .buf_arg(AABBCollide_k.Args.match_offsets, candidate_offsets)
            .buf_arg(AABBCollide_k.Args.candidates, candidate_counts)
            .buf_arg(AABBCollide_k.Args.key_map, key_map)
            .buf_arg(AABBCollide_k.Args.key_bank, key_bank)
            .buf_arg(AABBCollide_k.Args.bounds, GPGPU.core_memory.buffer(BufferType.HULL_AABB))
            .buf_arg(AABBCollide_k.Args.bounds_bank_data, GPGPU.core_memory.buffer(BufferType.HULL_AABB_KEY_TABLE))
            .buf_arg(AABBCollide_k.Args.hull_entity_ids, GPGPU.core_memory.buffer(BufferType.HULL_ENTITY_ID))
            .buf_arg(AABBCollide_k.Args.hull_flags, GPGPU.core_memory.buffer(BufferType.HULL_FLAG))
            .ptr_arg(AABBCollide_k.Args.key_counts, counts_data_ptr)
            .ptr_arg(AABBCollide_k.Args.key_offsets, offsets_data_ptr)
            .ptr_arg(AABBCollide_k.Args.counter, atomic_counter_ptr)
            .set_arg(AABBCollide_k.Args.x_subdivisions, uniform_grid.x_subdivisions)
            .set_arg(AABBCollide_k.Args.key_count_length, uniform_grid.directory_length);

        long finalize_candidates_k_ptr = locate_in_bounds.kernel_ptr(Kernel.finalize_candidates);
        finalize_candidates_k = new FinalizeCandidates_k(GPGPU.cl_cmd_queue_ptr, finalize_candidates_k_ptr)
            .buf_arg(FinalizeCandidates_k.Args.used, matches_used)
            .buf_arg(FinalizeCandidates_k.Args.matches, matches)
            .buf_arg(FinalizeCandidates_k.Args.match_offsets, candidate_offsets)
            .buf_arg(FinalizeCandidates_k.Args.input_candidates, candidate_counts)
            .buf_arg(FinalizeCandidates_k.Args.final_candidates, candidates);

        long sat_collide_k_ptr = sat_collide.kernel_ptr(Kernel.sat_collide);
        sat_collide_k = new SatCollide_k(GPGPU.cl_cmd_queue_ptr, sat_collide_k_ptr)
            .buf_arg(SatCollide_k.Args.candidates, candidates)
            .buf_arg(SatCollide_k.Args.hulls, GPGPU.core_memory.buffer(BufferType.HULL))
            .buf_arg(SatCollide_k.Args.hull_scales, GPGPU.core_memory.buffer(BufferType.HULL_SCALE))
            .buf_arg(SatCollide_k.Args.hull_frictions, GPGPU.core_memory.buffer(BufferType.HULL_FRICTION))
            .buf_arg(SatCollide_k.Args.hull_restitutions, GPGPU.core_memory.buffer(BufferType.HULL_RESTITUTION))
            .buf_arg(SatCollide_k.Args.hull_integrity, GPGPU.core_memory.buffer(BufferType.HULL_INTEGRITY))
            .buf_arg(SatCollide_k.Args.hull_point_tables, GPGPU.core_memory.buffer(BufferType.HULL_POINT_TABLE))
            .buf_arg(SatCollide_k.Args.hull_edge_tables, GPGPU.core_memory.buffer(BufferType.HULL_EDGE_TABLE))
            .buf_arg(SatCollide_k.Args.hull_entity_ids, GPGPU.core_memory.buffer(BufferType.HULL_ENTITY_ID))
            .buf_arg(SatCollide_k.Args.hull_flags, GPGPU.core_memory.buffer(BufferType.HULL_FLAG))
            .buf_arg(SatCollide_k.Args.point_flags, GPGPU.core_memory.buffer(BufferType.POINT_FLAG))
            .buf_arg(SatCollide_k.Args.points, GPGPU.core_memory.buffer(BufferType.POINT))
            .buf_arg(SatCollide_k.Args.edges, GPGPU.core_memory.buffer(BufferType.EDGE))
            .buf_arg(SatCollide_k.Args.edge_flags, GPGPU.core_memory.buffer(BufferType.EDGE_FLAG))
            .buf_arg(SatCollide_k.Args.reactions, reactions_in)
            .buf_arg(SatCollide_k.Args.reaction_index, reaction_index)
            .buf_arg(SatCollide_k.Args.point_reactions, point_reaction_counts)
            .buf_arg(SatCollide_k.Args.masses, GPGPU.core_memory.buffer(BufferType.ENTITY_MASS))
            .ptr_arg(SatCollide_k.Args.counter, atomic_counter_ptr)
            .set_arg(SatCollide_k.Args.dt, FIXED_TIME_STEP);

        long sort_reactions_k_ptr = sat_collide.kernel_ptr(Kernel.sort_reactions);
        sort_reactions_k = new SortReactions_k(GPGPU.cl_cmd_queue_ptr, sort_reactions_k_ptr)
            .buf_arg(SortReactions_k.Args.reactions_in, reactions_in)
            .buf_arg(SortReactions_k.Args.reactions_out, reactions_out)
            .buf_arg(SortReactions_k.Args.reaction_index, reaction_index)
            .buf_arg(SortReactions_k.Args.point_reactions, point_reaction_counts)
            .buf_arg(SortReactions_k.Args.point_offsets, point_reaction_offsets);

        long apply_reactions_k_ptr = sat_collide.kernel_ptr(Kernel.apply_reactions);
        apply_reactions_k = new ApplyReactions_k(GPGPU.cl_cmd_queue_ptr, apply_reactions_k_ptr)
            .buf_arg(ApplyReactions_k.Args.reactions, reactions_out)
            .buf_arg(ApplyReactions_k.Args.points, GPGPU.core_memory.buffer(BufferType.POINT))
            .buf_arg(ApplyReactions_k.Args.anti_gravity, GPGPU.core_memory.buffer(BufferType.POINT_ANTI_GRAV))
            .buf_arg(ApplyReactions_k.Args.point_flags, GPGPU.core_memory.buffer(BufferType.POINT_FLAG))
            .buf_arg(ApplyReactions_k.Args.point_hit_counts, GPGPU.core_memory.buffer(BufferType.POINT_HIT_COUNT))
            .buf_arg(ApplyReactions_k.Args.point_reactions, point_reaction_counts)
            .buf_arg(ApplyReactions_k.Args.point_offsets, point_reaction_offsets)
            .buf_arg(ApplyReactions_k.Args.point_hull_indices, GPGPU.core_memory.buffer(BufferType.POINT_HULL_INDEX))
            .buf_arg(ApplyReactions_k.Args.hull_flags, GPGPU.core_memory.buffer(BufferType.HULL_FLAG));

        long move_hulls_k_ptr = sat_collide.kernel_ptr(Kernel.move_hulls);
        move_hulls_k = new MoveHulls_k(GPGPU.cl_cmd_queue_ptr, move_hulls_k_ptr)
            .buf_arg(MoveHulls_k.Args.hulls, GPGPU.core_memory.buffer(BufferType.HULL))
            .buf_arg(MoveHulls_k.Args.hull_point_tables, GPGPU.core_memory.buffer(BufferType.HULL_POINT_TABLE))
            .buf_arg(MoveHulls_k.Args.points, GPGPU.core_memory.buffer(BufferType.POINT));

        long move_entities_k_ptr = sat_collide.kernel_ptr(Kernel.move_entities);
        move_entities_k = new MoveEntities_k(GPGPU.cl_cmd_queue_ptr, move_entities_k_ptr)
            .buf_arg(MoveEntities_k.Args.hulls, GPGPU.core_memory.buffer(BufferType.HULL))
            .buf_arg(MoveEntities_k.Args.entities, GPGPU.core_memory.buffer(BufferType.ENTITY))
            .buf_arg(MoveEntities_k.Args.entity_flags, GPGPU.core_memory.buffer(BufferType.ENTITY_FLAG))
            .buf_arg(MoveEntities_k.Args.entity_hull_tables, GPGPU.core_memory.buffer(BufferType.ENTITY_HULL_TABLE))
            .buf_arg(MoveEntities_k.Args.hull_point_tables, GPGPU.core_memory.buffer(BufferType.HULL_POINT_TABLE))
            .buf_arg(MoveEntities_k.Args.hull_flags, GPGPU.core_memory.buffer(BufferType.HULL_FLAG))
            .buf_arg(MoveEntities_k.Args.point_flags, GPGPU.core_memory.buffer(BufferType.POINT_FLAG))
            .buf_arg(MoveEntities_k.Args.point_hit_counts, GPGPU.core_memory.buffer(BufferType.POINT_HIT_COUNT))
            .buf_arg(MoveEntities_k.Args.points, GPGPU.core_memory.buffer(BufferType.POINT));

        long animate_entities_k_ptr = animate_hulls.kernel_ptr(Kernel.animate_entities);
        animate_entities_k = new AnimateEntities_k(GPGPU.cl_cmd_queue_ptr, animate_entities_k_ptr)
            .buf_arg(AnimateEntities_k.Args.armature_bones, GPGPU.core_memory.buffer(BufferType.ARMATURE_BONE))
            .buf_arg(AnimateEntities_k.Args.bone_bind_poses, GPGPU.core_memory.buffer(BufferType.BONE_BIND_POSE))
            .buf_arg(AnimateEntities_k.Args.model_transforms, GPGPU.core_memory.buffer(BufferType.MODEL_TRANSFORM))
            .buf_arg(AnimateEntities_k.Args.entity_flags, GPGPU.core_memory.buffer(BufferType.ENTITY_FLAG))
            .buf_arg(AnimateEntities_k.Args.armature_bone_reference_ids, GPGPU.core_memory.buffer(BufferType.ARMATURE_BONE_REFERENCE_ID))
            .buf_arg(AnimateEntities_k.Args.armature_bone_parent_ids, GPGPU.core_memory.buffer(BufferType.ARMATURE_BONE_PARENT_ID))
            .buf_arg(AnimateEntities_k.Args.bone_channel_tables, GPGPU.core_memory.buffer(BufferType.BONE_ANIM_TABLE))
            .buf_arg(AnimateEntities_k.Args.bone_pos_channel_tables, GPGPU.core_memory.buffer(BufferType.ANIM_POS_CHANNEL))
            .buf_arg(AnimateEntities_k.Args.bone_rot_channel_tables, GPGPU.core_memory.buffer(BufferType.ANIM_ROT_CHANNEL))
            .buf_arg(AnimateEntities_k.Args.bone_scl_channel_tables, GPGPU.core_memory.buffer(BufferType.ANIM_SCL_CHANNEL))
            .buf_arg(AnimateEntities_k.Args.entity_model_transforms, GPGPU.core_memory.buffer(BufferType.ENTITY_TRANSFORM_ID))
            .buf_arg(AnimateEntities_k.Args.entity_bone_tables, GPGPU.core_memory.buffer(BufferType.ENTITY_BONE_TABLE))
            .buf_arg(AnimateEntities_k.Args.key_frames, GPGPU.core_memory.buffer(BufferType.ANIM_KEY_FRAME))
            .buf_arg(AnimateEntities_k.Args.frame_times, GPGPU.core_memory.buffer(BufferType.ANIM_FRAME_TIME))
            .buf_arg(AnimateEntities_k.Args.animation_timing_indices, GPGPU.core_memory.buffer(BufferType.ANIM_TIMING_INDEX))
            .buf_arg(AnimateEntities_k.Args.animation_durations, GPGPU.core_memory.buffer(BufferType.ANIM_DURATION))
            .buf_arg(AnimateEntities_k.Args.animation_tick_rates, GPGPU.core_memory.buffer(BufferType.ANIM_TICK_RATE))
            .buf_arg(AnimateEntities_k.Args.entity_animation_indices, GPGPU.core_memory.buffer(BufferType.ENTITY_ANIM_INDEX))
            .buf_arg(AnimateEntities_k.Args.entity_animation_elapsed, GPGPU.core_memory.buffer(BufferType.ENTITY_ANIM_ELAPSED))
            .buf_arg(AnimateEntities_k.Args.entity_animation_blend, GPGPU.core_memory.buffer(BufferType.ENTITY_ANIM_BLEND));

        long animate_bones_k_ptr = animate_hulls.kernel_ptr(Kernel.animate_bones);
        animate_bones_k = new AnimateBones_k(GPGPU.cl_cmd_queue_ptr, animate_bones_k_ptr)
            .buf_arg(AnimateBones_k.Args.bones, GPGPU.core_memory.buffer(BufferType.HULL_BONE))
            .buf_arg(AnimateBones_k.Args.bone_references, GPGPU.core_memory.buffer(BufferType.BONE_REFERENCE))
            .buf_arg(AnimateBones_k.Args.armature_bones, GPGPU.core_memory.buffer(BufferType.ARMATURE_BONE))
            .buf_arg(AnimateBones_k.Args.hull_bind_pose_indicies, GPGPU.core_memory.buffer(BufferType.HULL_BONE_BIND_POSE))
            .buf_arg(AnimateBones_k.Args.hull_inv_bind_pose_indicies, GPGPU.core_memory.buffer(BufferType.HULL_BONE_INV_BIND_POSE));

        long animate_points_k_ptr = animate_hulls.kernel_ptr(Kernel.animate_points);
        animate_points_k = new AnimatePoints_k(GPGPU.cl_cmd_queue_ptr, animate_points_k_ptr)
            .buf_arg(AnimatePoints_k.Args.points, GPGPU.core_memory.buffer(BufferType.POINT))
            .buf_arg(AnimatePoints_k.Args.hull_scales, GPGPU.core_memory.buffer(BufferType.HULL_SCALE))
            .buf_arg(AnimatePoints_k.Args.hull_entity_ids, GPGPU.core_memory.buffer(BufferType.HULL_ENTITY_ID))
            .buf_arg(AnimatePoints_k.Args.hull_flags, GPGPU.core_memory.buffer(BufferType.HULL_FLAG))
            .buf_arg(AnimatePoints_k.Args.point_vertex_references, GPGPU.core_memory.buffer(BufferType.POINT_VERTEX_REFERENCE))
            .buf_arg(AnimatePoints_k.Args.point_hull_indices, GPGPU.core_memory.buffer(BufferType.POINT_HULL_INDEX))
            .buf_arg(AnimatePoints_k.Args.bone_tables, GPGPU.core_memory.buffer(BufferType.POINT_BONE_TABLE))
            .buf_arg(AnimatePoints_k.Args.vertex_weights, GPGPU.core_memory.buffer(BufferType.VERTEX_WEIGHT))
            .buf_arg(AnimatePoints_k.Args.entities, GPGPU.core_memory.buffer(BufferType.ENTITY))
            .buf_arg(AnimatePoints_k.Args.vertex_references, GPGPU.core_memory.buffer(BufferType.VERTEX_REFERENCE))
            .buf_arg(AnimatePoints_k.Args.bones, GPGPU.core_memory.buffer(BufferType.HULL_BONE));

        long resolve_constraints_k_ptr = resolve_constraints.kernel_ptr(Kernel.resolve_constraints);
        resolve_constraints_k = new ResolveConstraints_k(GPGPU.cl_cmd_queue_ptr, resolve_constraints_k_ptr)
            .buf_arg(ResolveConstraints_k.Args.hull_edge_tables, GPGPU.core_memory.buffer(BufferType.HULL_EDGE_TABLE))
            .buf_arg(ResolveConstraints_k.Args.bounds_bank_data, GPGPU.core_memory.buffer(BufferType.HULL_AABB_KEY_TABLE))
            .buf_arg(ResolveConstraints_k.Args.point, GPGPU.core_memory.buffer(BufferType.POINT))
            .buf_arg(ResolveConstraints_k.Args.edges, GPGPU.core_memory.buffer(BufferType.EDGE))
            .buf_arg(ResolveConstraints_k.Args.edge_lengths, GPGPU.core_memory.buffer(BufferType.EDGE_LENGTH));
    }

    //#region Input & Integration

    private void integrate()
    {
        long s = Editor.ACTIVE ? System.nanoTime() : 0;

        float[] args =
            {
                FIXED_TIME_STEP,
                GRAVITY_X,
                GRAVITY_Y,
                MOTION_DAMPING,
                uniform_grid.x_spacing,
                uniform_grid.y_spacing,
                uniform_grid.x_origin(),
                uniform_grid.y_origin(),
                uniform_grid.width,
                uniform_grid.height,
                uniform_grid.inner_x_origin(),
                uniform_grid.inner_y_origin(),
                uniform_grid.inner_width,
                uniform_grid.inner_height,
                uniform_grid.sector_origin_x(),
                uniform_grid.sector_origin_y(),
                uniform_grid.sector_width(),
                uniform_grid.sector_height(),
            };

        var arg_mem_ptr = GPGPU.cl_new_cpu_copy_buffer(args);

        integrate_k
            .ptr_arg(Integrate_k.Args.args, arg_mem_ptr)
            .call(arg_long(GPGPU.core_memory.next_hull()));

        integrate_entities_k
            .ptr_arg(IntegrateEntities_k.Args.args, arg_mem_ptr)
            .call(arg_long(GPGPU.core_memory.next_entity()));

        GPGPU.cl_release_buffer(arg_mem_ptr);

        if (Editor.ACTIVE)
        {
            long e = System.nanoTime() - s;
            Editor.queue_event("phys_integrate", String.valueOf(e));
        }
    }

    private void update_controllable_entities()
    {
        // todo: index and magnitudes only need to be set once, but may need some
        //  checks or logic to ensure characters don't get deleted
        var components = ecs.getComponents(Component.ControlPoints);

        control_point_flags.ensure_capacity(components.size());
        control_point_indices.ensure_capacity(components.size());
        control_point_tick_budgets.ensure_capacity(components.size());
        control_point_linear_mag.ensure_capacity(components.size());
        control_point_jump_mag.ensure_capacity(components.size());

        int target_count = 0;
        for (Map.Entry<String, GameComponent> entry : components.entrySet())
        {
            String entity_name = entry.getKey();
            GameComponent component = entry.getValue();
            ControlPoints controlPoints = Component.ControlPoints.coerce(component);
            EntityIndex entity_id = Component.EntityId.forEntity(ecs, entity_name);
            EntityIndex cursor_entity_id = Component.CursorId.forEntity(ecs, entity_name);
            LinearForce force = Component.LinearForce.forEntity(ecs, entity_name);

            Objects.requireNonNull(controlPoints);
            Objects.requireNonNull(entity_id);
            Objects.requireNonNull(force);

            int flags = 0;

            var inputStates = controlPoints.input_states();
            for (var binding : InputBinding.values())
            {
                var on = inputStates.get(binding);
                if (on)
                {
                    int flag = switch (binding)
                    {
                        case MOVE_UP -> Constants.ControlFlags.UP.bits;
                        case MOVE_DOWN -> Constants.ControlFlags.DOWN.bits;
                        case MOVE_LEFT -> Constants.ControlFlags.LEFT.bits;
                        case MOVE_RIGHT -> Constants.ControlFlags.RIGHT.bits;
                        case JUMP -> Constants.ControlFlags.JUMP.bits;
                        case RUN -> Constants.ControlFlags.RUN.bits;
                        case MOUSE_PRIMARY -> Constants.ControlFlags.MOUSE1.bits;
                        case MOUSE_SECONDARY -> Constants.ControlFlags.MOUSE2.bits;
                        case MOUSE_MIDDLE -> 0;
                        case MOUSE_BACK -> 0;
                        case MOUSE_FORWARD -> 0;
                    };
                    flags |= flag;
                }
            }

            set_control_points_k
                .set_arg(SetControlPoints_k.Args.target, target_count)
                .set_arg(SetControlPoints_k.Args.new_flags, flags)
                .set_arg(SetControlPoints_k.Args.new_index, entity_id.index())
                .set_arg(SetControlPoints_k.Args.new_jump_mag, GRAVITY_MAGNITUDE * 550)
                .set_arg(SetControlPoints_k.Args.new_linear_mag, force.magnitude())
                .call(GPGPU.global_single_size);

            var camera = Window.get().camera();
            float world_x = controlPoints.get_screen_target().x * camera.get_zoom() + camera.position.x;
            float world_y = (Window.get().height() - controlPoints.get_screen_target().y) * camera.get_zoom() + camera.position.y;
            GPGPU.core_memory.update_position(cursor_entity_id.index(), world_x, world_y);

            target_count++;
        }

        handle_movement_k.set_arg(HandleMovement_k.Args.dt, FIXED_TIME_STEP)
            .call(arg_long(target_count));
    }

    //#endregion

    //#region AABB Collision

    private void calculate_bank_offsets()
    {
        long s = Editor.ACTIVE ? System.nanoTime() : 0;
        int bank_size = scan_key_bounds(GPGPU.core_memory.buffer(BufferType.HULL_AABB_KEY_TABLE).pointer(), GPGPU.core_memory.next_hull());
        uniform_grid.resizeBank(bank_size);
        if (Editor.ACTIVE)
        {
            long e = System.nanoTime() - s;
            Editor.queue_event("phys_bank_offset", String.valueOf(e));
        }
    }

    private int scan_bounds_single_block(long data_ptr, int n)
    {
        long local_buffer_size = CLSize.cl_int * GPGPU.max_scan_block_size;

        GPGPU.cl_zero_buffer(GPGPU.cl_cmd_queue_ptr, atomic_counter_ptr, CLSize.cl_int);

        scan_bounds_single_block_k
            .ptr_arg(ScanBoundsSingleBlock_k.Args.bounds_bank_data, data_ptr)
            .ptr_arg(ScanBoundsSingleBlock_k.Args.sz, atomic_counter_ptr)
            .loc_arg(ScanBoundsSingleBlock_k.Args.buffer, local_buffer_size)
            .set_arg(ScanBoundsSingleBlock_k.Args.n, n)
            .call(GPGPU.local_work_default, GPGPU.local_work_default);

        return GPGPU.cl_read_svm_int(GPGPU.cl_cmd_queue_ptr, atomic_counter_ptr);
    }

    private int scan_bounds_multi_block(long data_ptr, int n, int k)
    {
        long s = Editor.ACTIVE ? System.nanoTime() : 0;

        long local_buffer_size = CLSize.cl_int * GPGPU.max_scan_block_size;
        long gx = k * GPGPU.max_scan_block_size;
        long[] global_work_size = arg_long(gx);
        int part_size = k * 2;
        long part_buf_size = ((long) CLSize.cl_int * ((long) part_size));
        var p_data = GPGPU.cl_new_buffer(part_buf_size);

        scan_bounds_multi_block_k
            .ptr_arg(ScanBoundsMultiBlock_k.Args.bounds_bank_data, data_ptr)
            .loc_arg(ScanBoundsMultiBlock_k.Args.buffer, local_buffer_size)
            .ptr_arg(ScanBoundsMultiBlock_k.Args.part, p_data)
            .set_arg(ScanBoundsMultiBlock_k.Args.n, n)
            .call(global_work_size, GPGPU.local_work_default);

        if (Editor.ACTIVE)
        {
            long e = System.nanoTime() - s;
            Editor.queue_event("phys_bank_scan_bounds", String.valueOf(e));
        }

        GPGPU.scan_int(p_data, part_size);

        GPGPU.cl_zero_buffer(GPGPU.cl_cmd_queue_ptr, atomic_counter_ptr, CLSize.cl_int);

        s = Editor.ACTIVE ? System.nanoTime() : 0;

        complete_bounds_multi_block_k
            .ptr_arg(CompleteBoundsMultiBlock_k.Args.bounds_bank_data, data_ptr)
            .ptr_arg(CompleteBoundsMultiBlock_k.Args.sz, atomic_counter_ptr)
            .loc_arg(CompleteBoundsMultiBlock_k.Args.buffer, local_buffer_size)
            .ptr_arg(CompleteBoundsMultiBlock_k.Args.part, p_data)
            .set_arg(CompleteBoundsMultiBlock_k.Args.n, n)
            .call(global_work_size, GPGPU.local_work_default);

        GPGPU.cl_release_buffer(p_data);

        if (Editor.ACTIVE)
        {
            long e = System.nanoTime() - s;
            Editor.queue_event("phys_bank_complete_bounds", String.valueOf(e));
        }

        s = Editor.ACTIVE ? System.nanoTime() : 0;
        int r = GPGPU.cl_read_svm_int(GPGPU.cl_cmd_queue_ptr, atomic_counter_ptr);
        if (Editor.ACTIVE)
        {
            long e = System.nanoTime() - s;
            Editor.queue_event("phys_bank_read_pinned", String.valueOf(e));
        }

        return r;
    }

    private int scan_key_bounds(long data_ptr, int n)
    {
        int k = GPGPU.work_group_count(n);
        if (k == 1)
        {
            return scan_bounds_single_block(data_ptr, n);
        }
        else
        {
            return scan_bounds_multi_block(data_ptr, n, k);
        }
    }

    private void generate_keys()
    {
        long s = Editor.ACTIVE ? System.nanoTime() : 0;
        if (uniform_grid.get_key_bank_size() < 1)
        {
            return;
        }

        key_bank.ensure_capacity(uniform_grid.get_key_bank_size());
        GPGPU.cl_zero_buffer(GPGPU.cl_cmd_queue_ptr, counts_data_ptr, grid_buffer_size);
        generate_keys_k
            .set_arg(GenerateKeys_k.Args.key_bank_length, uniform_grid.get_key_bank_size())
            .call(arg_long(GPGPU.core_memory.next_hull()));
        if (Editor.ACTIVE)
        {
            long e = System.nanoTime() - s;
            Editor.queue_event("phys_gen_keys", String.valueOf(e));
        }
    }

    private void build_key_map(UniformGrid uniform_grid)
    {
        long s = Editor.ACTIVE ? System.nanoTime() : 0;
        key_map.ensure_capacity(uniform_grid.getKey_map_size());
        GPGPU.cl_zero_buffer(GPGPU.cl_cmd_queue_ptr, counts_data_ptr, grid_buffer_size);
        build_key_map_k.call(arg_long(GPGPU.core_memory.next_hull()));
        if (Editor.ACTIVE)
        {
            long e = System.nanoTime() - s;
            Editor.queue_event("phys_key_map", String.valueOf(e));
        }
    }

    private void locate_in_bounds()
    {
        long s = Editor.ACTIVE ? System.nanoTime() : 0;
        int hull_count = GPGPU.core_memory.next_hull();
        in_bounds.ensure_capacity(hull_count);
        GPGPU.cl_zero_buffer(GPGPU.cl_cmd_queue_ptr, atomic_counter_ptr, CLSize.cl_int);

        locate_in_bounds_k
            .ptr_arg(LocateInBounds_k.Args.counter, atomic_counter_ptr)
            .call(arg_long(hull_count));

        candidate_buffer_count = GPGPU.cl_read_svm_int(GPGPU.cl_cmd_queue_ptr, atomic_counter_ptr);
        if (Editor.ACTIVE)
        {
            long e = System.nanoTime() - s;
            Editor.queue_event("phys_locate_inbounds", String.valueOf(e));
        }
    }

    private void calculate_match_candidates()
    {
        long s = Editor.ACTIVE ? System.nanoTime() : 0;
        candidate_counts.ensure_capacity(candidate_buffer_count);
        count_candidates_k.call(arg_long(candidate_buffer_count));
        if (Editor.ACTIVE)
        {
            long e = System.nanoTime() - s;
            Editor.queue_event("phys_match_candidates", String.valueOf(e));
        }
    }

    private int scan_single_block_candidates_out(long data_ptr, long o_data_ptr, int n)
    {
        long local_buffer_size = CLSize.cl_int * GPGPU.max_scan_block_size;

        GPGPU.cl_zero_buffer(GPGPU.cl_cmd_queue_ptr, atomic_counter_ptr, CLSize.cl_int);

        scan_candidates_single_block_out_k
            .ptr_arg(ScanCandidatesSingleBlockOut_k.Args.input, data_ptr)
            .ptr_arg(ScanCandidatesSingleBlockOut_k.Args.output, o_data_ptr)
            .ptr_arg(ScanCandidatesSingleBlockOut_k.Args.sz, atomic_counter_ptr)
            .loc_arg(ScanCandidatesSingleBlockOut_k.Args.buffer, local_buffer_size)
            .set_arg(ScanCandidatesSingleBlockOut_k.Args.n, n)
            .call(GPGPU.local_work_default, GPGPU.local_work_default);

        return GPGPU.cl_read_svm_int(GPGPU.cl_cmd_queue_ptr, atomic_counter_ptr);
    }

    private int scan_multi_block_candidates_out(long data_ptr, long o_data_ptr, int n, int k)
    {
        long local_buffer_size = CLSize.cl_int * GPGPU.max_scan_block_size;

        long gx = k * GPGPU.max_scan_block_size;
        long[] global_work_size = arg_long(gx);
        int part_size = k * 2;
        long part_buf_size = ((long) CLSize.cl_int * ((long) part_size));
        var p_data = GPGPU.cl_new_buffer(part_buf_size);

        scan_candidates_multi_block_out_k
            .ptr_arg(ScanCandidatesMultiBlockOut_k.Args.input, data_ptr)
            .ptr_arg(ScanCandidatesMultiBlockOut_k.Args.output, o_data_ptr)
            .loc_arg(ScanCandidatesMultiBlockOut_k.Args.buffer, local_buffer_size)
            .ptr_arg(ScanCandidatesMultiBlockOut_k.Args.part, p_data)
            .set_arg(ScanCandidatesMultiBlockOut_k.Args.n, n)
            .call(global_work_size, GPGPU.local_work_default);

        GPGPU.scan_int(p_data, part_size);

        GPGPU.cl_zero_buffer(GPGPU.cl_cmd_queue_ptr, atomic_counter_ptr, CLSize.cl_int);

        complete_candidates_multi_block_out_k
            .ptr_arg(CompleteCandidatesMultiBlockOut_k.Args.input, data_ptr)
            .ptr_arg(CompleteCandidatesMultiBlockOut_k.Args.output, o_data_ptr)
            .ptr_arg(CompleteCandidatesMultiBlockOut_k.Args.sz, atomic_counter_ptr)
            .loc_arg(CompleteCandidatesMultiBlockOut_k.Args.buffer, local_buffer_size)
            .ptr_arg(CompleteCandidatesMultiBlockOut_k.Args.part, p_data)
            .set_arg(CompleteCandidatesMultiBlockOut_k.Args.n, n)
            .call(global_work_size, GPGPU.local_work_default);

        GPGPU.cl_release_buffer(p_data);

        return GPGPU.cl_read_svm_int(GPGPU.cl_cmd_queue_ptr, atomic_counter_ptr);
    }

    private int scan_key_candidates(long data_ptr, long o_data_ptr, int n)
    {
        int k = GPGPU.work_group_count(n);
        if (k == 1)
        {
            return scan_single_block_candidates_out(data_ptr, o_data_ptr, n);
        }
        else
        {
            return scan_multi_block_candidates_out(data_ptr, o_data_ptr, n, k);
        }
    }

    private void calculate_match_offsets()
    {
        long s = Editor.ACTIVE ? System.nanoTime() : 0;
        candidate_offsets.ensure_capacity(candidate_buffer_count);
        match_buffer_count = scan_key_candidates(candidate_counts.pointer(), candidate_offsets.pointer(), (int) candidate_buffer_count);
        if (Editor.ACTIVE)
        {
            long e = System.nanoTime() - s;
            Editor.queue_event("phys_match_offsets", String.valueOf(e));
        }
    }

    private void aabb_collide()
    {
        long s = Editor.ACTIVE ? System.nanoTime() : 0;
        matches.ensure_capacity(match_buffer_count);
        matches_used.ensure_capacity(candidate_buffer_count);
        GPGPU.cl_zero_buffer(GPGPU.cl_cmd_queue_ptr, atomic_counter_ptr, CLSize.cl_int);
        aabb_collide_k.call(arg_long(candidate_buffer_count));
        candidate_count = GPGPU.cl_read_svm_int(GPGPU.cl_cmd_queue_ptr, atomic_counter_ptr);
        if (Editor.ACTIVE)
        {
            long e = System.nanoTime() - s;
            Editor.queue_event("phys_aabb_collide", String.valueOf(e));
        }
    }

    private void finalize_candidates()
    {
        long s = Editor.ACTIVE ? System.nanoTime() : 0;
        if (candidate_count <= 0)
        {
            return;
        }

        long buffer_size = (long) CLSize.cl_int2 * candidate_count;

        candidates.ensure_capacity(candidate_count);

        int[] counter = new int[]{0};
        var counter_ptr = GPGPU.cl_new_int_arg_buffer(counter);

        candidate_buffer_size = buffer_size;

        finalize_candidates_k
            .ptr_arg(FinalizeCandidates_k.Args.counter, counter_ptr)
            .call(arg_long(candidate_buffer_count));

        GPGPU.cl_release_buffer(counter_ptr);
        if (Editor.ACTIVE)
        {
            long e = System.nanoTime() - s;
            Editor.queue_event("phys_finalize_candidates", String.valueOf(e));
        }
    }
    //#endregion

    //#region SAT Collision

    private void sat_collide()
    {
        long s = Editor.ACTIVE ? System.nanoTime() : 0;
        int candidate_pair_size = (int) candidate_buffer_size / CLSize.cl_int2;
        long[] global_work_size = new long[]{candidate_pair_size};
        GPGPU.cl_zero_buffer(GPGPU.cl_cmd_queue_ptr, atomic_counter_ptr, CLSize.cl_int);

        long max_point_count = candidate_buffer_size
            * 2  // there are two bodies per collision pair
            * 2; // assume worst case is 2 points per body

        reactions_in.ensure_capacity(max_point_count);
        reactions_out.ensure_capacity(max_point_count);
        reaction_index.ensure_capacity(max_point_count);
        point_reaction_counts.ensure_capacity(GPGPU.core_memory.next_point());
        point_reaction_offsets.ensure_capacity(GPGPU.core_memory.next_point());
        sat_collide_k.call(global_work_size);
        reaction_count = GPGPU.cl_read_svm_int(GPGPU.cl_cmd_queue_ptr, atomic_counter_ptr);
        if (Editor.ACTIVE)
        {
            long e = System.nanoTime() - s;
            Editor.queue_event("phys_sat_collide", String.valueOf(e));
        }
    }

    private void scan_reactions()
    {
        long s = Editor.ACTIVE ? System.nanoTime() : 0;
        GPGPU.scan_int_out(point_reaction_counts.pointer(), point_reaction_offsets.pointer(), GPGPU.core_memory.next_point());
        point_reaction_counts.clear();
        if (Editor.ACTIVE)
        {
            long e = System.nanoTime() - s;
            Editor.queue_event("phys_sat_scan_reactions", String.valueOf(e));
        }
    }

    private void sort_reactions()
    {
        long s = Editor.ACTIVE ? System.nanoTime() : 0;
        sort_reactions_k.call(arg_long(reaction_count));
        if (Editor.ACTIVE)
        {
            long e = System.nanoTime() - s;
            Editor.queue_event("phys_sat_sort_reactions", String.valueOf(e));
        }
    }

    private void apply_reactions()
    {
        long s = Editor.ACTIVE ? System.nanoTime() : 0;
        apply_reactions_k.call(arg_long(GPGPU.core_memory.next_point()));
        if (Editor.ACTIVE)
        {
            long e = System.nanoTime() - s;
            Editor.queue_event("phys_sat_apply_reactions", String.valueOf(e));
        }
    }
    //#endregion

    //#region Animation

    private void animate_entities(float dt)
    {
        long s = Editor.ACTIVE ? System.nanoTime() : 0;
        animate_entities_k
            .set_arg(AnimateEntities_k.Args.delta_time, dt)
            .call(arg_long(GPGPU.core_memory.next_entity()));
        if (Editor.ACTIVE)
        {
            long e = System.nanoTime() - s;
            Editor.queue_event("phys_animate_entities", String.valueOf(e));
        }
    }

    private void animate_bones()
    {
        long s = Editor.ACTIVE ? System.nanoTime() : 0;
        animate_bones_k.call(arg_long(GPGPU.core_memory.next_hull_bone()));
        if (Editor.ACTIVE)
        {
            long e = System.nanoTime() - s;
            Editor.queue_event("phys_animate_bones", String.valueOf(e));
        }
    }

    private void animate_points()
    {
        long s = Editor.ACTIVE ? System.nanoTime() : 0;
        animate_points_k.call(arg_long(GPGPU.core_memory.next_point()));
        if (Editor.ACTIVE)
        {
            long e = System.nanoTime() - s;
            Editor.queue_event("phys_animate_points", String.valueOf(e));
        }
    }

    //#endregion

    //#region Constraint Resolvers

    private void resolve_constraints(int steps)
    {
        long s = Editor.ACTIVE ? System.nanoTime() : 0;
        boolean last_step;
        for (int i = 0; i < steps; i++)
        {
            last_step = i == steps - 1;
            int n = last_step
                ? 1
                : 0;

            resolve_constraints_k
                .set_arg(ResolveConstraints_k.Args.process_all, n)
                .call(arg_long(GPGPU.core_memory.next_hull()));
        }
        if (Editor.ACTIVE)
        {
            long e = System.nanoTime() - s;
            Editor.queue_event("phys_resolve_constraints", String.valueOf(e));
        }
    }

    private void resolve_entities()
    {
        long s = Editor.ACTIVE ? System.nanoTime() : 0;
        move_entities_k.call(arg_long(GPGPU.core_memory.next_entity()));
        if (Editor.ACTIVE)
        {
            long e = System.nanoTime() - s;
            Editor.queue_event("phys_move_entities", String.valueOf(e));
        }
    }

    private void resolve_hulls()
    {
        long s = Editor.ACTIVE ? System.nanoTime() : 0;
        move_hulls_k.call(arg_long(GPGPU.core_memory.next_hull()));
        if (Editor.ACTIVE)
        {
            long e = System.nanoTime() - s;
            Editor.queue_event("phys_move_hulls", String.valueOf(e));
        }
    }

    //#endregion

    //#region Simulation Functions

    /**
     * This is the core of the physics simulation. Upon return from this method, the simulation is
     * advanced one tick. Note that this class uses a fixed time step, so the time delta should always
     * be the same. Most work done within this method is delegated to the GPU for performance.
     */
    private void tick_simulation()
    {
        // The first order of business is to perform the mathematical steps required to calculate where the
        // individual points of each hull currently are. When this call returns, all tracked physics objects
        // will be in their new locations, and points will have their current and previous location values
        // updated for this tick cycle.
        integrate();

        /*
        - Broad Phase Collision -
        =========================
        Before the final and more computationally expensive collision checks are performed, A broad phase check
        is done to narrow down the potential collision candidates. Because this is implemented using parallelized
        compute kernels, the process is more verbose than a traditional CPU based approach. At a high level, this
        process is used in place of more complex constructs like Map<> and Set<>, with capacities of backing
        structures being pre-computed to fulfill the fixed memory size requirements of the GPU kernel.

        There are three top-level "conceptual" structures, a key bank, a key map, and the key itself.

        Keys in this context are simply two-dimensional integer vectors that point to a "cell" of the uniform
        grid, which is a structure that imposes a coarse grid over the viewable area of the screen. For every hull,
        if it is within view, it will be inside, or overlapping one or more of these cells. A "key" value simply
        describes this cell location.

        The key bank is a large block of memory that holds the actual key data for each object. Objects with entries
        in the key bank will have their corresponding key bank tables updated to point to their start and offset
        within this key bank. It is recomputed every tick, because the values and number of keys change depending
        on object location and orientation. Objects that are off-screen are handled such that they always have empty
        key banks, removing them from consideration before the broad phase even starts.

        The key map is a structure that maps each individual spatial key to the objects that have that key within
        their local key bank. Or to put it another way, every object that is touching the cell associated with a
        particular key, has its index stored in this map under that key. This makes it possible to then query
        the map by key, effectively getting a list of all objects that could be colliding with other objects,
        by virtue of the fact that they share a key.
        */

        // The first task before checking boundaries is to calculate the bank offsets for this frame. These offsets
        // determine how much space in the key bank is allocated to each possible collision candidate. The amount of
        // space varies based on the size and orientation of the object within the uniform grid.
        calculate_bank_offsets();

        // As a fail-safe, if the total bank size is zero, it means there's no tracked objects, so simply return.
        // This condition is unlikely to occur accept when the simulation is first starting up.
        if (uniform_grid.get_key_bank_size() == 0)
        {
            return;
        }

        // Once we know there are some objects to track, we can generate the keys needed to further process
        // the tracked objects. This call generates the keys for each object, and stores them in the global
        // key bank. Hull bounds tables are updated with the correct offsets and counts as needed.
        generate_keys();

        // After keys are generated, the next step is to calculate the space needed for the key map. This is
        // a similar process to calculating the bank offsets. The buffer is zeroed before use to clear out
        // data that is present after the previous tick.
        GPGPU.cl_zero_buffer(GPGPU.cl_cmd_queue_ptr, offsets_data_ptr, grid_buffer_size);
        GPGPU.scan_int_out(counts_data_ptr, offsets_data_ptr, uniform_grid.directory_length);

        // Now, the keymap itself is built. This is the structure that provides the ability to query
        // objects within the uniform grid structure.
        build_key_map(uniform_grid);

        // Hulls are now filtered to ensure that only objects that are within the uniform grid boundary
        // are considered for collisions. In this step, the maximum size of the match table is calculated
        // as well, which is needed in subsequent steps.
        locate_in_bounds();

        // In the first pass, the number of total possible candidates is calculated for each hull. This is
        // necessary to correctly determine how much of the table each hull will require.
        calculate_match_candidates();

        if (Editor.ACTIVE)
        {
            Editor.queue_event("phys_match_buffer_count", String.valueOf(match_buffer_count));
        }

        // In a second pass, candidate counts are scanned to determine the offsets into the match table that
        // correspond to each hull that will be checked for collisions.
        calculate_match_offsets();
        if (match_buffer_count > 100_000_000)
        {
            throw new RuntimeException("collision buffer too large:" + match_buffer_count);
        }

        // Finally, the actual broad phase collision check is performed. Once complete, the match table will
        // be filled in with all matches. There may be some unused sections of the table, because some objects
        // may be eliminated during the check.
        aabb_collide();

        // This last step cleans up the match table, retaining only the used sections of the buffer.
        // After this step, the matches are ready for the narrow phase check.
        finalize_candidates();

        // If there were no candidate collisions, there's nothing left to do
        if (candidate_count <= 0)
        {
            return;
        }

        /*
        - Narrow Phase Collision/Reaction -
        ===================================
        Objects found to be close enough for a narrow check are now fully examined to determine if they are
        actually colliding. Any collisions that are detected will have reactions calculated and forwarded
        to a series of kernels that work together to scan and sort the reactions, then ultimately apply them
        to the appropriate points.

        Collision is detected using the separating axis theorem for all collisions that involve polygons.
        Circle-to-circle collisions are handled using a simple distance/radius check. Because of this, circles
        are significantly less demanding to simulate.

        After all collision reactions have been applied, there is a final step that applies all hull movements
        to their parent entities. This last step is needed for complex models, to ensure the groups of hulls
        are moved together as a unit. Without this step, armature based entities will not collide correctly.
        */

        // Using the candidates generated by the AABB checks, we now do a full collision check. For any objects
        // that are found to be colliding, appropriate reaction vectors are generated and stored.
        sat_collide();

        // It is possible that after the check, no objects are found to be colliding. If that happens, exit.
        if (reaction_count == 0)
        {
            return;
        }

        // Since we did have some reactions, we need to figure out what points were affected. This is needed
        // so that reactions can be accumulated in series and applied to the point they affect.
        scan_reactions();

        // After the initial scan, the reaction buffers are sorted to match the layout computed in the scan
        // step. After this call, the buffers are in ascending order by point index.
        sort_reactions();

        // Now all points with reactions are able to sum all their reactions and apply them, as well as
        // enforcing constraints on the velocities of the affected points.
        apply_reactions();
    }

    private void process_world_buffer()
    {
        long sd = Editor.ACTIVE ? System.nanoTime() : 0;
        GPGPU.core_memory.process_world_buffer();
        if (Editor.ACTIVE)
        {
            long e = System.nanoTime() - sd;
            Editor.queue_event("sector_load", String.valueOf(e));
        }
    }

    private void simulate(float dt)
    {
        long s = Editor.ACTIVE ? System.nanoTime() : 0;

        // An initial constraint solve pass is done before simulation to ensure edges are in their "safe"
        // convex shape. Animations may move points into positions where the geometry becomes concave,
        // so this call prevents collision errors due to non-convex shapes.
        resolve_constraints(EDGE_STEPS);

        this.time_accumulator += dt;
        int sub_ticks = 0;
        float dropped_time = 0.0f;
        while (this.time_accumulator >= TICK_RATE)
        {
            for (int i = 0; i < TARGET_SUB_STEPS; i++)
            {
                sub_ticks++;

                // if we end up doing more sub ticks than is ideal, we will avoid ticking the simulation anymore
                // for this frame. This forces slower hardware to slow down a bit, which is less than ideal, but
                // is better than the alternative, which is system lockup.
                if (sub_ticks <= MAX_SUB_STEPS)
                {
                    this.time_accumulator -= FIXED_TIME_STEP;

                    // Before the GPU begins the step cycle, player input is handled and the memory structures
                    // in the GPU are updated with the proper values.
                    update_controllable_entities();

                    // perform one tick of the simulation
                    this.tick_simulation();

                    resolve_constraints(1);

                    resolve_hulls();

                    // Once all points have been relocated, all hulls are in their required positions for this frame.
                    // Movements applied to hulls are now accumulated and applied to their parent entities.
                    resolve_entities();

                    // Now we make a call to animate the vertices of bone-tracked hulls. This ensures that all tracked
                    // objects that have animation will have their hulls moved into position for the current tick. It
                    // may seem odd to process animations as part of physics and not rendering, however this is required
                    // as the animated objects need to be accounted for in physical space. The hulls representing the
                    // rendered meshes are what is actually moved, and the result of the hull movement is used to position
                    // the original mesh for rendering. This separation is necessary as model geometry is too complex to
                    // use as a collision boundary.
                    animate_points();

                    // Once positions are adjusted, edge constraints are enforced to ensure that rigid bodies maintain
                    // their defined shapes. Without this step, the individual points of the tracked physics hulls will
                    // deform on impact, possibly to non-convex shapes and typically causing simulation failure. The
                    // number of steps that are performed each tick has an impact on the accuracy of the hull boundaries
                    // within the simulation.
                    resolve_constraints(EDGE_STEPS);
                }
                else
                {
                    dropped_time = this.time_accumulator;
                    if (Editor.ACTIVE)
                    {
                        Editor.queue_event("dropped", String.format("%f", dropped_time));
                    }
                    this.time_accumulator = 0;
                }
            }
        }

        // TODO: read back mouse colliding objects here

        // zero out the acceleration buffer, so it is empty for the next frame
        GPGPU.core_memory.buffer(BufferType.ENTITY_ACCEL).clear();

        long sd = Editor.ACTIVE ? System.nanoTime() : 0;

        // Deletion of objects happens only once per simulation tick, instead of every sub-step
        // to ensure buffer compaction happens as infrequently as possible.
        GPGPU.core_memory.delete_and_compact();

        if (Editor.ACTIVE)
        {
            long e = System.nanoTime() - sd;
            Editor.queue_event("phys_compact", String.valueOf(e));
        }


        // Armatures and bones are animated once per time tick, after all simulation is done for this pass. The interplay between
        // animation and edge constraints may leave points in slightly incorrect positions. Animating here ensures the rendering
        // step always sees the objects exactly in their correct positions.
        animate_entities(dt - dropped_time);
        animate_bones();
        animate_points();

        if (Editor.ACTIVE)
        {
            long e = System.nanoTime() - s;
            Editor.queue_event("phys_cycle", String.valueOf(e));
        }
    }

    //#endregion

    //#region Public API

    @Override
    public void tick(float dt)
    {
        try
        {
            clFinish(GPGPU.gl_cmd_queue_ptr);
            long phys_time = last_phys_time.take();
            GPGPU.core_memory.await_sector();
            GPGPU.core_memory.reset_sector();
            clFinish(GPGPU.sector_cmd_queue_ptr);
            process_world_buffer();
            GPGPU.core_memory.mirror_buffers_ex();
            clFinish(GPGPU.cl_cmd_queue_ptr);
            next_phys_time.put(dt);

            if (Editor.ACTIVE)
            {
                Editor.queue_event("phys", String.valueOf(phys_time));
            }
        }
        catch (InterruptedException e)
        {
            throw new RuntimeException(e);
        }
    }

    @Override
    public void shutdown()
    {
        physics_simulation.interrupt();
        try
        {
            physics_simulation.join();
        }
        catch (InterruptedException e)
        {
            throw new RuntimeException(e);
        }

        control_entities.destroy();
        integrate.destroy();
        scan_key_bank.destroy();
        generate_keys.destroy();
        build_key_map.destroy();
        locate_in_bounds.destroy();
        scan_key_candidates.destroy();
        aabb_collide.destroy();
        sat_collide.destroy();
        animate_hulls.destroy();
        resolve_constraints.destroy();

        point_reaction_counts.release();
        point_reaction_offsets.release();
        reactions_in.release();
        reactions_out.release();
        reaction_index.release();
        key_map.release();
        key_bank.release();
        in_bounds.release();
        candidates.release();
        candidate_counts.release();
        candidate_offsets.release();
        matches.release();
        matches_used.release();
        control_point_flags.release();
        control_point_indices.release();
        control_point_tick_budgets.release();
        control_point_linear_mag.release();
        control_point_jump_mag.release();

        debug();

        GPGPU.cl_release_buffer(atomic_counter_ptr);
        GPGPU.cl_release_buffer(counts_data_ptr);
        GPGPU.cl_release_buffer(offsets_data_ptr);
    }

    //#endregion

    private void debug()
    {
        long total = 0;
        total += point_reaction_counts.debug_data();
        total += point_reaction_offsets.debug_data();
        total += reactions_in.debug_data();
        total += reactions_out.debug_data();
        total += reaction_index.debug_data();
        total += key_map.debug_data();
        total += key_bank.debug_data();
        total += in_bounds.debug_data();
        total += candidates.debug_data();
        total += candidate_counts.debug_data();
        total += candidate_offsets.debug_data();
        total += matches.debug_data();
        total += matches_used.debug_data();
        total += control_point_flags.debug_data();
        total += control_point_indices.debug_data();
        total += control_point_tick_budgets.debug_data();
        total += control_point_linear_mag.debug_data();
        total += control_point_jump_mag.debug_data();

        //System.out.println("---------------------------");
        System.out.println("Physics Memory Usage: MB " + ((float) total / 1024f / 1024f));
    }
}
