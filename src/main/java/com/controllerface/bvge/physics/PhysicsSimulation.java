package com.controllerface.bvge.physics;

import com.controllerface.bvge.cl.*;
import com.controllerface.bvge.cl.buffers.*;
import com.controllerface.bvge.cl.kernels.*;
import com.controllerface.bvge.cl.programs.*;
import com.controllerface.bvge.ecs.ECS;
import com.controllerface.bvge.ecs.components.*;
import com.controllerface.bvge.ecs.systems.GameSystem;
import com.controllerface.bvge.editor.Editor;
import com.controllerface.bvge.util.Constants;
import com.controllerface.bvge.window.Window;

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

    private final GPUProgram p_control_entities    = new ControlEntities();
    private final GPUProgram p_integrate           = new Integrate();
    private final GPUProgram p_scan_key_bank       = new ScanKeyBank();
    private final GPUProgram p_generate_keys       = new GenerateKeys();
    private final GPUProgram p_build_key_map       = new BuildKeyMap();
    private final GPUProgram p_locate_in_bounds    = new LocateInBounds();
    private final GPUProgram p_scan_key_candidates = new ScanKeyCandidates();
    private final GPUProgram p_aabb_collide        = new AabbCollide();
    private final GPUProgram p_sat_collide         = new SatCollide();
    private final GPUProgram p_animate_hulls       = new AnimateHulls();
    private final GPUProgram p_resolve_constraints = new ResolveConstraints();

    private final GPUKernel k_aabb_collide;
    private final GPUKernel k_animate_bones;
    private final GPUKernel k_animate_entities;
    private final GPUKernel k_animate_points;
    private final GPUKernel k_apply_reactions;
    private final GPUKernel k_build_key_map;
    private final GPUKernel k_complete_bounds_multi_block;
    private final GPUKernel k_complete_candidates_multi_block_out;
    private final GPUKernel k_count_candidates;
    private final GPUKernel k_finalize_candidates;
    private final GPUKernel k_generate_keys;
    private final GPUKernel k_handle_movement;
    private final GPUKernel k_integrate;
    private final GPUKernel k_integrate_entities;
    private final GPUKernel k_locate_in_bounds;
    private final GPUKernel k_move_entities;
    private final GPUKernel k_move_hulls;
    private final GPUKernel k_resolve_constraints;
    private final GPUKernel k_sat_collide;
    private final GPUKernel k_scan_bounds_multi_block;
    private final GPUKernel k_scan_bounds_single_block;
    private final GPUKernel k_scan_candidates_multi_block_out;
    private final GPUKernel k_scan_candidates_single_block_out;
    private final GPUKernel k_set_control_points;
    private final GPUKernel k_sort_reactions;

    //#endregion

    //#region Buffers & Counters

    private final long grid_buffer_size;
    private final long ptr_counts_data;
    private final long ptr_offsets_data;
    private final long svm_atomic_counter;

    public final ResizableBuffer b_point_reaction_counts;
    public final ResizableBuffer b_point_reaction_offsets;
    public final ResizableBuffer b_reactions_in;
    public final ResizableBuffer b_reactions_out;
    public final ResizableBuffer b_reaction_index;
    public final ResizableBuffer b_key_map;
    public final ResizableBuffer b_key_bank;
    public final ResizableBuffer b_in_bounds;
    public final ResizableBuffer b_candidates;
    public final ResizableBuffer b_candidate_counts;
    public final ResizableBuffer b_candidate_offsets;
    public final ResizableBuffer b_matches;
    public final ResizableBuffer b_matches_used;
    public final ResizableBuffer b_control_point_flags;
    public final ResizableBuffer b_control_point_indices;
    public final ResizableBuffer b_control_point_tick_budgets;
    public final ResizableBuffer b_control_point_linear_mag;
    public final ResizableBuffer b_control_point_jump_mag;

    private long candidate_count        = 0;
    private long reaction_count         = 0;
    private long candidate_buffer_size  = 0;
    private long match_buffer_count     = 0;
    private long candidate_buffer_count = 0;

    //#endregion

    //#region Simulation Data

    private final UniformGrid uniform_grid;

    private float time_accumulator = 0.0f;

    //#endregion

    //#region Thread & Sync

    private final BlockingQueue<Float> next_phys_time = new SynchronousQueue<>();
    private final BlockingQueue<Long> last_phys_time  = new SynchronousQueue<>();
    private final GPUScanScalarInt gpu_int_scan;
    private final GPUScanScalarIntOut gpu_int_scan_out;

    private final Thread physics_simulation = Thread.ofVirtual().name("Physics-Simulation").start(() ->
    {
        try { last_phys_time.put(0L); }
        catch (InterruptedException e) { throw new RuntimeException(e); }

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
        gpu_int_scan     = new GPUScanScalarInt(GPGPU.ptr_compute_queue);
        gpu_int_scan_out = new GPUScanScalarIntOut(GPGPU.ptr_compute_queue, gpu_int_scan);

        grid_buffer_size = (long) CLSize.cl_int * this.uniform_grid.directory_length;

        svm_atomic_counter = GPGPU.cl_new_pinned_int();
        ptr_counts_data    = GPGPU.cl_new_buffer(grid_buffer_size);
        ptr_offsets_data   = GPGPU.cl_new_buffer(grid_buffer_size);

        b_point_reaction_counts      = new TransientBuffer(GPGPU.ptr_compute_queue, CLSize.cl_int, 500_000L);
        b_point_reaction_offsets     = new TransientBuffer(GPGPU.ptr_compute_queue, CLSize.cl_int, 500_000L);
        b_reactions_in               = new TransientBuffer(GPGPU.ptr_compute_queue, CLSize.cl_float8, 500_000L);
        b_reactions_out              = new TransientBuffer(GPGPU.ptr_compute_queue, CLSize.cl_float8, 500_000L);
        b_reaction_index             = new TransientBuffer(GPGPU.ptr_compute_queue, CLSize.cl_int, 500_000L);
        b_key_map                    = new TransientBuffer(GPGPU.ptr_compute_queue, CLSize.cl_int, 500_000L);
        b_key_bank                   = new TransientBuffer(GPGPU.ptr_compute_queue, CLSize.cl_int, 500_000L);
        b_in_bounds                  = new TransientBuffer(GPGPU.ptr_compute_queue, CLSize.cl_int, 500_000L);
        b_candidates                 = new TransientBuffer(GPGPU.ptr_compute_queue, CLSize.cl_int2, 500_000L);
        b_candidate_counts           = new TransientBuffer(GPGPU.ptr_compute_queue, CLSize.cl_int2, 500_000L);
        b_candidate_offsets          = new TransientBuffer(GPGPU.ptr_compute_queue, CLSize.cl_int, 500_000L);
        b_matches                    = new TransientBuffer(GPGPU.ptr_compute_queue, CLSize.cl_int, 500_000L);
        b_matches_used               = new TransientBuffer(GPGPU.ptr_compute_queue, CLSize.cl_int, 500_000L);

        b_control_point_flags        = new PersistentBuffer(GPGPU.ptr_compute_queue, CLSize.cl_int, 1);
        b_control_point_indices      = new PersistentBuffer(GPGPU.ptr_compute_queue, CLSize.cl_int, 1);
        b_control_point_tick_budgets = new PersistentBuffer(GPGPU.ptr_compute_queue, CLSize.cl_int, 1);
        b_control_point_linear_mag   = new PersistentBuffer(GPGPU.ptr_compute_queue, CLSize.cl_float, 1);
        b_control_point_jump_mag     = new PersistentBuffer(GPGPU.ptr_compute_queue, CLSize.cl_float, 1);

        b_control_point_flags.ensure_capacity(1);
        b_control_point_indices.ensure_capacity(1);
        b_control_point_tick_budgets.ensure_capacity(1);
        b_control_point_linear_mag.ensure_capacity(1);
        b_control_point_jump_mag.ensure_capacity(1);

        p_control_entities.init();
        p_integrate.init();
        p_scan_key_bank.init();
        p_generate_keys.init();
        p_build_key_map.init();
        p_locate_in_bounds.init();
        p_scan_key_candidates.init();
        p_aabb_collide.init();
        p_sat_collide.init();
        p_animate_hulls.init();
        p_resolve_constraints.init();

        long k_ptr_set_control_points = p_control_entities.kernel_ptr(Kernel.set_control_points);
        k_set_control_points = new SetControlPoints_k(GPGPU.ptr_compute_queue, k_ptr_set_control_points)
            .buf_arg(SetControlPoints_k.Args.flags,      b_control_point_flags)
            .buf_arg(SetControlPoints_k.Args.indices,    b_control_point_indices)
            .buf_arg(SetControlPoints_k.Args.linear_mag, b_control_point_linear_mag)
            .buf_arg(SetControlPoints_k.Args.jump_mag,   b_control_point_jump_mag);

        long k_ptr_handle_movements = p_control_entities.kernel_ptr(Kernel.handle_movement);
        k_handle_movement = new HandleMovement_k(GPGPU.ptr_compute_queue, k_ptr_handle_movements)
            .buf_arg(HandleMovement_k.Args.entities,                 GPGPU.core_memory.get_buffer(CoreBufferType.ENTITY))
            .buf_arg(HandleMovement_k.Args.entity_accel,             GPGPU.core_memory.get_buffer(CoreBufferType.ENTITY_ACCEL))
            .buf_arg(HandleMovement_k.Args.entity_motion_states,     GPGPU.core_memory.get_buffer(CoreBufferType.ENTITY_MOTION_STATE))
            .buf_arg(HandleMovement_k.Args.entity_flags,             GPGPU.core_memory.get_buffer(CoreBufferType.ENTITY_FLAG))
            .buf_arg(HandleMovement_k.Args.entity_animation_indices, GPGPU.core_memory.get_buffer(CoreBufferType.ENTITY_ANIM_INDEX))
            .buf_arg(HandleMovement_k.Args.entity_animation_elapsed, GPGPU.core_memory.get_buffer(CoreBufferType.ENTITY_ANIM_ELAPSED))
            .buf_arg(HandleMovement_k.Args.entity_animation_blend,   GPGPU.core_memory.get_buffer(CoreBufferType.ENTITY_ANIM_BLEND))
            .buf_arg(HandleMovement_k.Args.flags,                    b_control_point_flags)
            .buf_arg(HandleMovement_k.Args.indices,                  b_control_point_indices)
            .buf_arg(HandleMovement_k.Args.tick_budgets,             b_control_point_tick_budgets)
            .buf_arg(HandleMovement_k.Args.linear_mag,               b_control_point_linear_mag)
            .buf_arg(HandleMovement_k.Args.jump_mag,                 b_control_point_jump_mag);

        long k_ptr_integrate = p_integrate.kernel_ptr(Kernel.integrate);
        k_integrate = new Integrate_k(GPGPU.ptr_compute_queue, k_ptr_integrate)
            .buf_arg(Integrate_k.Args.hulls,             GPGPU.core_memory.get_buffer(CoreBufferType.HULL))
            .buf_arg(Integrate_k.Args.hull_scales,       GPGPU.core_memory.get_buffer(CoreBufferType.HULL_SCALE))
            .buf_arg(Integrate_k.Args.hull_point_tables, GPGPU.core_memory.get_buffer(CoreBufferType.HULL_POINT_TABLE))
            .buf_arg(Integrate_k.Args.entity_accel,      GPGPU.core_memory.get_buffer(CoreBufferType.ENTITY_ACCEL))
            .buf_arg(Integrate_k.Args.hull_rotations,    GPGPU.core_memory.get_buffer(CoreBufferType.HULL_ROTATION))
            .buf_arg(Integrate_k.Args.points,            GPGPU.core_memory.get_buffer(CoreBufferType.POINT))
            .buf_arg(Integrate_k.Args.point_hit_counts,  GPGPU.core_memory.get_buffer(CoreBufferType.POINT_HIT_COUNT))
            .buf_arg(Integrate_k.Args.point_flags,       GPGPU.core_memory.get_buffer(CoreBufferType.POINT_FLAG))
            .buf_arg(Integrate_k.Args.bounds,            GPGPU.core_memory.get_buffer(CoreBufferType.HULL_AABB))
            .buf_arg(Integrate_k.Args.bounds_index_data, GPGPU.core_memory.get_buffer(CoreBufferType.HULL_AABB_INDEX))
            .buf_arg(Integrate_k.Args.bounds_bank_data,  GPGPU.core_memory.get_buffer(CoreBufferType.HULL_AABB_KEY_TABLE))
            .buf_arg(Integrate_k.Args.hull_flags,        GPGPU.core_memory.get_buffer(CoreBufferType.HULL_FLAG))
            .buf_arg(Integrate_k.Args.hull_entity_ids,   GPGPU.core_memory.get_buffer(CoreBufferType.HULL_ENTITY_ID))
            .buf_arg(Integrate_k.Args.anti_gravity,      GPGPU.core_memory.get_buffer(CoreBufferType.POINT_ANTI_GRAV));

        long k_ptr_integrate_entities = p_integrate.kernel_ptr(Kernel.integrate_entities);
        k_integrate_entities = new IntegrateEntities_k(GPGPU.ptr_compute_queue, k_ptr_integrate_entities)
            .buf_arg(IntegrateEntities_k.Args.entities,          GPGPU.core_memory.get_buffer(CoreBufferType.ENTITY))
            .buf_arg(IntegrateEntities_k.Args.entity_flags,      GPGPU.core_memory.get_buffer(CoreBufferType.ENTITY_FLAG))
            .buf_arg(IntegrateEntities_k.Args.entity_root_hulls, GPGPU.core_memory.get_buffer(CoreBufferType.ENTITY_ROOT_HULL))
            .buf_arg(IntegrateEntities_k.Args.entity_accel,      GPGPU.core_memory.get_buffer(CoreBufferType.ENTITY_ACCEL))
            .buf_arg(IntegrateEntities_k.Args.hull_flags,        GPGPU.core_memory.get_buffer(CoreBufferType.HULL_FLAG));

        long k_ptr_generate_keys = p_generate_keys.kernel_ptr(Kernel.generate_keys);
        k_generate_keys = new GenerateKeys_k(GPGPU.ptr_compute_queue, k_ptr_generate_keys)
            .buf_arg(GenerateKeys_k.Args.key_bank, b_key_bank)
            .buf_arg(GenerateKeys_k.Args.bounds_index_data, GPGPU.core_memory.get_buffer(CoreBufferType.HULL_AABB_INDEX))
            .buf_arg(GenerateKeys_k.Args.bounds_bank_data,  GPGPU.core_memory.get_buffer(CoreBufferType.HULL_AABB_KEY_TABLE))
            .ptr_arg(GenerateKeys_k.Args.key_counts,        ptr_counts_data)
            .set_arg(GenerateKeys_k.Args.x_subdivisions,    uniform_grid.x_subdivisions)
            .set_arg(GenerateKeys_k.Args.key_count_length,  uniform_grid.directory_length);

        long k_ptr_build_key_map = p_build_key_map.kernel_ptr(Kernel.build_key_map);
        k_build_key_map = new BuildKeyMap_k(GPGPU.ptr_compute_queue, k_ptr_build_key_map)
            .buf_arg(BuildKeyMap_k.Args.key_map,           b_key_map)
            .buf_arg(BuildKeyMap_k.Args.bounds_index_data, GPGPU.core_memory.get_buffer(CoreBufferType.HULL_AABB_INDEX))
            .buf_arg(BuildKeyMap_k.Args.bounds_bank_data,  GPGPU.core_memory.get_buffer(CoreBufferType.HULL_AABB_KEY_TABLE))
            .ptr_arg(BuildKeyMap_k.Args.key_offsets,       ptr_offsets_data)
            .ptr_arg(BuildKeyMap_k.Args.key_counts,        ptr_counts_data)
            .set_arg(BuildKeyMap_k.Args.x_subdivisions,    uniform_grid.x_subdivisions)
            .set_arg(BuildKeyMap_k.Args.key_count_length,  uniform_grid.directory_length);

        long k_ptr_locate_in_bounds = p_locate_in_bounds.kernel_ptr(Kernel.locate_in_bounds);
        k_locate_in_bounds = (new LocateInBounds_k(GPGPU.ptr_compute_queue, k_ptr_locate_in_bounds))
            .buf_arg(LocateInBounds_k.Args.in_bounds,        b_in_bounds)
            .buf_arg(LocateInBounds_k.Args.bounds_bank_data, GPGPU.core_memory.get_buffer(CoreBufferType.HULL_AABB_KEY_TABLE));

        long k_ptr_count_candidates = p_locate_in_bounds.kernel_ptr(Kernel.count_candidates);
        k_count_candidates = new CountCandidates_k(GPGPU.ptr_compute_queue, k_ptr_count_candidates)
            .buf_arg(CountCandidates_k.Args.candidates,       b_candidate_counts)
            .buf_arg(CountCandidates_k.Args.key_bank,         b_key_bank)
            .buf_arg(CountCandidates_k.Args.in_bounds,        b_in_bounds)
            .buf_arg(CountCandidates_k.Args.bounds_bank_data, GPGPU.core_memory.get_buffer(CoreBufferType.HULL_AABB_KEY_TABLE))
            .ptr_arg(CountCandidates_k.Args.key_counts,       ptr_counts_data)
            .set_arg(CountCandidates_k.Args.x_subdivisions,   uniform_grid.x_subdivisions)
            .set_arg(CountCandidates_k.Args.key_count_length, uniform_grid.directory_length);

        long k_ptr_aabb_collide = p_aabb_collide.kernel_ptr(Kernel.aabb_collide);
        k_aabb_collide = new AABBCollide_k(GPGPU.ptr_compute_queue, k_ptr_aabb_collide)
            .buf_arg(AABBCollide_k.Args.used,             b_matches_used)
            .buf_arg(AABBCollide_k.Args.matches,          b_matches)
            .buf_arg(AABBCollide_k.Args.match_offsets,    b_candidate_offsets)
            .buf_arg(AABBCollide_k.Args.candidates,       b_candidate_counts)
            .buf_arg(AABBCollide_k.Args.key_map,          b_key_map)
            .buf_arg(AABBCollide_k.Args.key_bank,         b_key_bank)
            .buf_arg(AABBCollide_k.Args.bounds,           GPGPU.core_memory.get_buffer(CoreBufferType.HULL_AABB))
            .buf_arg(AABBCollide_k.Args.bounds_bank_data, GPGPU.core_memory.get_buffer(CoreBufferType.HULL_AABB_KEY_TABLE))
            .buf_arg(AABBCollide_k.Args.hull_entity_ids,  GPGPU.core_memory.get_buffer(CoreBufferType.HULL_ENTITY_ID))
            .buf_arg(AABBCollide_k.Args.hull_flags,       GPGPU.core_memory.get_buffer(CoreBufferType.HULL_FLAG))
            .ptr_arg(AABBCollide_k.Args.key_counts,       ptr_counts_data)
            .ptr_arg(AABBCollide_k.Args.key_offsets,      ptr_offsets_data)
            .ptr_arg(AABBCollide_k.Args.counter,          svm_atomic_counter)
            .set_arg(AABBCollide_k.Args.x_subdivisions,   uniform_grid.x_subdivisions)
            .set_arg(AABBCollide_k.Args.key_count_length, uniform_grid.directory_length);

        long k_ptr_finalize_candidates = p_locate_in_bounds.kernel_ptr(Kernel.finalize_candidates);
        k_finalize_candidates = new FinalizeCandidates_k(GPGPU.ptr_compute_queue, k_ptr_finalize_candidates)
            .buf_arg(FinalizeCandidates_k.Args.used,             b_matches_used)
            .buf_arg(FinalizeCandidates_k.Args.matches,          b_matches)
            .buf_arg(FinalizeCandidates_k.Args.match_offsets,    b_candidate_offsets)
            .buf_arg(FinalizeCandidates_k.Args.input_candidates, b_candidate_counts)
            .buf_arg(FinalizeCandidates_k.Args.final_candidates, b_candidates);

        long k_ptr_sat_collide = p_sat_collide.kernel_ptr(Kernel.sat_collide);
        k_sat_collide = new SatCollide_k(GPGPU.ptr_compute_queue, k_ptr_sat_collide)
            .buf_arg(SatCollide_k.Args.hulls,                   GPGPU.core_memory.get_buffer(CoreBufferType.HULL))
            .buf_arg(SatCollide_k.Args.hull_scales,             GPGPU.core_memory.get_buffer(CoreBufferType.HULL_SCALE))
            .buf_arg(SatCollide_k.Args.hull_frictions,          GPGPU.core_memory.get_buffer(CoreBufferType.HULL_FRICTION))
            .buf_arg(SatCollide_k.Args.hull_restitutions,       GPGPU.core_memory.get_buffer(CoreBufferType.HULL_RESTITUTION))
            .buf_arg(SatCollide_k.Args.hull_integrity,          GPGPU.core_memory.get_buffer(CoreBufferType.HULL_INTEGRITY))
            .buf_arg(SatCollide_k.Args.hull_point_tables,       GPGPU.core_memory.get_buffer(CoreBufferType.HULL_POINT_TABLE))
            .buf_arg(SatCollide_k.Args.hull_edge_tables,        GPGPU.core_memory.get_buffer(CoreBufferType.HULL_EDGE_TABLE))
            .buf_arg(SatCollide_k.Args.hull_entity_ids,         GPGPU.core_memory.get_buffer(CoreBufferType.HULL_ENTITY_ID))
            .buf_arg(SatCollide_k.Args.hull_flags,              GPGPU.core_memory.get_buffer(CoreBufferType.HULL_FLAG))
            .buf_arg(SatCollide_k.Args.point_flags,             GPGPU.core_memory.get_buffer(CoreBufferType.POINT_FLAG))
            .buf_arg(SatCollide_k.Args.points,                  GPGPU.core_memory.get_buffer(CoreBufferType.POINT))
            .buf_arg(SatCollide_k.Args.edges,                   GPGPU.core_memory.get_buffer(CoreBufferType.EDGE))
            .buf_arg(SatCollide_k.Args.edge_flags,              GPGPU.core_memory.get_buffer(CoreBufferType.EDGE_FLAG))
            .buf_arg(SatCollide_k.Args.masses,                  GPGPU.core_memory.get_buffer(CoreBufferType.ENTITY_MASS))
            .buf_arg(SatCollide_k.Args.entity_model_transforms, GPGPU.core_memory.get_buffer(CoreBufferType.ENTITY_TRANSFORM_ID))
            .buf_arg(SatCollide_k.Args.entity_flags,            GPGPU.core_memory.get_buffer(CoreBufferType.ENTITY_FLAG))
            .buf_arg(SatCollide_k.Args.candidates,              b_candidates)
            .buf_arg(SatCollide_k.Args.reactions,               b_reactions_in)
            .buf_arg(SatCollide_k.Args.reaction_index,          b_reaction_index)
            .buf_arg(SatCollide_k.Args.point_reactions,         b_point_reaction_counts)
            .ptr_arg(SatCollide_k.Args.counter,                 svm_atomic_counter)
            .set_arg(SatCollide_k.Args.dt,                      FIXED_TIME_STEP);

        long k_ptr_sort_reactions = p_sat_collide.kernel_ptr(Kernel.sort_reactions);
        k_sort_reactions = new SortReactions_k(GPGPU.ptr_compute_queue, k_ptr_sort_reactions)
            .buf_arg(SortReactions_k.Args.reactions_in,    b_reactions_in)
            .buf_arg(SortReactions_k.Args.reactions_out,   b_reactions_out)
            .buf_arg(SortReactions_k.Args.reaction_index,  b_reaction_index)
            .buf_arg(SortReactions_k.Args.point_reactions, b_point_reaction_counts)
            .buf_arg(SortReactions_k.Args.point_offsets,   b_point_reaction_offsets);

        long k_ptr_apply_reactions = p_sat_collide.kernel_ptr(Kernel.apply_reactions);
        k_apply_reactions = new ApplyReactions_k(GPGPU.ptr_compute_queue, k_ptr_apply_reactions)
            .buf_arg(ApplyReactions_k.Args.reactions,          b_reactions_out)
            .buf_arg(ApplyReactions_k.Args.point_reactions,    b_point_reaction_counts)
            .buf_arg(ApplyReactions_k.Args.point_offsets,      b_point_reaction_offsets)
            .buf_arg(ApplyReactions_k.Args.points,             GPGPU.core_memory.get_buffer(CoreBufferType.POINT))
            .buf_arg(ApplyReactions_k.Args.anti_gravity,       GPGPU.core_memory.get_buffer(CoreBufferType.POINT_ANTI_GRAV))
            .buf_arg(ApplyReactions_k.Args.point_flags,        GPGPU.core_memory.get_buffer(CoreBufferType.POINT_FLAG))
            .buf_arg(ApplyReactions_k.Args.point_hit_counts,   GPGPU.core_memory.get_buffer(CoreBufferType.POINT_HIT_COUNT))
            .buf_arg(ApplyReactions_k.Args.point_hull_indices, GPGPU.core_memory.get_buffer(CoreBufferType.POINT_HULL_INDEX))
            .buf_arg(ApplyReactions_k.Args.hull_flags,         GPGPU.core_memory.get_buffer(CoreBufferType.HULL_FLAG));

        long k_ptr_move_hulls = p_sat_collide.kernel_ptr(Kernel.move_hulls);
        k_move_hulls = new MoveHulls_k(GPGPU.ptr_compute_queue, k_ptr_move_hulls)
            .buf_arg(MoveHulls_k.Args.hulls,                   GPGPU.core_memory.get_buffer(CoreBufferType.HULL))
            .buf_arg(MoveHulls_k.Args.hull_point_tables,       GPGPU.core_memory.get_buffer(CoreBufferType.HULL_POINT_TABLE))
            .buf_arg(MoveHulls_k.Args.points,                  GPGPU.core_memory.get_buffer(CoreBufferType.POINT));

        long k_ptr_move_entities = p_sat_collide.kernel_ptr(Kernel.move_entities);
        k_move_entities = new MoveEntities_k(GPGPU.ptr_compute_queue, k_ptr_move_entities)
            .buf_arg(MoveEntities_k.Args.hulls,                GPGPU.core_memory.get_buffer(CoreBufferType.HULL))
            .buf_arg(MoveEntities_k.Args.entities,             GPGPU.core_memory.get_buffer(CoreBufferType.ENTITY))
            .buf_arg(MoveEntities_k.Args.entity_flags,         GPGPU.core_memory.get_buffer(CoreBufferType.ENTITY_FLAG))
            .buf_arg(MoveEntities_k.Args.entity_motion_states, GPGPU.core_memory.get_buffer(CoreBufferType.ENTITY_MOTION_STATE))
            .buf_arg(MoveEntities_k.Args.entity_hull_tables,   GPGPU.core_memory.get_buffer(CoreBufferType.ENTITY_HULL_TABLE))
            .buf_arg(MoveEntities_k.Args.hull_point_tables,    GPGPU.core_memory.get_buffer(CoreBufferType.HULL_POINT_TABLE))
            .buf_arg(MoveEntities_k.Args.hull_integrity,       GPGPU.core_memory.get_buffer(CoreBufferType.HULL_INTEGRITY))
            .buf_arg(MoveEntities_k.Args.hull_flags,           GPGPU.core_memory.get_buffer(CoreBufferType.HULL_FLAG))
            .buf_arg(MoveEntities_k.Args.point_flags,          GPGPU.core_memory.get_buffer(CoreBufferType.POINT_FLAG))
            .buf_arg(MoveEntities_k.Args.point_hit_counts,     GPGPU.core_memory.get_buffer(CoreBufferType.POINT_HIT_COUNT))
            .buf_arg(MoveEntities_k.Args.points,               GPGPU.core_memory.get_buffer(CoreBufferType.POINT))
            .set_arg(MoveEntities_k.Args.dt,                   FIXED_TIME_STEP);

        long k_ptr_animate_entities = p_animate_hulls.kernel_ptr(Kernel.animate_entities);
        k_animate_entities = new AnimateEntities_k(GPGPU.ptr_compute_queue, k_ptr_animate_entities)
            .buf_arg(AnimateEntities_k.Args.armature_bones,              GPGPU.core_memory.get_buffer(CoreBufferType.ENTITY_BONE))
            .buf_arg(AnimateEntities_k.Args.bone_bind_poses,             GPGPU.core_memory.get_buffer(ReferenceBufferType.BONE_BIND_POSE))
            .buf_arg(AnimateEntities_k.Args.model_transforms,            GPGPU.core_memory.get_buffer(ReferenceBufferType.MODEL_TRANSFORM))
            .buf_arg(AnimateEntities_k.Args.entity_flags,                GPGPU.core_memory.get_buffer(CoreBufferType.ENTITY_FLAG))
            .buf_arg(AnimateEntities_k.Args.armature_bone_reference_ids, GPGPU.core_memory.get_buffer(CoreBufferType.ENTITY_BONE_REFERENCE_ID))
            .buf_arg(AnimateEntities_k.Args.armature_bone_parent_ids,    GPGPU.core_memory.get_buffer(CoreBufferType.ENTITY_BONE_PARENT_ID))
            .buf_arg(AnimateEntities_k.Args.bone_channel_tables,         GPGPU.core_memory.get_buffer(ReferenceBufferType.BONE_ANIM_CHANNEL_TABLE))
            .buf_arg(AnimateEntities_k.Args.bone_pos_channel_tables,     GPGPU.core_memory.get_buffer(ReferenceBufferType.ANIM_POS_CHANNEL))
            .buf_arg(AnimateEntities_k.Args.bone_rot_channel_tables,     GPGPU.core_memory.get_buffer(ReferenceBufferType.ANIM_ROT_CHANNEL))
            .buf_arg(AnimateEntities_k.Args.bone_scl_channel_tables,     GPGPU.core_memory.get_buffer(ReferenceBufferType.ANIM_SCL_CHANNEL))
            .buf_arg(AnimateEntities_k.Args.entity_model_transforms,     GPGPU.core_memory.get_buffer(CoreBufferType.ENTITY_TRANSFORM_ID))
            .buf_arg(AnimateEntities_k.Args.entity_bone_tables,          GPGPU.core_memory.get_buffer(CoreBufferType.ENTITY_BONE_TABLE))
            .buf_arg(AnimateEntities_k.Args.key_frames,                  GPGPU.core_memory.get_buffer(ReferenceBufferType.ANIM_KEY_FRAME))
            .buf_arg(AnimateEntities_k.Args.frame_times,                 GPGPU.core_memory.get_buffer(ReferenceBufferType.ANIM_FRAME_TIME))
            .buf_arg(AnimateEntities_k.Args.animation_timing_indices,    GPGPU.core_memory.get_buffer(ReferenceBufferType.ANIM_TIMING_INDEX))
            .buf_arg(AnimateEntities_k.Args.animation_durations,         GPGPU.core_memory.get_buffer(ReferenceBufferType.ANIM_DURATION))
            .buf_arg(AnimateEntities_k.Args.animation_tick_rates,        GPGPU.core_memory.get_buffer(ReferenceBufferType.ANIM_TICK_RATE))
            .buf_arg(AnimateEntities_k.Args.entity_animation_indices,    GPGPU.core_memory.get_buffer(CoreBufferType.ENTITY_ANIM_INDEX))
            .buf_arg(AnimateEntities_k.Args.entity_animation_elapsed,    GPGPU.core_memory.get_buffer(CoreBufferType.ENTITY_ANIM_ELAPSED))
            .buf_arg(AnimateEntities_k.Args.entity_animation_blend,      GPGPU.core_memory.get_buffer(CoreBufferType.ENTITY_ANIM_BLEND));

        long k_ptr_animate_bones = p_animate_hulls.kernel_ptr(Kernel.animate_bones);
        k_animate_bones = new AnimateBones_k(GPGPU.ptr_compute_queue, k_ptr_animate_bones)
            .buf_arg(AnimateBones_k.Args.bones,                       GPGPU.core_memory.get_buffer(CoreBufferType.HULL_BONE))
            .buf_arg(AnimateBones_k.Args.bone_references,             GPGPU.core_memory.get_buffer(ReferenceBufferType.BONE_REFERENCE))
            .buf_arg(AnimateBones_k.Args.armature_bones,              GPGPU.core_memory.get_buffer(CoreBufferType.ENTITY_BONE))
            .buf_arg(AnimateBones_k.Args.hull_bind_pose_indicies,     GPGPU.core_memory.get_buffer(CoreBufferType.HULL_BONE_BIND_POSE))
            .buf_arg(AnimateBones_k.Args.hull_inv_bind_pose_indicies, GPGPU.core_memory.get_buffer(CoreBufferType.HULL_BONE_INV_BIND_POSE));

        long k_ptr_animate_points = p_animate_hulls.kernel_ptr(Kernel.animate_points);
        k_animate_points = new AnimatePoints_k(GPGPU.ptr_compute_queue, k_ptr_animate_points)
            .buf_arg(AnimatePoints_k.Args.points,                  GPGPU.core_memory.get_buffer(CoreBufferType.POINT))
            .buf_arg(AnimatePoints_k.Args.hull_scales,             GPGPU.core_memory.get_buffer(CoreBufferType.HULL_SCALE))
            .buf_arg(AnimatePoints_k.Args.hull_entity_ids,         GPGPU.core_memory.get_buffer(CoreBufferType.HULL_ENTITY_ID))
            .buf_arg(AnimatePoints_k.Args.hull_flags,              GPGPU.core_memory.get_buffer(CoreBufferType.HULL_FLAG))
            .buf_arg(AnimatePoints_k.Args.point_vertex_references, GPGPU.core_memory.get_buffer(CoreBufferType.POINT_VERTEX_REFERENCE))
            .buf_arg(AnimatePoints_k.Args.point_hull_indices,      GPGPU.core_memory.get_buffer(CoreBufferType.POINT_HULL_INDEX))
            .buf_arg(AnimatePoints_k.Args.bone_tables,             GPGPU.core_memory.get_buffer(CoreBufferType.POINT_BONE_TABLE))
            .buf_arg(AnimatePoints_k.Args.vertex_weights,          GPGPU.core_memory.get_buffer(ReferenceBufferType.VERTEX_WEIGHT))
            .buf_arg(AnimatePoints_k.Args.entities,                GPGPU.core_memory.get_buffer(CoreBufferType.ENTITY))
            .buf_arg(AnimatePoints_k.Args.vertex_references,       GPGPU.core_memory.get_buffer(ReferenceBufferType.VERTEX_REFERENCE))
            .buf_arg(AnimatePoints_k.Args.bones,                   GPGPU.core_memory.get_buffer(CoreBufferType.HULL_BONE));

        long k_ptr_resolve_constraints = p_resolve_constraints.kernel_ptr(Kernel.resolve_constraints);
        k_resolve_constraints = new ResolveConstraints_k(GPGPU.ptr_compute_queue, k_ptr_resolve_constraints)
            .buf_arg(ResolveConstraints_k.Args.hull_edge_tables, GPGPU.core_memory.get_buffer(CoreBufferType.HULL_EDGE_TABLE))
            .buf_arg(ResolveConstraints_k.Args.bounds_bank_data, GPGPU.core_memory.get_buffer(CoreBufferType.HULL_AABB_KEY_TABLE))
            .buf_arg(ResolveConstraints_k.Args.point,            GPGPU.core_memory.get_buffer(CoreBufferType.POINT))
            .buf_arg(ResolveConstraints_k.Args.edges,            GPGPU.core_memory.get_buffer(CoreBufferType.EDGE))
            .buf_arg(ResolveConstraints_k.Args.edge_lengths,     GPGPU.core_memory.get_buffer(CoreBufferType.EDGE_LENGTH));
        
        long k_ptr_scan_bounds_single_block    = p_scan_key_bank.kernel_ptr(Kernel.scan_bounds_single_block);
        long k_ptr_scan_bounds_multi_block     = p_scan_key_bank.kernel_ptr(Kernel.scan_bounds_multi_block);
        long k_ptr_complete_bounds_multi_block = p_scan_key_bank.kernel_ptr(Kernel.complete_bounds_multi_block);
        k_scan_bounds_single_block    = new ScanBoundsSingleBlock_k(GPGPU.ptr_compute_queue, k_ptr_scan_bounds_single_block);
        k_scan_bounds_multi_block     = new ScanBoundsMultiBlock_k(GPGPU.ptr_compute_queue, k_ptr_scan_bounds_multi_block);
        k_complete_bounds_multi_block = new CompleteBoundsMultiBlock_k(GPGPU.ptr_compute_queue, k_ptr_complete_bounds_multi_block);

        long k_ptr_scan_candidates_single_block_out    = p_scan_key_candidates.kernel_ptr(Kernel.scan_candidates_single_block_out);
        long k_ptr_scan_candidates_multi_block_out     = p_scan_key_candidates.kernel_ptr(Kernel.scan_candidates_multi_block_out);
        long k_ptr_complete_candidates_multi_block_out = p_scan_key_candidates.kernel_ptr(Kernel.complete_candidates_multi_block_out);
        k_scan_candidates_single_block_out    = new ScanCandidatesSingleBlockOut_k(GPGPU.ptr_compute_queue, k_ptr_scan_candidates_single_block_out);
        k_scan_candidates_multi_block_out     = new ScanCandidatesMultiBlockOut_k(GPGPU.ptr_compute_queue, k_ptr_scan_candidates_multi_block_out);
        k_complete_candidates_multi_block_out = new CompleteCandidatesMultiBlockOut_k(GPGPU.ptr_compute_queue, k_ptr_complete_candidates_multi_block_out);
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

        k_integrate
            .ptr_arg(Integrate_k.Args.args, arg_mem_ptr)
            .call(arg_long(GPGPU.core_memory.sector_container().next_hull()));

        k_integrate_entities
            .ptr_arg(IntegrateEntities_k.Args.args, arg_mem_ptr)
            .call(arg_long(GPGPU.core_memory.sector_container().next_entity()));

        GPGPU.cl_release_buffer(arg_mem_ptr);

        if (Editor.ACTIVE)
        {
            long e = System.nanoTime() - s;
            Editor.queue_event("phys_integrate", String.valueOf(e));
        }
    }

    private void update_controllable_entities()
    {
        InputState inputState = Component.InputState.forEntity(ecs, Constants.PLAYER_ID);
        EntityIndex entity_id         = Component.EntityId.forEntity(ecs, Constants.PLAYER_ID);
        EntityIndex mouse_cursor      = Component.MouseCursorId.forEntity(ecs, Constants.PLAYER_ID);
        EntityIndex block_cursor      = Component.BlockCursorId.forEntity(ecs, Constants.PLAYER_ID);
        LinearForce force             = Component.LinearForce.forEntity(ecs, Constants.PLAYER_ID);

        Objects.requireNonNull(inputState);
        Objects.requireNonNull(entity_id);
        Objects.requireNonNull(mouse_cursor);
        Objects.requireNonNull(block_cursor);
        Objects.requireNonNull(force);

        int flags = 0;

        var inputStates = inputState.input_states();
        for (var binding : InputBinding.values())
        {
            var on = inputStates.get(binding);
            if (on == null) continue;
            if (on)
            {
                int flag = switch (binding)
                {
                    case MOVE_UP         -> Constants.ControlFlags.UP.bits;
                    case MOVE_DOWN       -> Constants.ControlFlags.DOWN.bits;
                    case MOVE_LEFT       -> Constants.ControlFlags.LEFT.bits;
                    case MOVE_RIGHT      -> Constants.ControlFlags.RIGHT.bits;
                    case JUMP            -> Constants.ControlFlags.JUMP.bits;
                    case RUN             -> Constants.ControlFlags.RUN.bits;
                    case MOUSE_PRIMARY   -> Constants.ControlFlags.MOUSE1.bits;
                    case MOUSE_SECONDARY -> Constants.ControlFlags.MOUSE2.bits;
                    case MOUSE_MIDDLE,
                         MOUSE_BACK,
                         MOUSE_FORWARD   -> 0;
                };
                flags |= flag;
            }
        }

        k_set_control_points
            .set_arg(SetControlPoints_k.Args.target, entity_id.index()) // todo: probably don't need this in 2 args
            .set_arg(SetControlPoints_k.Args.new_flags, flags)
            .set_arg(SetControlPoints_k.Args.new_index, entity_id.index())
            .set_arg(SetControlPoints_k.Args.new_jump_mag, GRAVITY_MAGNITUDE * 550)
            .set_arg(SetControlPoints_k.Args.new_linear_mag, force.magnitude())
            .call(GPGPU.global_single_size);

        var camera = Window.get().camera();
        float world_x = inputState.get_screen_target().x * camera.get_zoom() + camera.position().x;
        float world_y = (Window.get().height() - inputState.get_screen_target().y) * camera.get_zoom() + camera.position().y;
        GPGPU.core_memory.update_entity_position(mouse_cursor.index(), world_x, world_y);

        // todo: don't bother if block cursor is inactive.
        //  when block cursor becomes active, reset position to player entity x,y.
        //  it should then move toward the mouse instead of being pinned to it,
        //  so it shouldn't end up inside other geometry
        GPGPU.core_memory.update_entity_position(block_cursor.index(), world_x, world_y);

        k_handle_movement.set_arg(HandleMovement_k.Args.dt, FIXED_TIME_STEP)
            .call(GPGPU.global_single_size);
    }

    //#endregion

    //#region AABB Collision

    private void calculate_bank_offsets()
    {
        long s = Editor.ACTIVE ? System.nanoTime() : 0;
        int bank_size = scan_key_bounds(GPGPU.core_memory.get_buffer(CoreBufferType.HULL_AABB_KEY_TABLE).pointer(), GPGPU.core_memory.sector_container().next_hull());
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

        GPGPU.cl_zero_buffer(GPGPU.ptr_compute_queue, svm_atomic_counter, CLSize.cl_int);

        k_scan_bounds_single_block
            .ptr_arg(ScanBoundsSingleBlock_k.Args.bounds_bank_data, data_ptr)
            .ptr_arg(ScanBoundsSingleBlock_k.Args.sz, svm_atomic_counter)
            .loc_arg(ScanBoundsSingleBlock_k.Args.buffer, local_buffer_size)
            .set_arg(ScanBoundsSingleBlock_k.Args.n, n)
            .call(GPGPU.local_work_default, GPGPU.local_work_default);

        return GPGPU.cl_read_pinned_int(GPGPU.ptr_compute_queue, svm_atomic_counter);
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

        k_scan_bounds_multi_block
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

        gpu_int_scan.scan_int(p_data, part_size);

        GPGPU.cl_zero_buffer(GPGPU.ptr_compute_queue, svm_atomic_counter, CLSize.cl_int);

        s = Editor.ACTIVE ? System.nanoTime() : 0;

        k_complete_bounds_multi_block
            .ptr_arg(CompleteBoundsMultiBlock_k.Args.bounds_bank_data, data_ptr)
            .ptr_arg(CompleteBoundsMultiBlock_k.Args.sz, svm_atomic_counter)
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
        int r = GPGPU.cl_read_pinned_int(GPGPU.ptr_compute_queue, svm_atomic_counter);
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

        b_key_bank.ensure_capacity(uniform_grid.get_key_bank_size());
        GPGPU.cl_zero_buffer(GPGPU.ptr_compute_queue, ptr_counts_data, grid_buffer_size);
        k_generate_keys
            .set_arg(GenerateKeys_k.Args.key_bank_length, uniform_grid.get_key_bank_size())
            .call(arg_long(GPGPU.core_memory.sector_container().next_hull()));
        if (Editor.ACTIVE)
        {
            long e = System.nanoTime() - s;
            Editor.queue_event("phys_gen_keys", String.valueOf(e));
        }
    }

    private void build_key_map(UniformGrid uniform_grid)
    {
        long s = Editor.ACTIVE ? System.nanoTime() : 0;
        b_key_map.ensure_capacity(uniform_grid.getKey_map_size());
        GPGPU.cl_zero_buffer(GPGPU.ptr_compute_queue, ptr_counts_data, grid_buffer_size);
        k_build_key_map.call(arg_long(GPGPU.core_memory.sector_container().next_hull()));
        if (Editor.ACTIVE)
        {
            long e = System.nanoTime() - s;
            Editor.queue_event("phys_key_map", String.valueOf(e));
        }
    }

    private void locate_in_bounds()
    {
        long s = Editor.ACTIVE ? System.nanoTime() : 0;
        int hull_count = GPGPU.core_memory.sector_container().next_hull();
        b_in_bounds.ensure_capacity(hull_count);
        GPGPU.cl_zero_buffer(GPGPU.ptr_compute_queue, svm_atomic_counter, CLSize.cl_int);

        k_locate_in_bounds
            .ptr_arg(LocateInBounds_k.Args.counter, svm_atomic_counter)
            .call(arg_long(hull_count));

        candidate_buffer_count = GPGPU.cl_read_pinned_int(GPGPU.ptr_compute_queue, svm_atomic_counter);
        if (Editor.ACTIVE)
        {
            long e = System.nanoTime() - s;
            Editor.queue_event("phys_locate_inbounds", String.valueOf(e));
        }
    }

    private void calculate_match_candidates()
    {
        long s = Editor.ACTIVE ? System.nanoTime() : 0;
        b_candidate_counts.ensure_capacity(candidate_buffer_count);
        k_count_candidates.call(arg_long(candidate_buffer_count));
        if (Editor.ACTIVE)
        {
            long e = System.nanoTime() - s;
            Editor.queue_event("phys_match_candidates", String.valueOf(e));
        }
    }

    private int scan_single_block_candidates_out(long data_ptr, long o_data_ptr, int n)
    {
        long local_buffer_size = CLSize.cl_int * GPGPU.max_scan_block_size;

        GPGPU.cl_zero_buffer(GPGPU.ptr_compute_queue, svm_atomic_counter, CLSize.cl_int);

        k_scan_candidates_single_block_out
            .ptr_arg(ScanCandidatesSingleBlockOut_k.Args.input, data_ptr)
            .ptr_arg(ScanCandidatesSingleBlockOut_k.Args.output, o_data_ptr)
            .ptr_arg(ScanCandidatesSingleBlockOut_k.Args.sz, svm_atomic_counter)
            .loc_arg(ScanCandidatesSingleBlockOut_k.Args.buffer, local_buffer_size)
            .set_arg(ScanCandidatesSingleBlockOut_k.Args.n, n)
            .call(GPGPU.local_work_default, GPGPU.local_work_default);

        return GPGPU.cl_read_pinned_int(GPGPU.ptr_compute_queue, svm_atomic_counter);
    }

    private int scan_multi_block_candidates_out(long data_ptr, long o_data_ptr, int n, int k)
    {
        long local_buffer_size = CLSize.cl_int * GPGPU.max_scan_block_size;

        long gx = k * GPGPU.max_scan_block_size;
        long[] global_work_size = arg_long(gx);
        int part_size = k * 2;
        long part_buf_size = ((long) CLSize.cl_int * ((long) part_size));
        var p_data = GPGPU.cl_new_buffer(part_buf_size);

        k_scan_candidates_multi_block_out
            .ptr_arg(ScanCandidatesMultiBlockOut_k.Args.input, data_ptr)
            .ptr_arg(ScanCandidatesMultiBlockOut_k.Args.output, o_data_ptr)
            .loc_arg(ScanCandidatesMultiBlockOut_k.Args.buffer, local_buffer_size)
            .ptr_arg(ScanCandidatesMultiBlockOut_k.Args.part, p_data)
            .set_arg(ScanCandidatesMultiBlockOut_k.Args.n, n)
            .call(global_work_size, GPGPU.local_work_default);

        gpu_int_scan.scan_int(p_data, part_size);

        GPGPU.cl_zero_buffer(GPGPU.ptr_compute_queue, svm_atomic_counter, CLSize.cl_int);

        k_complete_candidates_multi_block_out
            .ptr_arg(CompleteCandidatesMultiBlockOut_k.Args.input, data_ptr)
            .ptr_arg(CompleteCandidatesMultiBlockOut_k.Args.output, o_data_ptr)
            .ptr_arg(CompleteCandidatesMultiBlockOut_k.Args.sz, svm_atomic_counter)
            .loc_arg(CompleteCandidatesMultiBlockOut_k.Args.buffer, local_buffer_size)
            .ptr_arg(CompleteCandidatesMultiBlockOut_k.Args.part, p_data)
            .set_arg(CompleteCandidatesMultiBlockOut_k.Args.n, n)
            .call(global_work_size, GPGPU.local_work_default);

        GPGPU.cl_release_buffer(p_data);

        return GPGPU.cl_read_pinned_int(GPGPU.ptr_compute_queue, svm_atomic_counter);
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
        b_candidate_offsets.ensure_capacity(candidate_buffer_count);
        match_buffer_count = scan_key_candidates(b_candidate_counts.pointer(), b_candidate_offsets.pointer(), (int) candidate_buffer_count);
        if (Editor.ACTIVE)
        {
            long e = System.nanoTime() - s;
            Editor.queue_event("phys_match_offsets", String.valueOf(e));
        }
    }

    private void aabb_collide()
    {
        long s = Editor.ACTIVE ? System.nanoTime() : 0;
        b_matches.ensure_capacity(match_buffer_count);
        b_matches_used.ensure_capacity(candidate_buffer_count);
        GPGPU.cl_zero_buffer(GPGPU.ptr_compute_queue, svm_atomic_counter, CLSize.cl_int);
        k_aabb_collide.call(arg_long(candidate_buffer_count));
        candidate_count = GPGPU.cl_read_pinned_int(GPGPU.ptr_compute_queue, svm_atomic_counter);
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

        b_candidates.ensure_capacity(candidate_count);

        int[] counter = new int[]{0};
        var counter_ptr = GPGPU.cl_new_int_arg_buffer(counter);

        candidate_buffer_size = buffer_size;

        k_finalize_candidates
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
        GPGPU.cl_zero_buffer(GPGPU.ptr_compute_queue, svm_atomic_counter, CLSize.cl_int);

        long max_point_count = candidate_buffer_size
            * 2  // there are two bodies per collision pair
            * 2; // assume worst case is 2 points per body

        b_reactions_in.ensure_capacity(max_point_count);
        b_reactions_out.ensure_capacity(max_point_count);
        b_reaction_index.ensure_capacity(max_point_count);
        b_point_reaction_counts.ensure_capacity(GPGPU.core_memory.sector_container().next_point());
        b_point_reaction_offsets.ensure_capacity(GPGPU.core_memory.sector_container().next_point());
        k_sat_collide.call(global_work_size);
        reaction_count = GPGPU.cl_read_pinned_int(GPGPU.ptr_compute_queue, svm_atomic_counter);
        if (Editor.ACTIVE)
        {
            long e = System.nanoTime() - s;
            Editor.queue_event("phys_sat_collide", String.valueOf(e));
        }
    }

    private void scan_reactions()
    {
        long s = Editor.ACTIVE ? System.nanoTime() : 0;
        gpu_int_scan_out.scan_int_out(b_point_reaction_counts.pointer(), b_point_reaction_offsets.pointer(), GPGPU.core_memory.sector_container().next_point());
        b_point_reaction_counts.clear();
        if (Editor.ACTIVE)
        {
            long e = System.nanoTime() - s;
            Editor.queue_event("phys_sat_scan_reactions", String.valueOf(e));
        }
    }

    private void sort_reactions()
    {
        long s = Editor.ACTIVE ? System.nanoTime() : 0;
        k_sort_reactions.call(arg_long(reaction_count));
        if (Editor.ACTIVE)
        {
            long e = System.nanoTime() - s;
            Editor.queue_event("phys_sat_sort_reactions", String.valueOf(e));
        }
    }

    private void apply_reactions()
    {
        long s = Editor.ACTIVE ? System.nanoTime() : 0;
        k_apply_reactions.call(arg_long(GPGPU.core_memory.sector_container().next_point()));
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
        k_animate_entities
            .set_arg(AnimateEntities_k.Args.delta_time, dt)
            .call(arg_long(GPGPU.core_memory.sector_container().next_entity()));
        if (Editor.ACTIVE)
        {
            long e = System.nanoTime() - s;
            Editor.queue_event("phys_animate_entities", String.valueOf(e));
        }
    }

    private void animate_bones()
    {
        long s = Editor.ACTIVE ? System.nanoTime() : 0;
        k_animate_bones.call(arg_long(GPGPU.core_memory.sector_container().next_hull_bone()));
        if (Editor.ACTIVE)
        {
            long e = System.nanoTime() - s;
            Editor.queue_event("phys_animate_bones", String.valueOf(e));
        }
    }

    private void animate_points()
    {
        long s = Editor.ACTIVE ? System.nanoTime() : 0;
        k_animate_points.call(arg_long(GPGPU.core_memory.sector_container().next_point()));
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

            k_resolve_constraints
                .set_arg(ResolveConstraints_k.Args.process_all, n)
                .call(arg_long(GPGPU.core_memory.sector_container().next_hull()));
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
        k_move_entities.call(arg_long(GPGPU.core_memory.sector_container().next_entity()));
        if (Editor.ACTIVE)
        {
            long e = System.nanoTime() - s;
            Editor.queue_event("phys_move_entities", String.valueOf(e));
        }
    }

    private void resolve_hulls()
    {
        long s = Editor.ACTIVE ? System.nanoTime() : 0;
        k_move_hulls.call(arg_long(GPGPU.core_memory.sector_container().next_hull()));
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
        GPGPU.cl_zero_buffer(GPGPU.ptr_compute_queue, ptr_offsets_data, grid_buffer_size);
        gpu_int_scan_out.scan_int_out(ptr_counts_data, ptr_offsets_data, uniform_grid.directory_length);

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

    private void simulate(float dt)
    {
        long s = Editor.ACTIVE ? System.nanoTime() : 0;

        //----------------------//
        // Pre-Simulation Setup //
        //----------------------//

        // An initial constraint solve pass is done before simulation to ensure edges are in their "safe"
        // convex shape. Animations may move points into positions where the geometry becomes concave,
        // so this call prevents collision errors due to non-convex shapes.
        resolve_constraints(EDGE_STEPS);

        //-----------------//
        // Simulation Loop //
        //-----------------//

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

        //-------------------------//
        // Post Simulation Cleanup //
        //-------------------------//

        // TODO: read back mouse colliding objects here

        // zero out the acceleration buffer, so it is empty for the next frame
        GPGPU.core_memory.get_buffer(CoreBufferType.ENTITY_ACCEL).clear();

        long se = Editor.ACTIVE ? System.nanoTime() : 0;
        // Entities that are exiting the playable area are considered to be "in egress",
        // this step determines how many of each object type is in that state, so they
        // can be transferred into the egress buffer, and eventually onto disk.
        int[] egress_counts = GPGPU.core_memory.count_egress_entities();
        GPGPU.core_memory.egress(egress_counts);

        if (Editor.ACTIVE)
        {
            long e = System.nanoTime() - se;
            Editor.queue_event("phys_egress", String.valueOf(e));
        }

        // Deletion of objects happens only once per simulation tick, instead of every sub-step
        // to ensure buffer compaction happens as infrequently as possible.
        long sd = Editor.ACTIVE ? System.nanoTime() : 0;
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
        /*
         * Synchronization Notes:
         *     This method relies on a number of threads co-ordinating with each other to ensure data
         *     integrity and system stability. Because of the importance of getting this right, the
         *     changes in thread state are documented inline with a "STATE:" and "QUEUE:" prefixed
         *     "shorthand" comments. State comments describe the state of certain threads just before,
         *     or just after calls that affect them. Queue comments describe tasks that are assumed to
         *     be complete, though it is helpful to know that a queue being "finished" only guarantees
         *     work "in-flight" has completed before returning, it does not guarantee some thread will
         *     not queue more work afterward. Putting it more directly, the GPU command queues associated
         *     with specific tasks are asynchronous with the CPU tasks that use them. Before entering
         *     the try block, it should be assumed that all relevant threads may still be running.
         */
        try
        {
            clFinish(GPGPU.ptr_render_queue);           // QUEUE: render complete

            long phys_time = last_phys_time.take();     // STATE: main      -> block on   : `phys_time`
                                                        // STATE: main      -> unblock
                                                        // STATE: physics   -> block on   : `dt`
                                                        // STATE: ingress   -> blocked on : `world_barrier`
                                                        // STATE: egress    -> blocked on : `world_barrier`
                                                        // STATE: inventory -> blocked on : `world_barrier`

            GPGPU.core_memory.await_world_barrier();    // STATE: main      -> block on   : `world_barrier`
            GPGPU.core_memory.release_world_barrier();  // STATE: main      -> unblock
                                                        // STATE: ingress   -> block on   : `dt`
                                                        // STATE: egress    -> block on   : `dt`
                                                        // STATE: inventory -> block on   : `dt`

            clFinish(GPGPU.ptr_sector_queue);           // QUEUE: previous sector processing complete

            /*
            * At this point, all important threads are blocked, and all relevant queues are idle.
            * Now, any data that needs to transfer between ingress/egress/render buffers is moved.
            * Any work that occurs here should be as fast as possible, avoid adding excess latency.
            * */

            GPGPU.core_memory.swap_ingress_buffers();
            GPGPU.core_memory.swap_egress_buffers();
            GPGPU.core_memory.swap_render_buffers();

            clFinish(GPGPU.ptr_compute_queue);          // QUEUE: current sector processing complete

            next_phys_time.put(dt);                     // STATE: physics   -> unblock

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

        gpu_int_scan.destroy();
        gpu_int_scan_out.destroy();

        p_control_entities.destroy();
        p_integrate.destroy();
        p_scan_key_bank.destroy();
        p_generate_keys.destroy();
        p_build_key_map.destroy();
        p_locate_in_bounds.destroy();
        p_scan_key_candidates.destroy();
        p_aabb_collide.destroy();
        p_sat_collide.destroy();
        p_animate_hulls.destroy();
        p_resolve_constraints.destroy();

        b_point_reaction_counts.release();
        b_point_reaction_offsets.release();
        b_reactions_in.release();
        b_reactions_out.release();
        b_reaction_index.release();
        b_key_map.release();
        b_key_bank.release();
        b_in_bounds.release();
        b_candidates.release();
        b_candidate_counts.release();
        b_candidate_offsets.release();
        b_matches.release();
        b_matches_used.release();
        b_control_point_flags.release();
        b_control_point_indices.release();
        b_control_point_tick_budgets.release();
        b_control_point_linear_mag.release();
        b_control_point_jump_mag.release();

        debug();

        GPGPU.cl_release_buffer(svm_atomic_counter);
        GPGPU.cl_release_buffer(ptr_counts_data);
        GPGPU.cl_release_buffer(ptr_offsets_data);
    }

    //#endregion

    private void debug()
    {
        long total = 0;
        total += b_point_reaction_counts.debug_data();
        total += b_point_reaction_offsets.debug_data();
        total += b_reactions_in.debug_data();
        total += b_reactions_out.debug_data();
        total += b_reaction_index.debug_data();
        total += b_key_map.debug_data();
        total += b_key_bank.debug_data();
        total += b_in_bounds.debug_data();
        total += b_candidates.debug_data();
        total += b_candidate_counts.debug_data();
        total += b_candidate_offsets.debug_data();
        total += b_matches.debug_data();
        total += b_matches_used.debug_data();
        total += b_control_point_flags.debug_data();
        total += b_control_point_indices.debug_data();
        total += b_control_point_tick_budgets.debug_data();
        total += b_control_point_linear_mag.debug_data();
        total += b_control_point_jump_mag.debug_data();

        //System.out.println("---------------------------");
        System.out.println("Physics Memory Usage: MB " + ((float) total / 1024f / 1024f));
    }
}
