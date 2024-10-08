package com.controllerface.bvge.physics;

import com.controllerface.bvge.ecs.ECS;
import com.controllerface.bvge.ecs.GameSystem;
import com.controllerface.bvge.editor.Editor;
import com.controllerface.bvge.game.state.PlayerController;
import com.controllerface.bvge.gpu.GPU;
import com.controllerface.bvge.gpu.cl.buffers.BufferGroup;
import com.controllerface.bvge.gpu.cl.buffers.CL_Buffer;
import com.controllerface.bvge.gpu.cl.buffers.PersistentBuffer;
import com.controllerface.bvge.gpu.cl.buffers.ResizableBuffer;
import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;
import com.controllerface.bvge.gpu.cl.kernels.animation.AnimateBones_k;
import com.controllerface.bvge.gpu.cl.kernels.animation.AnimateEntities_k;
import com.controllerface.bvge.gpu.cl.kernels.animation.AnimatePoints_k;
import com.controllerface.bvge.gpu.cl.kernels.physics.*;
import com.controllerface.bvge.gpu.cl.programs.GPUProgram;
import com.controllerface.bvge.gpu.cl.programs.physics.*;
import com.controllerface.bvge.gpu.cl.programs.scan.GPUScanScalarInt;
import com.controllerface.bvge.gpu.cl.programs.scan.GPUScanScalarIntOut;
import com.controllerface.bvge.gpu.cl.programs.scan.ScanKeyBank;
import com.controllerface.bvge.gpu.cl.programs.scan.ScanKeyCandidates;
import com.controllerface.bvge.memory.types.CoreBufferType;
import com.controllerface.bvge.memory.types.PhysicsBufferType;

import java.util.concurrent.BlockingQueue;
import java.util.concurrent.SynchronousQueue;

import static com.controllerface.bvge.gpu.GPU.CL.arg_long;
import static com.controllerface.bvge.gpu.cl.buffers.CL_DataTypes.*;

public class PhysicsSimulation extends GameSystem
{
    //#region Constants

    private static final float TARGET_FPS = 60.0f;
    private static final float TICK_RATE = 1.0f / TARGET_FPS;
    private static final int TARGET_SUB_STEPS = 6;
    private static final int MAX_SUB_STEPS = 8;
    private static final int EDGE_STEPS = 16;

    public static final float FIXED_TIME_STEP = TICK_RATE / TARGET_SUB_STEPS;

    // todo: gravity should not be a constant but calculated based on proximity next to planets and other large bodies
    private static final float GRAVITY_MAGNITUDE = 9.8f * 5;
    private static final float GRAVITY_X = 0;
    private static final float GRAVITY_Y = -GRAVITY_MAGNITUDE * TARGET_FPS;

    // todo: investigate if this should be variable as well. It may make sense to increase damping in some cases,
    //  and lower it in others, for example in space vs on a planet. It may also be useful to set the direction
    //  or make damping interact with the gravity vector in some way. Kernels already do this now, may be helpful
    //  to have this as a variable.
    private static final float MOTION_DAMPING = .990f;

    private static final long INIT_BUFFER_SIZE = 500_000L;
    private static final String CANDIDATE_BUFFER_NAME = "Physics Collision";
    private static final String KEY_BANK_BUFFER_NAME  = "Physics AABB Key Bank/Map";
    private static final String REACTION_BUFFER_NAME  = "Physics Collision Reactions";
    private static final String MATCH_BUFFER_NAME     = "Physics Matches";

    //#endregion

    //#region GPU Programs & Kernels

    private final GPUProgram p_integrate                = new Integrate();
    private final GPUProgram p_scan_key_bank            = new ScanKeyBank();
    private final GPUProgram p_build_key_bank           = new BuildKeyBank();
    private final GPUProgram p_build_key_bank_edge      = new BuildKeyBank();
    private final GPUProgram p_build_key_map            = new BuildKeyMap();
    private final GPUProgram p_build_key_map_edge       = new BuildKeyMap();
    private final GPUProgram p_locate_in_bounds         = new LocateInBounds();
    private final GPUProgram p_locate_in_bounds_edge    = new LocateInBounds();
    private final GPUProgram p_scan_key_candidates      = new ScanKeyCandidates();
    private final GPUProgram p_aabb_collide             = new AabbCollide();
    private final GPUProgram p_sat_collide              = new SatCollide();
    private final GPUProgram p_animate_hulls            = new AnimateHulls();
    private final GPUProgram p_resolve_constraints      = new ResolveConstraints();

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
    private final GPUKernel k_build_key_bank;
    private final GPUKernel k_integrate;
    private final GPUKernel k_integrate_entities;
    private final GPUKernel k_calculate_hull_aabb;
    private final GPUKernel k_locate_in_bounds;
    private final GPUKernel k_move_entities;
    private final GPUKernel k_move_hulls;
    private final GPUKernel k_resolve_constraints;
    private final GPUKernel k_sat_collide;
    private final GPUKernel k_scan_bounds_multi_block;
    private final GPUKernel k_scan_bounds_single_block;
    private final GPUKernel k_scan_candidates_multi_block_out;
    private final GPUKernel k_scan_candidates_single_block_out;
    private final GPUKernel k_sort_reactions;

    //#endregion

    //#region Buffers & Counters

    private final long grid_buffer_size;
    private final CL_Buffer counts_buf;
    private final CL_Buffer offsets_buf;
    private final CL_Buffer atomic_counter;

    public final ResizableBuffer b_control_point_flags;
    public final ResizableBuffer b_control_point_indices;
    public final ResizableBuffer b_control_point_tick_budgets;
    public final ResizableBuffer b_control_point_linear_mag;
    public final ResizableBuffer b_control_point_jump_mag;

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

    private final BlockingQueue<Float> next_phys_time   = new SynchronousQueue<>();
    private final BlockingQueue<Long> last_phys_time    = new SynchronousQueue<>();
    private final GPUScanScalarInt gpu_int_scan;
    private final GPUScanScalarIntOut gpu_int_scan_out;

    private final PlayerController player_controller;

    private final BufferGroup<PhysicsBufferType> reaction_buffers;
    private final BufferGroup<PhysicsBufferType> key_buffers;
    private final BufferGroup<PhysicsBufferType> candidate_buffers;
    private final BufferGroup<PhysicsBufferType> match_buffers;

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

    public PhysicsSimulation(ECS ecs, UniformGrid uniform_grid, PlayerController player_controller)
    {
        super(ecs);
        this.uniform_grid = uniform_grid;
        this.player_controller = player_controller;

        gpu_int_scan = new GPUScanScalarInt(GPU.compute.physics_queue);
        gpu_int_scan_out = new GPUScanScalarIntOut(GPU.compute.physics_queue, gpu_int_scan);

        grid_buffer_size = (long) cl_int.size() * this.uniform_grid.directory_length;

        atomic_counter = GPU.CL.new_pinned_int(GPU.compute.context);
        counts_buf = GPU.CL.new_buffer(GPU.compute.context, grid_buffer_size);
        offsets_buf = GPU.CL.new_buffer(GPU.compute.context, grid_buffer_size);

        reaction_buffers = new BufferGroup<>(GPU.compute.physics_queue, PhysicsBufferType.class, REACTION_BUFFER_NAME, false);
        reaction_buffers.init_buffer(PhysicsBufferType.POINT_REACTION_COUNTS, INIT_BUFFER_SIZE);
        reaction_buffers.init_buffer(PhysicsBufferType.POINT_REACTION_OFFSETS, INIT_BUFFER_SIZE);
        reaction_buffers.init_buffer(PhysicsBufferType.REACTIONS_IN, INIT_BUFFER_SIZE);
        reaction_buffers.init_buffer(PhysicsBufferType.REACTIONS_OUT, INIT_BUFFER_SIZE);
        reaction_buffers.init_buffer(PhysicsBufferType.REACTION_INDEX, INIT_BUFFER_SIZE);

        key_buffers = new BufferGroup<>(GPU.compute.physics_queue, PhysicsBufferType.class, KEY_BANK_BUFFER_NAME, false);
        key_buffers.init_buffer(PhysicsBufferType.KEY_MAP, INIT_BUFFER_SIZE);
        key_buffers.init_buffer(PhysicsBufferType.KEY_BANK, INIT_BUFFER_SIZE);

        candidate_buffers = new BufferGroup<>(GPU.compute.physics_queue, PhysicsBufferType.class, CANDIDATE_BUFFER_NAME, false);
        candidate_buffers.init_buffer(PhysicsBufferType.IN_BOUNDS, INIT_BUFFER_SIZE);
        candidate_buffers.init_buffer(PhysicsBufferType.CANDIDATES, INIT_BUFFER_SIZE);
        candidate_buffers.init_buffer(PhysicsBufferType.CANDIDATE_COUNTS, INIT_BUFFER_SIZE);
        candidate_buffers.init_buffer(PhysicsBufferType.CANDIDATE_OFFSETS, INIT_BUFFER_SIZE);

        match_buffers = new BufferGroup<>(GPU.compute.physics_queue, PhysicsBufferType.class, MATCH_BUFFER_NAME, false);
        match_buffers.init_buffer(PhysicsBufferType.MATCHES, INIT_BUFFER_SIZE);
        match_buffers.init_buffer(PhysicsBufferType.MATCHES_USED, INIT_BUFFER_SIZE);

        b_control_point_flags        = new PersistentBuffer(GPU.compute.physics_queue, cl_int.size(), 1);
        b_control_point_indices      = new PersistentBuffer(GPU.compute.physics_queue, cl_int.size(), 1);
        b_control_point_tick_budgets = new PersistentBuffer(GPU.compute.physics_queue, cl_int.size(), 1);
        b_control_point_linear_mag   = new PersistentBuffer(GPU.compute.physics_queue, cl_float.size(), 1);
        b_control_point_jump_mag     = new PersistentBuffer(GPU.compute.physics_queue, cl_float.size(), 1);

        p_integrate.init();
        p_scan_key_bank.init();
        p_build_key_bank.init();
        p_build_key_bank_edge.init();
        p_build_key_map.init();
        p_build_key_map_edge.init();
        p_locate_in_bounds.init();
        p_locate_in_bounds_edge.init();
        p_scan_key_candidates.init();
        p_aabb_collide.init();
        p_sat_collide.init();
        p_animate_hulls.init();
        p_resolve_constraints.init();

        k_integrate = new Integrate_k(GPU.compute.physics_queue, p_integrate).init();
        k_integrate_entities = new IntegrateEntities_k(GPU.compute.physics_queue, p_integrate).init();
        k_calculate_hull_aabb = new CalculateHullAABB_k(GPU.compute.physics_queue, p_integrate).init();

        k_build_key_bank = new BuildKeyBank_k(GPU.compute.physics_queue, p_build_key_bank)
            .init(key_buffers, counts_buf, uniform_grid);

        k_build_key_map = new BuildKeyMap_k(GPU.compute.physics_queue, p_build_key_map)
            .init(key_buffers, offsets_buf, counts_buf, uniform_grid);

        k_locate_in_bounds = (new LocateInBounds_k(GPU.compute.physics_queue, p_locate_in_bounds)).init(candidate_buffers);

        k_count_candidates = new CountCandidates_k(GPU.compute.physics_queue, p_locate_in_bounds)
            .init(candidate_buffers, key_buffers, counts_buf, uniform_grid);

        k_aabb_collide = new AABBCollide_k(GPU.compute.physics_queue, p_aabb_collide)
            .init(match_buffers, candidate_buffers, key_buffers, counts_buf, offsets_buf, atomic_counter, uniform_grid);

        k_finalize_candidates = new FinalizeCandidates_k(GPU.compute.physics_queue, p_locate_in_bounds)
            .init(match_buffers, candidate_buffers);

        k_sat_collide = new SatCollide_k(GPU.compute.physics_queue, p_sat_collide)
            .init(candidate_buffers, reaction_buffers, atomic_counter, FIXED_TIME_STEP);

        k_sort_reactions = new SortReactions_k(GPU.compute.physics_queue, p_sat_collide).init(reaction_buffers);
        k_apply_reactions = new ApplyReactions_k(GPU.compute.physics_queue, p_sat_collide).init(reaction_buffers);

        k_move_hulls = new MoveHulls_k(GPU.compute.physics_queue, p_sat_collide).init();
        k_move_entities = new MoveEntities_k(GPU.compute.physics_queue, p_sat_collide).init(FIXED_TIME_STEP);

        k_animate_entities = new AnimateEntities_k(GPU.compute.physics_queue, p_animate_hulls).init();
        k_animate_bones = new AnimateBones_k(GPU.compute.physics_queue, p_animate_hulls).init();
        k_animate_points = new AnimatePoints_k(GPU.compute.physics_queue, p_animate_hulls).init();
        k_resolve_constraints = new ResolveConstraints_k(GPU.compute.physics_queue, p_resolve_constraints).init();

        k_scan_bounds_single_block = new ScanBoundsSingleBlock_k(GPU.compute.physics_queue, p_scan_key_bank);
        k_scan_bounds_multi_block = new ScanBoundsMultiBlock_k(GPU.compute.physics_queue, p_scan_key_bank);
        k_complete_bounds_multi_block = new CompleteBoundsMultiBlock_k(GPU.compute.physics_queue, p_scan_key_bank);

        k_scan_candidates_single_block_out = new ScanCandidatesSingleBlockOut_k(GPU.compute.physics_queue, p_scan_key_candidates);
        k_scan_candidates_multi_block_out = new ScanCandidatesMultiBlockOut_k(GPU.compute.physics_queue, p_scan_key_candidates);
        k_complete_candidates_multi_block_out = new CompleteCandidatesMultiBlockOut_k(GPU.compute.physics_queue, p_scan_key_candidates);
    }

    //#region Input & Integration

    private void integrate()
    {
        long s = Editor.ACTIVE
            ? System.nanoTime()
            : 0;

        float[] args =
            {
                FIXED_TIME_STEP,
                GRAVITY_X,
                GRAVITY_Y,
                MOTION_DAMPING,
                uniform_grid.sector_origin_x(),
                uniform_grid.sector_origin_y(),
                uniform_grid.sector_width(),
                uniform_grid.sector_height(),
            };

        var arg_mem_buf = GPU.CL.new_cpu_copy_buffer(GPU.compute.context, args);

        int hull_count = GPU.memory.sector_container().next_hull();
        int hull_size = GPU.compute.calculate_preferred_global_size(hull_count);
        int entity_count = GPU.memory.sector_container().next_entity();
        int entity_size = GPU.compute.calculate_preferred_global_size(entity_count);

        k_integrate
            .buf_arg(Integrate_k.Args.args, arg_mem_buf)
            .set_arg(Integrate_k.Args.max_hull, hull_count)
            .call(arg_long(hull_size), GPU.compute.preferred_work_size);

        k_integrate_entities
            .buf_arg(IntegrateEntities_k.Args.args, arg_mem_buf)
            .set_arg(IntegrateEntities_k.Args.max_entity, entity_count)
            .call(arg_long(entity_size), GPU.compute.preferred_work_size);

        arg_mem_buf.release();

        if (Editor.ACTIVE)
        {
            long e = System.nanoTime() - s;
            Editor.queue_event("phys_integrate", String.valueOf(e));
        }
    }

    private void calculate_hull_aabb()
    {

        float[] args =
            {
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
            };

        var arg_mem_buf = GPU.CL.new_cpu_copy_buffer(GPU.compute.context, args);

        long s = Editor.ACTIVE
            ? System.nanoTime()
            : 0;

        int hull_count = GPU.memory.sector_container().next_hull();
        int hull_size = GPU.compute.calculate_preferred_global_size(hull_count);

        k_calculate_hull_aabb
            .buf_arg(CalculateHullAABB_k.Args.args, arg_mem_buf)
            .set_arg(CalculateHullAABB_k.Args.max_hull, hull_count)
            .call(arg_long(hull_size), GPU.compute.preferred_work_size);

        arg_mem_buf.release();

        if (Editor.ACTIVE)
        {
            long e = System.nanoTime() - s;
            Editor.queue_event("phys_calculate_hull_aabb", String.valueOf(e));
        }
    }

    //#endregion

    //#region AABB Collision

    private void calculate_bank_offsets()
    {
        long s = Editor.ACTIVE
            ? System.nanoTime()
            : 0;
        int bank_size = scan_key_bounds(GPU.memory.get_buffer(CoreBufferType.HULL_AABB_KEY_TABLE).pointer(), GPU.memory.sector_container().next_hull());
        uniform_grid.resizeBank(bank_size);
        if (Editor.ACTIVE)
        {
            long e = System.nanoTime() - s;
            Editor.queue_event("phys_bank_offset", String.valueOf(e));
        }
    }

    private int scan_bounds_single_block(long data_ptr, int n)
    {
        long local_buffer_size = cl_int.size() * GPU.compute.max_scan_block_size;

        GPU.CL.zero_buffer(GPU.compute.physics_queue, atomic_counter, cl_int.size());

        k_scan_bounds_single_block
            .ptr_arg(ScanBoundsSingleBlock_k.Args.bounds_bank_data, data_ptr)
            .buf_arg(ScanBoundsSingleBlock_k.Args.sz, atomic_counter)
            .loc_arg(ScanBoundsSingleBlock_k.Args.buffer, local_buffer_size)
            .set_arg(ScanBoundsSingleBlock_k.Args.n, n)
            .call(GPU.compute.local_work_default, GPU.compute.local_work_default);

        return GPU.CL.read_pinned_int(GPU.compute.physics_queue, atomic_counter);
    }

    private int scan_bounds_multi_block(long data_ptr, int n, int k)
    {
        long s = Editor.ACTIVE
            ? System.nanoTime()
            : 0;

        long local_buffer_size = cl_int.size() * GPU.compute.max_scan_block_size;
        long gx = k * GPU.compute.max_scan_block_size;
        long[] global_work_size = arg_long(gx);
        int part_size = k * 2;
        long part_buf_size = ((long) cl_int.size() * ((long) part_size));
        var p_data = GPU.CL.new_buffer(GPU.compute.context, part_buf_size);

        k_scan_bounds_multi_block
            .ptr_arg(ScanBoundsMultiBlock_k.Args.bounds_bank_data, data_ptr)
            .loc_arg(ScanBoundsMultiBlock_k.Args.buffer, local_buffer_size)
            .buf_arg(ScanBoundsMultiBlock_k.Args.part, p_data)
            .set_arg(ScanBoundsMultiBlock_k.Args.n, n)
            .call(global_work_size, GPU.compute.local_work_default);

        if (Editor.ACTIVE)
        {
            long e = System.nanoTime() - s;
            Editor.queue_event("phys_bank_scan_bounds", String.valueOf(e));
        }

        gpu_int_scan.scan_int(p_data.ptr(), part_size);

        GPU.CL.zero_buffer(GPU.compute.physics_queue, atomic_counter, cl_int.size());

        s = Editor.ACTIVE
            ? System.nanoTime()
            : 0;

        k_complete_bounds_multi_block
            .ptr_arg(CompleteBoundsMultiBlock_k.Args.bounds_bank_data, data_ptr)
            .buf_arg(CompleteBoundsMultiBlock_k.Args.sz, atomic_counter)
            .loc_arg(CompleteBoundsMultiBlock_k.Args.buffer, local_buffer_size)
            .buf_arg(CompleteBoundsMultiBlock_k.Args.part, p_data)
            .set_arg(CompleteBoundsMultiBlock_k.Args.n, n)
            .call(global_work_size, GPU.compute.local_work_default);

        p_data.release();

        if (Editor.ACTIVE)
        {
            long e = System.nanoTime() - s;
            Editor.queue_event("phys_bank_complete_bounds", String.valueOf(e));
        }

        s = Editor.ACTIVE
            ? System.nanoTime()
            : 0;
        int r = GPU.CL.read_pinned_int(GPU.compute.physics_queue, atomic_counter);
        if (Editor.ACTIVE)
        {
            long e = System.nanoTime() - s;
            Editor.queue_event("phys_bank_read_pinned", String.valueOf(e));
        }

        return r;
    }

    private int scan_key_bounds(long data_ptr, int n)
    {
        int k = GPU.compute.work_group_count(n);
        if (k == 1)
        {
            return scan_bounds_single_block(data_ptr, n);
        }
        else
        {
            return scan_bounds_multi_block(data_ptr, n, k);
        }
    }

    private void build_key_bank()
    {
        long s = Editor.ACTIVE
            ? System.nanoTime()
            : 0;
        if (uniform_grid.get_key_bank_size() < 1)
        {
            return;
        }

        int hull_count = GPU.memory.sector_container().next_hull();
        int hull_size = GPU.compute.calculate_preferred_global_size(hull_count);

        key_buffers.buffer(PhysicsBufferType.KEY_BANK).ensure_capacity(uniform_grid.get_key_bank_size());
        GPU.CL.zero_buffer(GPU.compute.physics_queue, counts_buf, grid_buffer_size);
        k_build_key_bank
            .set_arg(BuildKeyBank_k.Args.key_bank_length, uniform_grid.get_key_bank_size())
            .set_arg(BuildKeyBank_k.Args.max_hull, hull_count)
            .call(arg_long(hull_size), GPU.compute.preferred_work_size);
        if (Editor.ACTIVE)
        {
            long e = System.nanoTime() - s;
            Editor.queue_event("phys_gen_keys", String.valueOf(e));
        }
    }

    private void build_key_map(UniformGrid uniform_grid)
    {
        long s = Editor.ACTIVE
            ? System.nanoTime()
            : 0;
        key_buffers.buffer(PhysicsBufferType.KEY_MAP).ensure_capacity(uniform_grid.getKey_map_size());
        int hull_count = GPU.memory.sector_container().next_hull();
        int hull_size = GPU.compute.calculate_preferred_global_size(hull_count);
        GPU.CL.zero_buffer(GPU.compute.physics_queue, counts_buf, grid_buffer_size);
        k_build_key_map
            .set_arg(BuildKeyMap_k.Args.max_hull, hull_count)
            .call(arg_long(hull_size), GPU.compute.preferred_work_size);
        if (Editor.ACTIVE)
        {
            long e = System.nanoTime() - s;
            Editor.queue_event("phys_key_map", String.valueOf(e));
        }
    }

    private void locate_in_bounds()
    {
        long s = Editor.ACTIVE
            ? System.nanoTime()
            : 0;
        int hull_count = GPU.memory.sector_container().next_hull();
        int hull_size = GPU.compute.calculate_preferred_global_size(hull_count);
        candidate_buffers.buffer(PhysicsBufferType.IN_BOUNDS).ensure_capacity(hull_count);
        GPU.CL.zero_buffer(GPU.compute.physics_queue, atomic_counter, cl_int.size());

        k_locate_in_bounds
            .buf_arg(LocateInBounds_k.Args.counter, atomic_counter)
            .set_arg(LocateInBounds_k.Args.max_bound, hull_count)
            .call(arg_long(hull_size), GPU.compute.preferred_work_size);

        candidate_buffer_count = GPU.CL.read_pinned_int(GPU.compute.physics_queue, atomic_counter);
        if (Editor.ACTIVE)
        {
            long e = System.nanoTime() - s;
            Editor.queue_event("phys_locate_inbounds", String.valueOf(e));
        }
    }

    private void calculate_match_candidates()
    {
        long s = Editor.ACTIVE
            ? System.nanoTime()
            : 0;
        int candidate_size = GPU.compute.calculate_preferred_global_size((int) candidate_buffer_count);
        candidate_buffers.buffer(PhysicsBufferType.CANDIDATE_COUNTS).ensure_capacity(candidate_buffer_count);
        k_count_candidates
            .set_arg(CountCandidates_k.Args.max_index, (int) candidate_buffer_count)
            .call(arg_long(candidate_size), GPU.compute.preferred_work_size);
        if (Editor.ACTIVE)
        {
            long e = System.nanoTime() - s;
            Editor.queue_event("phys_match_candidates", String.valueOf(e));
        }
    }

    private int scan_single_block_candidates_out(long data_ptr, long o_data_ptr, int n)
    {
        long local_buffer_size = cl_int.size() * GPU.compute.max_scan_block_size;

        GPU.CL.zero_buffer(GPU.compute.physics_queue, atomic_counter, cl_int.size());

        k_scan_candidates_single_block_out
            .ptr_arg(ScanCandidatesSingleBlockOut_k.Args.input, data_ptr)
            .ptr_arg(ScanCandidatesSingleBlockOut_k.Args.output, o_data_ptr)
            .buf_arg(ScanCandidatesSingleBlockOut_k.Args.sz, atomic_counter)
            .loc_arg(ScanCandidatesSingleBlockOut_k.Args.buffer, local_buffer_size)
            .set_arg(ScanCandidatesSingleBlockOut_k.Args.n, n)
            .call(GPU.compute.local_work_default, GPU.compute.local_work_default);

        return GPU.CL.read_pinned_int(GPU.compute.physics_queue, atomic_counter);
    }

    private int scan_multi_block_candidates_out(long data_ptr, long o_data_ptr, int n, int k)
    {
        long local_buffer_size = cl_int.size() * GPU.compute.max_scan_block_size;

        long gx = k * GPU.compute.max_scan_block_size;
        long[] global_work_size = arg_long(gx);
        int part_size = k * 2;
        long part_buf_size = ((long) cl_int.size() * ((long) part_size));
        var p_data = GPU.CL.new_buffer(GPU.compute.context, part_buf_size);

        k_scan_candidates_multi_block_out
            .ptr_arg(ScanCandidatesMultiBlockOut_k.Args.input, data_ptr)
            .ptr_arg(ScanCandidatesMultiBlockOut_k.Args.output, o_data_ptr)
            .loc_arg(ScanCandidatesMultiBlockOut_k.Args.buffer, local_buffer_size)
            .buf_arg(ScanCandidatesMultiBlockOut_k.Args.part, p_data)
            .set_arg(ScanCandidatesMultiBlockOut_k.Args.n, n)
            .call(global_work_size, GPU.compute.local_work_default);

        gpu_int_scan.scan_int(p_data.ptr(), part_size);

        GPU.CL.zero_buffer(GPU.compute.physics_queue, atomic_counter, cl_int.size());

        k_complete_candidates_multi_block_out
            .ptr_arg(CompleteCandidatesMultiBlockOut_k.Args.input, data_ptr)
            .ptr_arg(CompleteCandidatesMultiBlockOut_k.Args.output, o_data_ptr)
            .buf_arg(CompleteCandidatesMultiBlockOut_k.Args.sz, atomic_counter)
            .loc_arg(CompleteCandidatesMultiBlockOut_k.Args.buffer, local_buffer_size)
            .buf_arg(CompleteCandidatesMultiBlockOut_k.Args.part, p_data)
            .set_arg(CompleteCandidatesMultiBlockOut_k.Args.n, n)
            .call(global_work_size, GPU.compute.local_work_default);

        p_data.release();

        return GPU.CL.read_pinned_int(GPU.compute.physics_queue, atomic_counter);
    }

    private int scan_key_candidates(long data_ptr, long o_data_ptr, int n)
    {
        int k = GPU.compute.work_group_count(n);
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
        long s = Editor.ACTIVE
            ? System.nanoTime()
            : 0;
        candidate_buffers.buffer(PhysicsBufferType.CANDIDATE_OFFSETS).ensure_capacity(candidate_buffer_count);
        long counts_ptr = candidate_buffers.buffer(PhysicsBufferType.CANDIDATE_COUNTS).pointer();
        long offsets_ptr = candidate_buffers.buffer(PhysicsBufferType.CANDIDATE_OFFSETS).pointer();
        match_buffer_count = scan_key_candidates(counts_ptr, offsets_ptr, (int) candidate_buffer_count);
        if (Editor.ACTIVE)
        {
            long e = System.nanoTime() - s;
            Editor.queue_event("phys_match_offsets", String.valueOf(e));
        }
    }

    private void aabb_collide()
    {
        long s = Editor.ACTIVE
            ? System.nanoTime()
            : 0;
        int candidate_size = GPU.compute.calculate_preferred_global_size((int) candidate_buffer_count);
        match_buffers.buffer(PhysicsBufferType.MATCHES).ensure_capacity(match_buffer_count);
        match_buffers.buffer(PhysicsBufferType.MATCHES_USED).ensure_capacity(candidate_buffer_count);
        GPU.CL.zero_buffer(GPU.compute.physics_queue, atomic_counter, cl_int.size());
        k_aabb_collide
            .set_arg(AABBCollide_k.Args.max_index, (int) candidate_buffer_count)
            .call(arg_long(candidate_size), GPU.compute.preferred_work_size);
        candidate_count = GPU.CL.read_pinned_int(GPU.compute.physics_queue, atomic_counter);
        if (Editor.ACTIVE)
        {
            long e = System.nanoTime() - s;
            Editor.queue_event("phys_aabb_collide", String.valueOf(e));
        }
    }

    private void finalize_candidates()
    {
        long s = Editor.ACTIVE
            ? System.nanoTime()
            : 0;
        if (candidate_count <= 0)
        {
            return;
        }

        long buffer_size = (long) cl_int2.size() * candidate_count;

        candidate_buffers.buffer(PhysicsBufferType.CANDIDATES).ensure_capacity(candidate_count);

        int[] counter = new int[]{0};
        var counter_buf = GPU.CL.new_int_arg_buffer(GPU.compute.context, counter);

        candidate_buffer_size = buffer_size;

        int candidate_size = GPU.compute.calculate_preferred_global_size((int) candidate_buffer_count);

        k_finalize_candidates
            .buf_arg(FinalizeCandidates_k.Args.counter, counter_buf)
            .set_arg(FinalizeCandidates_k.Args.max_index, (int) candidate_buffer_count)
            .call(arg_long(candidate_size), GPU.compute.preferred_work_size);

        counter_buf.release();
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
        long s = Editor.ACTIVE
            ? System.nanoTime()
            : 0;
        int candidate_pair_size = (int) candidate_buffer_size / cl_int2.size();
        int candidate_size = GPU.compute.calculate_preferred_global_size(candidate_pair_size);
        GPU.CL.zero_buffer(GPU.compute.physics_queue, atomic_counter, cl_int.size());

        long max_point_count = candidate_buffer_size
            * 2  // there are two bodies per collision pair
            * 2; // assume worst case is 2 points per body

        reaction_buffers.buffer(PhysicsBufferType.REACTIONS_IN).ensure_capacity(max_point_count);
        reaction_buffers.buffer(PhysicsBufferType.REACTIONS_OUT).ensure_capacity(max_point_count);
        reaction_buffers.buffer(PhysicsBufferType.REACTION_INDEX).ensure_capacity(max_point_count);
        reaction_buffers.buffer(PhysicsBufferType.POINT_REACTION_COUNTS).ensure_capacity(GPU.memory.sector_container().next_point());
        reaction_buffers.buffer(PhysicsBufferType.POINT_REACTION_OFFSETS).ensure_capacity(GPU.memory.sector_container().next_point());
        k_sat_collide
            .set_arg(SatCollide_k.Args.max_index, candidate_pair_size)
            .call(arg_long(candidate_size), GPU.compute.preferred_work_size);
        reaction_count = GPU.CL.read_pinned_int(GPU.compute.physics_queue, atomic_counter);
        if (Editor.ACTIVE)
        {
            long e = System.nanoTime() - s;
            Editor.queue_event("phys_sat_collide", String.valueOf(e));
        }
    }

    private void scan_reactions()
    {
        long s = Editor.ACTIVE
            ? System.nanoTime()
            : 0;
        long counts_ptr = reaction_buffers.buffer(PhysicsBufferType.POINT_REACTION_COUNTS).pointer();
        long offsets_ptr = reaction_buffers.buffer(PhysicsBufferType.POINT_REACTION_OFFSETS).pointer();
        gpu_int_scan_out.scan_int_out(counts_ptr, offsets_ptr, GPU.memory.sector_container().next_point());
        reaction_buffers.buffer(PhysicsBufferType.POINT_REACTION_COUNTS).clear();
        if (Editor.ACTIVE)
        {
            long e = System.nanoTime() - s;
            Editor.queue_event("phys_sat_scan_reactions", String.valueOf(e));
        }
    }

    private void sort_reactions()
    {
        long s = Editor.ACTIVE
            ? System.nanoTime()
            : 0;
        int reaction_size = GPU.compute.calculate_preferred_global_size((int) reaction_count);
        k_sort_reactions
            .set_arg(SortReactions_k.Args.max_index, (int) reaction_count)
            .call(arg_long(reaction_size), GPU.compute.preferred_work_size);
        if (Editor.ACTIVE)
        {
            long e = System.nanoTime() - s;
            Editor.queue_event("phys_sat_sort_reactions", String.valueOf(e));
        }
    }

    private void apply_reactions()
    {
        long s = Editor.ACTIVE
            ? System.nanoTime()
            : 0;
        int point_count = GPU.memory.sector_container().next_point();
        int point_size = GPU.compute.calculate_preferred_global_size(point_count);
        k_apply_reactions
            .set_arg(ApplyReactions_k.Args.max_point, point_count)
            .call(arg_long(point_size), GPU.compute.preferred_work_size);
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
        long s = Editor.ACTIVE
            ? System.nanoTime()
            : 0;
        int entity_count = GPU.memory.sector_container().next_entity();
        int entity_size = GPU.compute.calculate_preferred_global_size(entity_count);
        k_animate_entities
            .set_arg(AnimateEntities_k.Args.delta_time, dt)
            .set_arg(AnimateEntities_k.Args.max_entity, entity_count)
            .call(arg_long(entity_size), GPU.compute.preferred_work_size);
        if (Editor.ACTIVE)
        {
            long e = System.nanoTime() - s;
            Editor.queue_event("phys_animate_entities", String.valueOf(e));
        }
    }

    private void animate_bones()
    {
        long s = Editor.ACTIVE
            ? System.nanoTime()
            : 0;
        int hull_bone_count = GPU.memory.sector_container().next_hull_bone();
        int hull_bone_size = GPU.compute.calculate_preferred_global_size(hull_bone_count);
        k_animate_bones
            .set_arg(AnimateBones_k.Args.max_hull_bone, hull_bone_count)
            .call(arg_long(hull_bone_size), GPU.compute.preferred_work_size);
        if (Editor.ACTIVE)
        {
            long e = System.nanoTime() - s;
            Editor.queue_event("phys_animate_bones", String.valueOf(e));
        }
    }

    private void animate_points()
    {
        long s = Editor.ACTIVE
            ? System.nanoTime()
            : 0;
        int point_count = GPU.memory.sector_container().next_point();
        int point_size = GPU.compute.calculate_preferred_global_size(point_count);
        k_animate_points
            .set_arg(AnimatePoints_k.Args.max_point, point_count)
            .call(arg_long(point_size), GPU.compute.preferred_work_size);
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
        long s = Editor.ACTIVE
            ? System.nanoTime()
            : 0;
        boolean last_step;
        int hull_count = GPU.memory.sector_container().next_hull();
        int hull_size = GPU.compute.calculate_preferred_global_size(hull_count);
        long[] hull_global_size = arg_long(hull_size);

        for (int i = 0; i < steps; i++)
        {
            last_step = i == steps - 1;
            int n = last_step
                ? 1
                : 0;

            k_resolve_constraints
                .set_arg(ResolveConstraints_k.Args.process_all, n)
                .set_arg(ResolveConstraints_k.Args.max_hull, hull_count)
                .call(hull_global_size, GPU.compute.preferred_work_size);
        }
        if (Editor.ACTIVE)
        {
            long e = System.nanoTime() - s;
            Editor.queue_event("phys_resolve_constraints", String.valueOf(e));
        }
    }

    private void resolve_entities()
    {
        long s = Editor.ACTIVE
            ? System.nanoTime()
            : 0;
        int entity_count = GPU.memory.sector_container().next_entity();
        int entity_size = GPU.compute.calculate_preferred_global_size(entity_count);
        k_move_entities
            .set_arg(MoveEntities_k.Args.max_entity, entity_count)
            .call(arg_long(entity_size), GPU.compute.preferred_work_size);
        if (Editor.ACTIVE)
        {
            long e = System.nanoTime() - s;
            Editor.queue_event("phys_move_entities", String.valueOf(e));
        }
    }

    private void resolve_hulls()
    {
        long s = Editor.ACTIVE
            ? System.nanoTime()
            : 0;
        int hull_count = GPU.memory.sector_container().next_hull();
        int hull_size = GPU.compute.calculate_preferred_global_size(hull_count);
        k_move_hulls
            .set_arg(MoveHulls_k.Args.max_hull, hull_count)
            .call(arg_long(hull_size), GPU.compute.preferred_work_size);
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

        // Axis-Aligned Bounding Boxes (AABBs) are generated first. These bounding boxes used to assign hulls to
        // one or more cells in the uniform grid. This defines a rough area of possible collision that can be mapped
        // into the uniform grid.
        calculate_hull_aabb();

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
        build_key_bank();

        // After keys are generated, the next step is to calculate the space needed for the key map. This is
        // a similar process to calculating the bank offsets. The buffer is zeroed before use to clear out
        // data that is present after the previous tick.
        GPU.CL.zero_buffer(GPU.compute.physics_queue, offsets_buf, grid_buffer_size);
        gpu_int_scan_out.scan_int_out(counts_buf.ptr(), offsets_buf.ptr(), uniform_grid.directory_length);

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

        // In a second pass, candidate counts are scanned to determine the offsets into the match table that
        // correspond to each hull that will be checked for collisions.
        calculate_match_offsets();
        if (Editor.ACTIVE)
        {
            Editor.queue_event("phys_match_buffer_count", String.valueOf(match_buffer_count));
        }
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

        //shift_origins(true);
    }

    private void simulate(float dt)
    {
        long s = Editor.ACTIVE
            ? System.nanoTime()
            : 0;

        //----------------------//
        // Pre-Simulation Setup //
        //----------------------//

        // todo: run translate kernels to move locations into a local co-ordinate space

        // Armatures and bones are animated once per time tick, after all simulation is done for this pass. The interplay between
        // animation and edge constraints may leave points in slightly incorrect positions. Animating here ensures the rendering
        // step always sees the objects exactly in their correct positions.
        animate_entities(dt);
        animate_bones();

        // An initial constraint solve pass is done before simulation to ensure edges are in their "safe"
        // convex shape. Animations may move points into positions where the geometry becomes concave,
        // so this call prevents collision errors due to non-convex shapes.
        //resolve_constraints(EDGE_STEPS);

        //-----------------//
        // Simulation Loop //
        //-----------------//

        this.time_accumulator += dt;
        int sub_ticks = 0;
        float dropped_time;
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
                    player_controller.update_player_state();

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

        // todo: run translate kernels to move locations back into the global co-ordinate space

        // zero out the acceleration buffer, so it is empty for the next frame
        GPU.memory.get_buffer(CoreBufferType.ENTITY_ACCEL).clear();

        long se = Editor.ACTIVE
            ? System.nanoTime()
            : 0;
        // Entities that are exiting the playable area are considered to be "in egress",
        // this step determines how many of each object type is in that state, so they
        // can be transferred into the egress buffer, and eventually onto disk.
        int[] egress_counts = GPU.memory.count_egress_entities();
        GPU.memory.egress(egress_counts);

        if (Editor.ACTIVE)
        {
            long e = System.nanoTime() - se;
            Editor.queue_event("phys_egress", String.valueOf(e));
        }

        // Deletion of objects happens only once per simulation tick, instead of every sub-step
        // to ensure buffer compaction happens as infrequently as possible.
        long sd = Editor.ACTIVE
            ? System.nanoTime()
            : 0;
        GPU.memory.delete_and_compact();
        if (Editor.ACTIVE)
        {
            long e = System.nanoTime() - sd;
            Editor.queue_event("phys_compact", String.valueOf(e));
        }

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
            GPU.compute.render_queue.finish();        // QUEUE: render complete

            long phys_time = last_phys_time.take();     // STATE: main      -> block on   : `phys_time`
            //                                          // STATE: main      -> unblock
            //                                          // STATE: physics   -> block on   : `dt`
            //                                          // STATE: ingress   -> blocked on : `world_barrier`
            //                                          // STATE: egress    -> blocked on : `world_barrier`
            //                                          // STATE: inventory -> blocked on : `world_barrier`

            GPU.memory.await_world_barrier();    // STATE: main      -> block on   : `world_barrier`
            GPU.memory.release_world_barrier();  // STATE: main      -> unblock
            //                                          // STATE: ingress   -> block on   : `dt`
            //                                          // STATE: egress    -> block on   : `dt`
            //                                          // STATE: inventory -> block on   : `dt`

            GPU.compute.sector_queue.finish();        // QUEUE: previous sector processing complete

            /*
             * At this point, all important threads are blocked, and all relevant queues are idle.
             * Now, any data that needs to transfer between ingress/egress/render buffers is moved.
             * Any work that occurs here should be as fast as possible, avoid adding excess latency.
             * */

            GPU.memory.swap_ingress_buffers();
            GPU.memory.swap_egress_buffers();
            GPU.memory.swap_render_buffers();

            GPU.compute.physics_queue.finish();       // QUEUE: current sector processing complete

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

        gpu_int_scan.release();
        gpu_int_scan_out.release();

        candidate_buffers.release();
        key_buffers.release();
        match_buffers.release();
        reaction_buffers.release();

        p_integrate.release();
        p_scan_key_bank.release();
        p_build_key_bank.release();
        p_build_key_bank_edge.release();
        p_build_key_map.release();
        p_build_key_map_edge.release();
        p_locate_in_bounds.release();
        p_locate_in_bounds_edge.release();
        p_scan_key_candidates.release();
        p_aabb_collide.release();
        p_sat_collide.release();
        p_animate_hulls.release();
        p_resolve_constraints.release();

        debug();

        atomic_counter.release();
        counts_buf.release();
        offsets_buf.release();
    }

    //#endregion

    private void debug()
    {
        long total = 0;
        total += b_control_point_flags.debug_data();
        total += b_control_point_indices.debug_data();
        total += b_control_point_tick_budgets.debug_data();
        total += b_control_point_linear_mag.debug_data();
        total += b_control_point_jump_mag.debug_data();

        //System.out.println("---------------------------");
        System.out.println("Physics Memory Usage: MB " + ((float) total / 1024f / 1024f));
    }
}
