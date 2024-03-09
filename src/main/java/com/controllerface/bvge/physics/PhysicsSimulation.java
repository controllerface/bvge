package com.controllerface.bvge.physics;

import com.controllerface.bvge.cl.*;
import com.controllerface.bvge.cl.kernels.*;
import com.controllerface.bvge.cl.programs.*;
import com.controllerface.bvge.ecs.ECS;
import com.controllerface.bvge.ecs.components.*;
import com.controllerface.bvge.ecs.systems.GameSystem;
import org.joml.Vector2f;

import java.util.*;

import static com.controllerface.bvge.cl.CLUtils.arg_long;
import static org.lwjgl.opencl.CL10.clReleaseMemObject;

public class PhysicsSimulation extends GameSystem
{
    private static final float TARGET_FPS = 60.0f;
    private static final float TICK_RATE = 1.0f / TARGET_FPS;
    private static final int TARGET_SUB_STEPS = 8;
    private static final float SUB_STEP = TICK_RATE / TARGET_SUB_STEPS;
    private static final int EDGE_STEPS = 8;
    private static final float GRAVITY_MAGNITUDE = -9.8f * 4;

    private float accumulator = 0.0f;

    // todo: gravity should not be a constant but calculated based on proximity next to planets and other large bodies
    private final static float GRAVITY_X = 0;
    private final static float GRAVITY_Y = GRAVITY_MAGNITUDE * TARGET_FPS;

    // todo: investigate if this should be variable as well. It may make sense to increase damping in some cases,
    //  and lower it in others, for example in space vs on a planet. It may also be useful to set the direction
    //  or make damping interact with the gravity vector in some way.
    private final static float MOTION_DAMPING = .990f;

    private final UniformGrid uniform_grid;
    private PhysicsBuffer physics_buffer;

    /**
     * This buffer is reused each tick to avoid creating a new one every frame and for each object.
     * It should always be zeroed before each use.
     */
    private final Vector2f vectorBuffer = new Vector2f();

    private final GPUProgram integrate = new Integrate();
    private final GPUProgram scan_key_bank = new ScanKeyBank();
    private final GPUProgram generate_keys = new GenerateKeys();
    private final GPUProgram build_key_map = new BuildKeyMap();
    private final GPUProgram locate_in_bounds = new LocateInBounds();
    private final GPUProgram scan_key_candidates = new ScanKeyCandidates();
    private final GPUProgram aabb_collide = new AabbCollide();
    private final GPUProgram sat_collide = new SatCollide();

    private GPUKernel integrate_k;
    private GPUKernel scan_bounds_single_block_k;
    private GPUKernel scan_bounds_multi_block_k;
    private GPUKernel complete_bounds_multi_block_k;
    private GPUKernel generate_keys_k;
    private GPUKernel build_key_map_k;
    private GPUKernel locate_in_bounds_k;
    private GPUKernel count_candidates_k;

    private GPUKernel scan_candidates_single_block_out_k;
    private GPUKernel scan_candidates_multi_block_out_k;
    private GPUKernel complete_candidates_multi_block_out_k;

    private GPUKernel aabb_collide_k;
    private GPUKernel finalize_candidates_k;

    private GPUKernel sat_collide_k;
    private GPUKernel sort_reactions_k;
    private GPUKernel apply_reactions_k;
    private GPUKernel move_armatures_k;


    public PhysicsSimulation(ECS ecs, UniformGrid uniform_grid)
    {
        super(ecs);
        this.uniform_grid = uniform_grid;
        init_kernels();
    }

    private void init_kernels()
    {
        integrate.init();
        scan_key_bank.init();
        generate_keys.init();
        build_key_map.init();
        locate_in_bounds.init();
        scan_key_candidates.init();
        aabb_collide.init();
        sat_collide.init();

        long integrate_k_ptr = integrate.kernel_ptr(GPU.Kernel.integrate);
        integrate_k = new Integrate_k(GPU.command_queue_ptr, integrate_k_ptr)
            .mem_arg(Integrate_k.Args.hulls, GPU.Buffer.hulls.memory)
            .mem_arg(Integrate_k.Args.armatures, GPU.Buffer.armatures.memory)
            .mem_arg(Integrate_k.Args.armature_flags, GPU.Buffer.armature_flags.memory)
            .mem_arg(Integrate_k.Args.element_tables, GPU.Buffer.hull_element_tables.memory)
            .mem_arg(Integrate_k.Args.armature_accel, GPU.Buffer.armature_accel.memory)
            .mem_arg(Integrate_k.Args.hull_rotations, GPU.Buffer.hull_rotation.memory)
            .mem_arg(Integrate_k.Args.points, GPU.Buffer.points.memory)
            .mem_arg(Integrate_k.Args.bounds, GPU.Buffer.aabb.memory)
            .mem_arg(Integrate_k.Args.bounds_index_data, GPU.Buffer.aabb_index.memory)
            .mem_arg(Integrate_k.Args.bounds_bank_data, GPU.Buffer.aabb_key_table.memory)
            .mem_arg(Integrate_k.Args.hull_flags, GPU.Buffer.hull_flags.memory)
            .mem_arg(Integrate_k.Args.anti_gravity, GPU.Buffer.point_anti_gravity.memory);

        long scan_bounds_single_block_k_ptr = scan_key_bank.kernel_ptr(GPU.Kernel.scan_bounds_single_block);
        scan_bounds_single_block_k = new ScanBoundsSingleBlock_k(GPU.command_queue_ptr, scan_bounds_single_block_k_ptr);

        long scan_bounds_multi_block_k_ptr = scan_key_bank.kernel_ptr(GPU.Kernel.scan_bounds_multi_block);
        scan_bounds_multi_block_k = new ScanBoundsMultiBlock_k(GPU.command_queue_ptr, scan_bounds_multi_block_k_ptr);

        long complete_bounds_multi_block_k_ptr = scan_key_bank.kernel_ptr(GPU.Kernel.complete_bounds_multi_block);
        complete_bounds_multi_block_k = new CompleteBoundsMultiBlock_k(GPU.command_queue_ptr, complete_bounds_multi_block_k_ptr);

        long generate_keys_k_ptr = generate_keys.kernel_ptr(GPU.Kernel.generate_keys);
        generate_keys_k = new GenerateKeys_k(GPU.command_queue_ptr, generate_keys_k_ptr)
            .mem_arg(GenerateKeys_k.Args.bounds_index_data, GPU.Buffer.aabb_index.memory)
            .mem_arg(GenerateKeys_k.Args.bounds_bank_data, GPU.Buffer.aabb_key_table.memory);

        long build_key_map_k_ptr = build_key_map.kernel_ptr(GPU.Kernel.build_key_map);
        build_key_map_k = new BuildKeyMap_k(GPU.command_queue_ptr, build_key_map_k_ptr)
            .mem_arg(BuildKeyMap_k.Args.bounds_index_data, GPU.Buffer.aabb_index.memory)
            .mem_arg(BuildKeyMap_k.Args.bounds_bank_data, GPU.Buffer.aabb_key_table.memory);

        long locate_in_bounds_k_ptr = locate_in_bounds.kernel_ptr(GPU.Kernel.locate_in_bounds);
        locate_in_bounds_k = (new LocateInBounds_k(GPU.command_queue_ptr, locate_in_bounds_k_ptr))
            .mem_arg(LocateInBounds_k.Args.bounds_bank_data, GPU.Buffer.aabb_key_table.memory);

        long count_candidates_k_ptr = locate_in_bounds.kernel_ptr(GPU.Kernel.count_candidates);
        count_candidates_k = new CountCandidates_k(GPU.command_queue_ptr, count_candidates_k_ptr)
            .mem_arg(CountCandidates_k.Args.bounds_bank_data, GPU.Buffer.aabb_key_table.memory);

        long scan_candidates_single_block_out_k_ptr = scan_key_candidates.kernel_ptr(GPU.Kernel.scan_candidates_single_block_out);
        scan_candidates_single_block_out_k = new ScanCandidatesSingleBlockOut_k(GPU.command_queue_ptr, scan_candidates_single_block_out_k_ptr);

        long scan_candidates_multi_block_out_k_ptr = scan_key_candidates.kernel_ptr(GPU.Kernel.scan_candidates_multi_block_out);
        scan_candidates_multi_block_out_k = new ScanCandidatesMultiBlockOut_k(GPU.command_queue_ptr, scan_candidates_multi_block_out_k_ptr);

        long complete_candidates_multi_block_out_k_ptr = scan_key_candidates.kernel_ptr(GPU.Kernel.complete_candidates_multi_block_out);
        complete_candidates_multi_block_out_k = new CompleteCandidatesMultiBlockOut_k(GPU.command_queue_ptr, complete_candidates_multi_block_out_k_ptr);

        long aabb_collide_k_ptr = aabb_collide.kernel_ptr(GPU.Kernel.aabb_collide);
        aabb_collide_k = new AABBCollide_k(GPU.command_queue_ptr, aabb_collide_k_ptr)
            .mem_arg(AABBCollide_k.Args.bounds, GPU.Buffer.aabb.memory)
            .mem_arg(AABBCollide_k.Args.bounds_bank_data, GPU.Buffer.aabb_key_table.memory)
            .mem_arg(AABBCollide_k.Args.hull_flags, GPU.Buffer.hull_flags.memory);

        long finalize_candidates_k_ptr = locate_in_bounds.kernel_ptr(GPU.Kernel.finalize_candidates);
        finalize_candidates_k = new FinalizeCandidates_k(GPU.command_queue_ptr, finalize_candidates_k_ptr);

        long sat_collide_k_ptr = sat_collide.kernel_ptr(GPU.Kernel.sat_collide);
        sat_collide_k = new SatCollide_k(GPU.command_queue_ptr, sat_collide_k_ptr)
            .ptr_arg(SatCollide_k.Args.counter, GPU.atomic_counter_ptr)
            .mem_arg(SatCollide_k.Args.hulls, GPU.Buffer.hulls.memory)
            .mem_arg(SatCollide_k.Args.element_tables, GPU.Buffer.hull_element_tables.memory)
            .mem_arg(SatCollide_k.Args.hull_flags, GPU.Buffer.hull_flags.memory)
            .mem_arg(SatCollide_k.Args.vertex_tables, GPU.Buffer.point_vertex_tables.memory)
            .mem_arg(SatCollide_k.Args.points, GPU.Buffer.points.memory)
            .mem_arg(SatCollide_k.Args.edges, GPU.Buffer.edges.memory)
            .mem_arg(SatCollide_k.Args.point_reactions, GPU.Buffer.point_reactions.memory)
            .mem_arg(SatCollide_k.Args.masses, GPU.Buffer.armature_mass.memory);

        long sort_reactions_k_ptr = sat_collide.kernel_ptr(GPU.Kernel.sort_reactions);
        sort_reactions_k = new SortReactions_k(GPU.command_queue_ptr, sort_reactions_k_ptr)
            .mem_arg(SortReactions_k.Args.point_reactions, GPU.Buffer.point_reactions.memory)
            .mem_arg(SortReactions_k.Args.point_offsets, GPU.Buffer.point_offsets.memory);

        long apply_reactions_k_ptr = sat_collide.kernel_ptr(GPU.Kernel.apply_reactions);
        apply_reactions_k = new ApplyReactions_k(GPU.command_queue_ptr, apply_reactions_k_ptr)
            .mem_arg(ApplyReactions_k.Args.points, GPU.Buffer.points.memory)
            .mem_arg(ApplyReactions_k.Args.anti_gravity, GPU.Buffer.point_anti_gravity.memory)
            .mem_arg(ApplyReactions_k.Args.point_reactions, GPU.Buffer.point_reactions.memory)
            .mem_arg(ApplyReactions_k.Args.point_offsets, GPU.Buffer.point_offsets.memory);

        long move_armatures_k_ptr = sat_collide.kernel_ptr(GPU.Kernel.move_armatures);
        move_armatures_k = new MoveArmatures_k(GPU.command_queue_ptr, move_armatures_k_ptr)
            .mem_arg(MoveArmatures_k.Args.hulls, GPU.Buffer.hulls.memory)
            .mem_arg(MoveArmatures_k.Args.armatures, GPU.Buffer.armatures.memory)
            .mem_arg(MoveArmatures_k.Args.hull_tables, GPU.Buffer.armature_hull_table.memory)
            .mem_arg(MoveArmatures_k.Args.element_tables, GPU.Buffer.hull_element_tables.memory)
            .mem_arg(MoveArmatures_k.Args.hull_flags, GPU.Buffer.hull_flags.memory)
            .mem_arg(MoveArmatures_k.Args.points, GPU.Buffer.points.memory);
    }

    private void integrate(float delta_time)
    {
        float[] args =
            {
                delta_time,
                uniform_grid.x_spacing,
                uniform_grid.y_spacing,
                uniform_grid.getX_origin(),
                uniform_grid.getY_origin(),
                uniform_grid.width,
                uniform_grid.height,
                (float) uniform_grid.x_subdivisions,
                (float) uniform_grid.y_subdivisions,
                physics_buffer.get_gravity_x(),
                physics_buffer.get_gravity_y(),
                physics_buffer.get_damping()
            };

        var arg_mem_ptr = GPU.cl_new_cpu_copy_buffer(args);

        integrate_k
            .ptr_arg(Integrate_k.Args.args, arg_mem_ptr)
            .call(arg_long(GPU.core_memory.next_hull()));

        GPU.release_buffer(arg_mem_ptr);
    }

    private int scan_bounds_single_block(long data_ptr, int n)
    {
        long local_buffer_size = CLSize.cl_int * GPU.max_scan_block_size;

        GPU.cl_zero_buffer(GPU.atomic_counter_ptr, CLSize.cl_int);

        scan_bounds_single_block_k
            .ptr_arg(ScanBoundsSingleBlock_k.Args.bounds_bank_data, data_ptr)
            .ptr_arg(ScanBoundsSingleBlock_k.Args.sz, GPU.atomic_counter_ptr)
            .loc_arg(ScanBoundsSingleBlock_k.Args.buffer, local_buffer_size)
            .set_arg(ScanBoundsSingleBlock_k.Args.n, n)
            .call(GPU.local_work_default, GPU.local_work_default);

        return GPU.cl_read_pinned_int(GPU.atomic_counter_ptr);
    }

    private int scan_bounds_multi_block(long data_ptr, int n, int k)
    {
        long local_buffer_size = CLSize.cl_int * GPU.max_scan_block_size;
        long gx = k * GPU.max_scan_block_size;
        long[] global_work_size = arg_long(gx);
        int part_size = k * 2;
        long part_buf_size = ((long) CLSize.cl_int * ((long) part_size));
        var p_data = GPU.cl_new_buffer(part_buf_size);

        scan_bounds_multi_block_k
            .ptr_arg(ScanBoundsMultiBlock_k.Args.bounds_bank_data, data_ptr)
            .loc_arg(ScanBoundsMultiBlock_k.Args.buffer, local_buffer_size)
            .ptr_arg(ScanBoundsMultiBlock_k.Args.part, p_data)
            .set_arg(ScanBoundsMultiBlock_k.Args.n, n)
            .call(global_work_size, GPU.local_work_default);

        GPU.scan_int(p_data, part_size);

        GPU.cl_zero_buffer(GPU.atomic_counter_ptr, CLSize.cl_int);

        complete_bounds_multi_block_k
            .ptr_arg(CompleteBoundsMultiBlock_k.Args.bounds_bank_data, data_ptr)
            .ptr_arg(CompleteBoundsMultiBlock_k.Args.sz, GPU.atomic_counter_ptr)
            .loc_arg(CompleteBoundsMultiBlock_k.Args.buffer, local_buffer_size)
            .ptr_arg(CompleteBoundsMultiBlock_k.Args.part, p_data)
            .set_arg(CompleteBoundsMultiBlock_k.Args.n, n)
            .call(global_work_size, GPU.local_work_default);

        GPU.release_buffer(p_data);

        return GPU.cl_read_pinned_int(GPU.atomic_counter_ptr);
    }

    private int scan_key_bounds(long data_ptr, int n)
    {
        int k = GPU.work_group_count(n);
        if (k == 1)
        {
            return scan_bounds_single_block(data_ptr, n);
        }
        else
        {
            return scan_bounds_multi_block(data_ptr, n, k);
        }
    }

    private void calculate_bank_offsets()
    {
        int bank_size = scan_key_bounds(GPU.Buffer.aabb_key_table.memory.pointer(), GPU.core_memory.next_hull());
        uniform_grid.resizeBank(bank_size);
    }

    private void generate_keys()
    {
        if (uniform_grid.get_key_bank_size() < 1)
        {
            return;
        }

        long bank_buf_size = (long) CLSize.cl_int * uniform_grid.get_key_bank_size();
        long bank_data_ptr = GPU.cl_new_buffer(bank_buf_size);

        // set the counts buffer to all zeroes
        GPU.cl_zero_buffer(GPU.counts_data_ptr, GPU.counts_buf_size);

        physics_buffer.key_bank = new GPUMemory(bank_data_ptr);

        generate_keys_k
            .ptr_arg(GenerateKeys_k.Args.key_bank, physics_buffer.key_bank.pointer())
            .set_arg(GenerateKeys_k.Args.key_bank_length, uniform_grid.get_key_bank_size())
            .call(arg_long(GPU.core_memory.next_hull()));
    }

    private void build_key_map(UniformGrid uniform_grid)
    {
        long map_buf_size = (long) CLSize.cl_int * uniform_grid.getKey_map_size();

        var map_data = GPU.cl_new_buffer(map_buf_size);

        // reset the counts buffer to all zeroes
        GPU.cl_zero_buffer(GPU.counts_data_ptr, GPU.counts_buf_size);

        physics_buffer.key_map = new GPUMemory(map_data);

        build_key_map_k
            .ptr_arg(BuildKeyMap_k.Args.key_map, map_data)
            .call(arg_long(GPU.core_memory.next_hull()));
    }

    private void locate_in_bounds()
    {
        int hull_count = GPU.core_memory.next_hull();

        long inbound_buf_size = (long) CLSize.cl_int * hull_count;
        var inbound_data = GPU.cl_new_buffer(inbound_buf_size);

        physics_buffer.in_bounds = new GPUMemory(inbound_data);

        GPU.cl_zero_buffer(GPU.atomic_counter_ptr, CLSize.cl_int);

        locate_in_bounds_k
            .ptr_arg(LocateInBounds_k.Args.in_bounds, physics_buffer.in_bounds.pointer())
            .ptr_arg(LocateInBounds_k.Args.counter, GPU.atomic_counter_ptr)
            .call(arg_long(hull_count));

        int size = GPU.cl_read_pinned_int(GPU.atomic_counter_ptr);

        physics_buffer.set_candidate_buffer_count(size);
    }

    private void calculate_match_candidates()
    {
        long candidate_buf_size = (long) CLSize.cl_int2 * physics_buffer.get_candidate_buffer_count();
        var candidate_data = GPU.cl_new_buffer(candidate_buf_size);
        physics_buffer.candidate_counts = new GPUMemory(candidate_data);

        count_candidates_k
            .ptr_arg(CountCandidates_k.Args.in_bounds, physics_buffer.in_bounds.pointer())
            .ptr_arg(CountCandidates_k.Args.key_bank, physics_buffer.key_bank.pointer())
            .ptr_arg(CountCandidates_k.Args.candidates, physics_buffer.candidate_counts.pointer())
            .call(arg_long(physics_buffer.get_candidate_buffer_count()));
    }

    private int scan_single_block_candidates_out(long data_ptr, long o_data_ptr, int n)
    {
        long local_buffer_size = CLSize.cl_int * GPU.max_scan_block_size;

        GPU.cl_zero_buffer(GPU.atomic_counter_ptr, CLSize.cl_int);

        scan_candidates_single_block_out_k
            .ptr_arg(ScanCandidatesSingleBlockOut_k.Args.input, data_ptr)
            .ptr_arg(ScanCandidatesSingleBlockOut_k.Args.output, o_data_ptr)
            .ptr_arg(ScanCandidatesSingleBlockOut_k.Args.sz, GPU.atomic_counter_ptr)
            .loc_arg(ScanCandidatesSingleBlockOut_k.Args.buffer, local_buffer_size)
            .set_arg(ScanCandidatesSingleBlockOut_k.Args.n, n)
            .call(GPU.local_work_default, GPU.local_work_default);

        return GPU.cl_read_pinned_int(GPU.atomic_counter_ptr);
    }

    private int scan_multi_block_candidates_out(long data_ptr, long o_data_ptr, int n, int k)
    {
        long local_buffer_size = CLSize.cl_int * GPU.max_scan_block_size;

        long gx = k * GPU.max_scan_block_size;
        long[] global_work_size = arg_long(gx);
        int part_size = k * 2;
        long part_buf_size = ((long) CLSize.cl_int * ((long) part_size));
        var p_data = GPU.cl_new_buffer(part_buf_size);

        scan_candidates_multi_block_out_k
            .ptr_arg(ScanCandidatesMultiBlockOut_k.Args.input, data_ptr)
            .ptr_arg(ScanCandidatesMultiBlockOut_k.Args.output, o_data_ptr)
            .loc_arg(ScanCandidatesMultiBlockOut_k.Args.buffer, local_buffer_size)
            .ptr_arg(ScanCandidatesMultiBlockOut_k.Args.part, p_data)
            .set_arg(ScanCandidatesMultiBlockOut_k.Args.n, n)
            .call(global_work_size, GPU.local_work_default);

        GPU.scan_int(p_data, part_size);

        GPU.cl_zero_buffer(GPU.atomic_counter_ptr, CLSize.cl_int);

        complete_candidates_multi_block_out_k
            .ptr_arg(CompleteCandidatesMultiBlockOut_k.Args.input, data_ptr)
            .ptr_arg(CompleteCandidatesMultiBlockOut_k.Args.output, o_data_ptr)
            .ptr_arg(CompleteCandidatesMultiBlockOut_k.Args.sz, GPU.atomic_counter_ptr)
            .loc_arg(CompleteCandidatesMultiBlockOut_k.Args.buffer, local_buffer_size)
            .ptr_arg(CompleteCandidatesMultiBlockOut_k.Args.part, p_data)
            .set_arg(CompleteCandidatesMultiBlockOut_k.Args.n, n)
            .call(global_work_size, GPU.local_work_default);

        GPU.release_buffer(p_data);

        return GPU.cl_read_pinned_int(GPU.atomic_counter_ptr);
    }

    private int scan_key_candidates(long data_ptr, long o_data_ptr, int n)
    {
        int k = GPU.work_group_count(n);
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
        int buffer_count = physics_buffer.get_candidate_buffer_count();
        long offset_buf_size = (long) CLSize.cl_int * buffer_count;
        var offset_data = GPU.cl_new_buffer(offset_buf_size);
        physics_buffer.candidate_offsets = new GPUMemory(offset_data);
        int match_count = scan_key_candidates(physics_buffer.candidate_counts.pointer(), offset_data, buffer_count);
        physics_buffer.set_candidate_match_count(match_count);
    }

    private void aabb_collide()
    {
        long matches_buf_size = (long) CLSize.cl_int * physics_buffer.get_candidate_match_count();
        var matches_data = GPU.cl_new_buffer(matches_buf_size);
        physics_buffer.matches = new GPUMemory(matches_data);

        long used_buf_size = (long) CLSize.cl_int * physics_buffer.get_candidate_buffer_count();
        var used_data = GPU.cl_new_buffer(used_buf_size);
        physics_buffer.matches_used = new GPUMemory(used_data);

        GPU.cl_zero_buffer(GPU.atomic_counter_ptr, CLSize.cl_int);

        aabb_collide_k
            .ptr_arg(AABBCollide_k.Args.candidates, physics_buffer.candidate_counts.pointer())
            .ptr_arg(AABBCollide_k.Args.match_offsets, physics_buffer.candidate_offsets.pointer())
            .ptr_arg(AABBCollide_k.Args.key_map, physics_buffer.key_map.pointer())
            .ptr_arg(AABBCollide_k.Args.key_bank, physics_buffer.key_bank.pointer())
            .ptr_arg(AABBCollide_k.Args.matches, physics_buffer.matches.pointer())
            .ptr_arg(AABBCollide_k.Args.used, physics_buffer.matches_used.pointer())
            .call(arg_long(physics_buffer.get_candidate_buffer_count()));

        int count = GPU.cl_read_pinned_int(GPU.atomic_counter_ptr);
        physics_buffer.set_candidate_count(count);
    }

    private void finalize_candidates()
    {
        if (physics_buffer.get_candidate_count() <= 0)
        {
            return;
        }

        // create an empty buffer that the kernel will use to store finalized candidates
        long final_buf_size = (long) CLSize.cl_int2 * physics_buffer.get_candidate_count();
        var finals_data = GPU.cl_new_buffer(final_buf_size);

        // the kernel will use this value as an internal atomic counter, always initialize to zero
        int[] counter = new int[]{ 0 };
        var counter_ptr = GPU.cl_new_int_arg_buffer(counter);

        physics_buffer.set_final_size(final_buf_size);
        physics_buffer.candidates = new GPUMemory(finals_data);

        finalize_candidates_k
            .ptr_arg(FinalizeCandidates_k.Args.input_candidates, physics_buffer.candidate_counts.pointer())
            .ptr_arg(FinalizeCandidates_k.Args.match_offsets, physics_buffer.candidate_offsets.pointer())
            .ptr_arg(FinalizeCandidates_k.Args.matches, physics_buffer.matches.pointer())
            .ptr_arg(FinalizeCandidates_k.Args.used, physics_buffer.matches_used.pointer())
            .ptr_arg(FinalizeCandidates_k.Args.counter, counter_ptr)
            .ptr_arg(FinalizeCandidates_k.Args.final_candidates, physics_buffer.candidates.pointer())
            .call(arg_long(physics_buffer.get_candidate_buffer_count()));

        GPU.release_buffer(counter_ptr);
    }





    private void sat_collide()
    {
        int candidates_size = (int) physics_buffer.get_final_size() / CLSize.cl_int;

        // candidates are pairs of integer indices, so the global size is half the count
        long[] global_work_size = new long[]{candidates_size / 2};

        GPU.cl_zero_buffer(GPU.atomic_counter_ptr, CLSize.cl_int);

        long max_point_count = physics_buffer.get_final_size()
            * 2  // there are two bodies per collision pair
            * 2; // assume worst case is 2 points per body

        // sizes for the reaction buffers
        long reaction_buf_size = (long) CLSize.cl_float2 * max_point_count;
        long index_buf_size = (long) CLSize.cl_int * max_point_count;

        if (reaction_buf_size > GPU.reaction_buf_size
            || index_buf_size > GPU.index_buf_size)
        {
            GPU.reactions_in.release();
            GPU.reactions_out.release();
            GPU.reaction_index.release();

            GPU.reaction_buf_size = reaction_buf_size;
            GPU.index_buf_size = index_buf_size;

            var reaction_data = GPU.cl_new_buffer(reaction_buf_size);
            var reaction_data_out = GPU.cl_new_buffer(reaction_buf_size);
            var index_data = GPU.cl_new_buffer(index_buf_size);

            GPU.reactions_in = new GPUMemory(reaction_data);
            GPU.reactions_out = new GPUMemory(reaction_data_out);
            GPU.reaction_index = new GPUMemory(index_data);
        }

        sat_collide_k
            .ptr_arg(SatCollide_k.Args.candidates, physics_buffer.candidates.pointer())
            .ptr_arg(SatCollide_k.Args.reactions, GPU.reactions_in.pointer())
            .ptr_arg(SatCollide_k.Args.reaction_index, GPU.reaction_index.pointer())
            .call(global_work_size);

        int size = GPU.cl_read_pinned_int(GPU.atomic_counter_ptr);
        physics_buffer.set_reaction_count(size);
    }

    private void scan_reactions()
    {
        GPU.scan_int_out(GPU.Buffer.point_reactions.memory.pointer(), GPU.Buffer.point_offsets.memory.pointer(), GPU.core_memory.next_point());
        // it is important to zero out the reactions buffer after the scan. It will be reused during sorting
        GPU.Buffer.point_reactions.clear();
    }

    private void sort_reactions()
    {
        sort_reactions_k
            .ptr_arg(SortReactions_k.Args.reactions_in, GPU.reactions_in.pointer())
            .ptr_arg(SortReactions_k.Args.reactions_out, GPU.reactions_out.pointer())
            .ptr_arg(SortReactions_k.Args.reaction_index, GPU.reaction_index.pointer())
            .call(arg_long(physics_buffer.get_reaction_count()));
    }

    private void apply_reactions()
    {
        apply_reactions_k
            .ptr_arg(ApplyReactions_k.Args.reactions, GPU.reactions_out.pointer())
            .call(arg_long(GPU.core_memory.next_point()));
    }

    private void move_armatures()
    {
        move_armatures_k.call(arg_long(GPU.core_memory.next_armature()));
    }




    private void updateControllableBodies(float dt)
    {
        var components = ecs.getComponents(Component.ControlPoints);
        for (Map.Entry<String, GameComponent> entry : components.entrySet())
        {
            String entity = entry.getKey();
            GameComponent component = entry.getValue();
            ControlPoints controlPoints = Component.ControlPoints.coerce(component);
            ArmatureIndex armature = Component.Armature.forEntity(ecs, entity);
            LinearForce force = Component.LinearForce.forEntity(ecs, entity);

            Objects.requireNonNull(controlPoints);
            Objects.requireNonNull(armature);
            Objects.requireNonNull(force);

            vectorBuffer.zero();
            if (controlPoints.is_moving_left())
            {
                vectorBuffer.x -= force.magnitude();
            }
            if (controlPoints.is_moving_right())
            {
                vectorBuffer.x += force.magnitude();
            }
            if (controlPoints.is_moving_up())
            {
                vectorBuffer.y += force.magnitude();
            }
            if (controlPoints.is_moving_down())
            {
                vectorBuffer.y -= force.magnitude();
            }
            if (controlPoints.is_space_bar_down())
            {
                vectorBuffer.y -= GRAVITY_Y;
            }

            if (vectorBuffer.x != 0f || vectorBuffer.y != 0)
            {
                GPU.core_memory.update_accel(armature.index(), vectorBuffer.x, vectorBuffer.y);
            }

            // todo: implement this for armatures
//            if (controlPoints.is_rotating_right() ^ controlPoints.is_rotating_left())
//            {
//                float angle = controlPoints.is_rotating_right() ? -200f : 200f;
//                OpenCL.rotate_hull(hull.index(), angle * dt * dt);
//            }
        }
    }


    /**
     * This is the core of the physics simulation. Upon return from this method, the simulation is
     * advanced one tick. Note that this class uses a fixed time step, so the time delta should always
     * be the same. Most work done within this method is delegated to the GPU for performance.
     *
     * @param dt amount of time that is simulated during the physics tick.
     */
    private void tickSimulation(float dt)
    {
        /*
        * CPU Side - Setup
        * */

        // Before the GPU begins the simulation cycle, player input is handled and the memory structures
        // in the GPU are updated with the proper values.
        updateControllableBodies(dt);

        /*
        * GPU Side - Physics
        * */

        // The first order of business is to perform the mathematical steps required to calculate where the
        // individual points of each hull currently are. When this call returns, all tracked physics objects
        // will be in their new locations, and points will have their current and previous location values
        // updated for this tick cycle.
        integrate(dt);

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
        // a similar process to calculating the bank offsets.
        GPU.scan_int_out(GPU.counts_data_ptr, GPU.offsets_data_ptr, uniform_grid.directory_length);

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

        // Finally, the actual broad phase collision check is performed. Once complete, the match table will
        // be filled in with all matches. There may be some unused sections of the table, because some objects
        // may be eliminated during the check.
        aabb_collide();

        // This last step cleans up the match table, retaining only the used sections of the buffer.
        // After this step, the matches are ready for the narrow phase check.
        finalize_candidates();

        // If there were no candidate collisions, there's nothing left to do
        if (physics_buffer.candidates == null)
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
        to their parent armatures. This last step is needed for complex models, to ensure the groups of hulls
        are moved together as a unit. Without this step, armature based objects will not collide correctly.
        */

        // Using the candidates generated by the AABB checks, we now do a full collision check. For any objects
        // that are found to be colliding, appropriate reaction vectors are generated and stored.
        sat_collide();

        // It is possible that after the check, no objects are found to be colliding. If that happens, exit.
        if (physics_buffer.get_reaction_count() == 0)
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

        // Once all points have been relocated, all hulls are in their required positions for this frame.
        // Movements applied to hulls are now accumulated and applied to their parent armatures.
        move_armatures();
    }

    private void simulate(float dt)
    {
        var armatures = ecs.getComponents(Component.Armature);

        // if there are no armatures, just bail. things may still be setting up
        if (armatures == null || armatures.isEmpty())
        {
            return;
        }

        // Bones are animated once per time tick
        GPU.animate_armatures(dt);
        GPU.animate_bones();

        // An initial constraint solve pass is done before simulation to ensure edges are in their "safe"
        // convex shape. Animations may move points into positions where the geometry is slightly concave,
        // so this call acts as a small hedge against this happening before collision checks can be performed.
        GPU.resolve_constraints(EDGE_STEPS);

        this.accumulator += dt;
        int sub_ticks = 0;
        float skipped = 0f;
        while (this.accumulator >= TICK_RATE)
        {
            for (int i = 0; i < TARGET_SUB_STEPS; i++)
            {
                sub_ticks++;

                // if we end up doing more sub ticks than is ideal, we will avoid ticking the simulation anymore
                // for this frame. This forces slower hardware to slow down a bit, which is less than ideal, but
                // is better than the alternative, which is system lockup.
                // todo: test a few different values on some lower-end hardware and try to find a sweet spot.
                if (sub_ticks <= TARGET_SUB_STEPS)
                {
                    this.accumulator -= SUB_STEP;
                    this.tickSimulation(SUB_STEP);

                    // Now we make a call to animate the vertices of bone-tracked hulls. This ensures that all tracked
                    // objects that have animation will have their hulls moved into position for the current tick. It
                    // may seem odd to process animations as part of physics and not rendering, however this is required
                    // as the animated objects need to be accounted for in physical space. The hulls representing the
                    // rendered meshes are what is actually moved, and the result of the hull movement is used to position
                    // the original mesh for rendering. This separation is necessary as model geometry is too complex to
                    // use as a collision boundary.
                    GPU.animate_points();

                    // Once positions are adjusted, edge constraints are enforced to ensure that rigid bodies maintain
                    // their defined shapes. Without this step, the individual points of the tracked physics hulls will
                    // deform on impact, and may fly off in random directions, typically causing simulation failure. The
                    // number of steps that are performed each tick has an impact on the accuracy of the hull boundaries
                    // within the simulation.
                    GPU.resolve_constraints(EDGE_STEPS);

                    physics_buffer.finishTick();
                }
                else
                {
                    skipped = this.accumulator;
                    this.accumulator = 0;
                }
            }
        }

        if (skipped > 0f)
        {
            System.out.println("skipped: " + skipped);
        }

        // Deletion of objects happens only once per simulation cycle, instead of every tick
        // to ensure buffer compaction happens as infrequently as possible.
        GPU.core_memory.delete_and_compact();

        // After all simulation is done for this pass, do one last animate pass so that vertices are all in
        // the expected location for rendering. The interplay between animation and edge constraints may leave
        // the points in slightly incorrect positions. This makes sure everything is good for the render step.
        GPU.animate_points();
    }

    @Override
    public void tick(float dt)
    {
        if (physics_buffer == null)
        {
            physics_buffer = new PhysicsBuffer();
            physics_buffer.set_gravity_x(GRAVITY_X);
            physics_buffer.set_gravity_y(GRAVITY_Y);
            physics_buffer.set_damping(MOTION_DAMPING);
            GPU.set_physics_buffer(physics_buffer);
            GPU.set_uniform_grid_constants(uniform_grid);

            generate_keys_k
                .ptr_arg(GenerateKeys_k.Args.key_counts, GPU.counts_data_ptr)
                .set_arg(GenerateKeys_k.Args.x_subdivisions, uniform_grid.x_subdivisions)
                .set_arg(GenerateKeys_k.Args.key_count_length, uniform_grid.directory_length);

            build_key_map_k
                .ptr_arg(BuildKeyMap_k.Args.key_offsets, GPU.offsets_data_ptr)
                .ptr_arg(BuildKeyMap_k.Args.key_counts, GPU.counts_data_ptr)
                .set_arg(BuildKeyMap_k.Args.x_subdivisions, uniform_grid.x_subdivisions)
                .set_arg(BuildKeyMap_k.Args.key_count_length, uniform_grid.directory_length);

            count_candidates_k
                .ptr_arg(CountCandidates_k.Args.key_counts, GPU.counts_data_ptr)
                .set_arg(CountCandidates_k.Args.x_subdivisions, uniform_grid.x_subdivisions)
                .set_arg(CountCandidates_k.Args.key_count_length, uniform_grid.directory_length);

            aabb_collide_k
                .ptr_arg(AABBCollide_k.Args.key_counts, GPU.counts_data_ptr)
                .ptr_arg(AABBCollide_k.Args.key_offsets, GPU.offsets_data_ptr)
                .ptr_arg(AABBCollide_k.Args.counter, GPU.atomic_counter_ptr)
                .set_arg(AABBCollide_k.Args.x_subdivisions, uniform_grid.x_subdivisions)
                .set_arg(AABBCollide_k.Args.key_count_length, uniform_grid.directory_length);
        }

        simulate(dt);
    }
}
