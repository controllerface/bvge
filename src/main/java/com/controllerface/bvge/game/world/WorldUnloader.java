package com.controllerface.bvge.game.world;

import com.controllerface.bvge.cl.GPGPU;
import com.controllerface.bvge.ecs.ECS;
import com.controllerface.bvge.ecs.systems.GameSystem;
import com.controllerface.bvge.game.world.sectors.BrokenObjectBuffer;
import com.controllerface.bvge.game.world.sectors.Sector;
import com.controllerface.bvge.game.world.sectors.UnorderedCoreBufferGroup;
import com.controllerface.bvge.geometry.*;
import com.controllerface.bvge.physics.PhysicsEntityBatch;
import com.controllerface.bvge.physics.UniformGrid;
import com.controllerface.bvge.substances.Solid;
import com.controllerface.bvge.substances.SubstanceTools;
import com.controllerface.bvge.util.Constants;
import com.github.benmanes.caffeine.cache.Cache;

import java.util.*;
import java.util.concurrent.*;

public class WorldUnloader extends GameSystem
{
    private final UnorderedCoreBufferGroup.Raw raw_sectors        = new UnorderedCoreBufferGroup.Raw();
    private final BrokenObjectBuffer.Raw raw_broken               = new BrokenObjectBuffer.Raw();
    private final Map<Sector, PhysicsEntityBatch> running_batches = new HashMap<>();
    private final BlockingQueue<Float> next_dt                    = new ArrayBlockingQueue<>(1);
    private final Cache<Sector, PhysicsEntityBatch> sector_cache;
    private final Queue<PhysicsEntityBatch> load_queue;
    private final Queue<Sector> unload_queue;
    private final Thread task_thread;
    private final Semaphore world_permit;

    public WorldUnloader(ECS ecs,
                         Cache<Sector, PhysicsEntityBatch> sector_cache,
                         Queue<PhysicsEntityBatch> load_queue,
                         Queue<Sector> unload_queue,
                         Semaphore world_permit)
    {
        super(ecs);
        this.sector_cache = sector_cache;
        this.load_queue = load_queue;
        this.unload_queue = unload_queue;
        this.world_permit = world_permit;
        this.task_thread = Thread.ofVirtual().start(new SectorUnloadTask());
        boolean ok = this.next_dt.offer(-1f);
        assert ok : "unable to start SectorLoader";
    }

    private class SectorUnloadTask implements Runnable
    {
        @Override
        public void run()
        {
            while (!Thread.currentThread().isInterrupted())
            {
                try
                {
                    float dt = next_dt.take();
                    world_permit.acquire();
                    if ((dt != -1f))
                    {
                        int[] last_counts = GPGPU.core_memory.last_egress_counts();
                        unload_sectors(last_counts);
                        unload_broken(last_counts);
                    }
                    GPGPU.core_memory.await_world_barrier();
                }
                catch (InterruptedException e)
                {
                    Thread.currentThread().interrupt();
                }
            }
        }
    }

    private void unload_sectors(int[] last_counts)
    {
        int entity_count = last_counts[0];
        if (entity_count > 0)
        {
            GPGPU.core_memory.unload_sectors(raw_sectors, last_counts);
            for (int entity_offset = 0; entity_offset < entity_count; entity_offset++)
            {
                int entity_4_x = entity_offset * 4;
                int entity_4_y = entity_4_x + 1;
                int entity_4_z = entity_4_x + 2;
                int entity_4_w = entity_4_x + 3;
                int entity_2_x = entity_offset * 2;
                int entity_2_y = entity_2_x + 1;

                var entity_x               = raw_sectors.entity[entity_4_x];
                var entity_y               = raw_sectors.entity[entity_4_y];
                var entity_z               = raw_sectors.entity[entity_4_z];
                var entity_w               = raw_sectors.entity[entity_4_w];
                var entity_anim_time_x     = raw_sectors.entity_anim_time[entity_2_x];
                var entity_anim_time_y     = raw_sectors.entity_anim_time[entity_2_y];
                var entity_prev_time_x     = raw_sectors.entity_prev_time[entity_2_x];
                var entity_prev_time_y     = raw_sectors.entity_prev_time[entity_2_y];
                var entity_motion_state_x  = raw_sectors.entity_motion_state[entity_2_x];
                var entity_motion_state_y  = raw_sectors.entity_motion_state[entity_2_y];
                var entity_anim_layer_x    = raw_sectors.entity_anim_layers[entity_2_x];
                var entity_anim_layer_y    = raw_sectors.entity_anim_layers[entity_2_y];
                var entity_anim_prev_x     = raw_sectors.entity_anim_previous[entity_2_x];
                var entity_anim_prev_y     = raw_sectors.entity_anim_previous[entity_2_y];
                var entity_model_id        = raw_sectors.entity_model_id[entity_offset];
                var entity_model_transform = raw_sectors.entity_model_transform[entity_offset];
                var entity_mass            = raw_sectors.entity_mass[entity_offset];
                var entity_root_hull       = raw_sectors.entity_root_hull[entity_offset];
                var entity_type            = raw_sectors.entity_type[entity_offset];
                var entity_flag            = raw_sectors.entity_flag[entity_offset];
                var entity_bone_table_x    = raw_sectors.entity_bone_table[entity_2_x];
                var entity_bone_table_y    = raw_sectors.entity_bone_table[entity_2_y];
                var entity_hull_table_x    = raw_sectors.entity_hull_table[entity_2_x];
                var entity_hull_table_y    = raw_sectors.entity_hull_table[entity_2_y];

                int entity_bone_table_length = entity_bone_table_y - entity_bone_table_x + 1;
                var entity_bones = new UnloadedEntityBone[entity_bone_table_length];
                var entity_hulls = new UnloadedHull[entity_hull_table_y - entity_hull_table_x + 1];

                if (entity_bone_table_length > 0 && raw_sectors.entity_bone.length == 0)
                {
                    System.out.println("unexpected bone table size: " + entity_bone_table_length);
                    throw new RuntimeException("unexpected bone table size: " + entity_bone_table_length);
                }

                int entity_bone_count = 0;
                for (int entity_bone_offset = entity_bone_table_x; entity_bone_offset <= entity_bone_table_y; entity_bone_offset++)
                {
                    int entity_bone_16_s0 = entity_bone_offset * 16;

                    float[] bone_data = new float[16];
                    bone_data[0]  = raw_sectors.entity_bone[entity_bone_16_s0];
                    bone_data[1]  = raw_sectors.entity_bone[entity_bone_16_s0 + 1];
                    bone_data[2]  = raw_sectors.entity_bone[entity_bone_16_s0 + 2];
                    bone_data[3]  = raw_sectors.entity_bone[entity_bone_16_s0 + 3];
                    bone_data[4]  = raw_sectors.entity_bone[entity_bone_16_s0 + 4];
                    bone_data[5]  = raw_sectors.entity_bone[entity_bone_16_s0 + 5];
                    bone_data[6]  = raw_sectors.entity_bone[entity_bone_16_s0 + 6];
                    bone_data[7]  = raw_sectors.entity_bone[entity_bone_16_s0 + 7];
                    bone_data[8]  = raw_sectors.entity_bone[entity_bone_16_s0 + 8];
                    bone_data[9]  = raw_sectors.entity_bone[entity_bone_16_s0 + 9];
                    bone_data[10] = raw_sectors.entity_bone[entity_bone_16_s0 + 10];
                    bone_data[11] = raw_sectors.entity_bone[entity_bone_16_s0 + 11];
                    bone_data[12] = raw_sectors.entity_bone[entity_bone_16_s0 + 12];
                    bone_data[13] = raw_sectors.entity_bone[entity_bone_16_s0 + 13];
                    bone_data[14] = raw_sectors.entity_bone[entity_bone_16_s0 + 14];
                    bone_data[15] = raw_sectors.entity_bone[entity_bone_16_s0 + 15];
                    int ref_id    = raw_sectors.entity_bone_reference_id[entity_bone_offset];
                    int parent_id = raw_sectors.entity_bone_parent_id[entity_bone_offset];

                    entity_bones[entity_bone_count++] = new UnloadedEntityBone(bone_data, ref_id, parent_id);
                }

                int entity_hull_count = 0;
                for (int hull_offset = entity_hull_table_x; hull_offset <= entity_hull_table_y; hull_offset++)
                {
                    int hull_4_x = hull_offset * 4;
                    int hull_4_y = hull_4_x + 1;
                    int hull_4_z = hull_4_x + 2;
                    int hull_4_w = hull_4_x + 3;
                    int hull_2_x = hull_offset * 2;
                    int hull_2_y = hull_2_x + 1;

                    var hull_x             = raw_sectors.hull[hull_4_x];
                    var hull_y             = raw_sectors.hull[hull_4_y];
                    var hull_z             = raw_sectors.hull[hull_4_z];
                    var hull_w             = raw_sectors.hull[hull_4_w];
                    var hull_scale_x       = raw_sectors.hull_scale[hull_2_x];
                    var hull_scale_y       = raw_sectors.hull_scale[hull_2_y];
                    var hull_rotation_x    = raw_sectors.hull_rotation[hull_2_x];
                    var hull_rotation_y    = raw_sectors.hull_rotation[hull_2_y];
                    var hull_friction      = raw_sectors.hull_friction[hull_offset];
                    var hull_restitution   = raw_sectors.hull_restitution[hull_offset];
                    var hull_integrity     = raw_sectors.hull_integrity[hull_offset];
                    var hull_mesh_id       = raw_sectors.hull_mesh_id[hull_offset];
                    var hull_entity_id     = raw_sectors.hull_entity_id[hull_offset];
                    var hull_uv_offset     = raw_sectors.hull_uv_offset[hull_offset];
                    var hull_flags         = raw_sectors.hull_flag[hull_offset];
                    var hull_point_table_x = raw_sectors.hull_point_table[hull_2_x];
                    var hull_point_table_y = raw_sectors.hull_point_table[hull_2_y];
                    var hull_edge_table_x  = raw_sectors.hull_edge_table[hull_2_x];
                    var hull_edge_table_y  = raw_sectors.hull_edge_table[hull_2_y];
                    var hull_bone_table_x  = raw_sectors.hull_bone_table[hull_2_x];
                    var hull_bone_table_y  = raw_sectors.hull_bone_table[hull_2_y];

                    var hull_points = new UnloadedPoint[hull_point_table_y - hull_point_table_x + 1];
                    var hull_edges = new UnloadedEdge[hull_edge_table_y - hull_edge_table_x + 1];
                    var hull_bones = new UnloadedHullBone[hull_bone_table_y - hull_bone_table_x + 1];

                    int hull_point_count = 0;
                    for (int point_offset = hull_point_table_x; point_offset <= hull_point_table_y; point_offset++)
                    {
                        int point_4_x = point_offset * 4;
                        int point_4_y = point_4_x + 1;
                        int point_4_z = point_4_x + 2;
                        int point_4_w = point_4_x + 3;

                        var point_x                = raw_sectors.point[point_4_x];
                        var point_y                = raw_sectors.point[point_4_y];
                        var point_z                = raw_sectors.point[point_4_z];
                        var point_w                = raw_sectors.point[point_4_w];
                        var point_bone_table_x     = raw_sectors.point_bone_table[point_4_x];
                        var point_bone_table_y     = raw_sectors.point_bone_table[point_4_y];
                        var point_bone_table_z     = raw_sectors.point_bone_table[point_4_z];
                        var point_bone_table_w     = raw_sectors.point_bone_table[point_4_w];
                        var point_vertex_reference = raw_sectors.point_vertex_reference[point_offset];
                        var point_hull_index       = raw_sectors.point_hull_index[point_offset];
                        var point_hit_count        = raw_sectors.point_hit_count[point_offset];
                        var point_flags            = raw_sectors.point_flag[point_offset];

                        hull_points[hull_point_count++] = new UnloadedPoint(point_x, point_y, point_z, point_w,
                            point_bone_table_x, point_bone_table_y, point_bone_table_z, point_bone_table_w,
                            point_vertex_reference, point_hull_index, point_hit_count, point_flags);
                    }

                    int hull_edge_count = 0;
                    for (int edge_offset = hull_edge_table_x; edge_offset <= hull_edge_table_y; edge_offset++)
                    {
                        int edge_2_x = edge_offset * 2;
                        int edge_2_y = edge_2_x + 1;

                        var edge_p1     = raw_sectors.edge[edge_2_x];
                        var edge_p2     = raw_sectors.edge[edge_2_y];
                        var edge_length = raw_sectors.edge_length[edge_offset];
                        var edge_flags  = raw_sectors.edge_flag[edge_offset];

                        edge_p1 = edge_p1 - hull_point_table_x;
                        edge_p2 = edge_p2 - hull_point_table_x;

                        hull_edges[hull_edge_count++] = new UnloadedEdge(edge_p1, edge_p2, edge_length, edge_flags);
                    }

                    int hull_bone_count = 0;
                    for (int hull_bone_offset = hull_bone_table_x; hull_bone_offset <= hull_bone_table_y; hull_bone_offset++)
                    {
                        int hull_bone_16_s0 = hull_bone_offset * 16;

                        float[] bone_data = new float[16];
                        bone_data[0]    = raw_sectors.hull_bone[hull_bone_16_s0];
                        bone_data[1]    = raw_sectors.hull_bone[hull_bone_16_s0 + 1];
                        bone_data[2]    = raw_sectors.hull_bone[hull_bone_16_s0 + 2];
                        bone_data[3]    = raw_sectors.hull_bone[hull_bone_16_s0 + 3];
                        bone_data[4]    = raw_sectors.hull_bone[hull_bone_16_s0 + 4];
                        bone_data[5]    = raw_sectors.hull_bone[hull_bone_16_s0 + 5];
                        bone_data[6]    = raw_sectors.hull_bone[hull_bone_16_s0 + 6];
                        bone_data[7]    = raw_sectors.hull_bone[hull_bone_16_s0 + 7];
                        bone_data[8]    = raw_sectors.hull_bone[hull_bone_16_s0 + 8];
                        bone_data[9]    = raw_sectors.hull_bone[hull_bone_16_s0 + 9];
                        bone_data[10]   = raw_sectors.hull_bone[hull_bone_16_s0 + 10];
                        bone_data[11]   = raw_sectors.hull_bone[hull_bone_16_s0 + 11];
                        bone_data[12]   = raw_sectors.hull_bone[hull_bone_16_s0 + 12];
                        bone_data[13]   = raw_sectors.hull_bone[hull_bone_16_s0 + 13];
                        bone_data[14]   = raw_sectors.hull_bone[hull_bone_16_s0 + 14];
                        bone_data[15]   = raw_sectors.hull_bone[hull_bone_16_s0 + 15];
                        int bind_id     = raw_sectors.hull_bone_bind_pose_id[hull_bone_offset];
                        int inv_bind_id = raw_sectors.hull_bone_inv_bind_pose_id[hull_bone_offset];

                        hull_bones[hull_bone_count++] = new UnloadedHullBone(bone_data, bind_id, inv_bind_id);
                    }

                    entity_hulls[entity_hull_count++] = new UnloadedHull(hull_x, hull_y, hull_z, hull_w,
                        hull_scale_x, hull_scale_y, hull_rotation_x, hull_rotation_y,
                        hull_friction, hull_restitution, hull_integrity,
                        hull_mesh_id, hull_entity_id, hull_uv_offset, hull_flags,
                        hull_points, hull_edges, hull_bones);
                }

                // todo: sector objects can probably be cached since they are immutable records
                var raw_sector = UniformGrid.get_sector_for_point(entity_x, entity_y);
                var sec = new Sector(raw_sector[0], raw_sector[1]);
                var batch = running_batches.get(sec);
                Objects.requireNonNull(batch);
                int adjusted_root_hull = entity_root_hull - entity_hull_table_x;

                var unloaded_entity = new UnloadedEntity(entity_x, entity_y, entity_z, entity_w,
                    entity_anim_time_x, entity_anim_time_y,
                    entity_prev_time_x, entity_prev_time_y,
                    entity_motion_state_x, entity_motion_state_y,
                    entity_anim_layer_x, entity_anim_layer_y,
                    entity_anim_prev_x, entity_anim_prev_y,
                    entity_model_id, entity_model_transform,
                    entity_mass, adjusted_root_hull,
                    entity_type, entity_flag,
                    entity_hulls, entity_bones);

                batch.new_entity(unloaded_entity);
            }
            for (var entry : running_batches.entrySet())
            {
                sector_cache.put(entry.getKey(), entry.getValue());
            }
            running_batches.clear();
        }
    }

    private void unload_broken(int[] last_counts)
    {
        int broken_count = last_counts[6];
        if (broken_count > 0)
        {
            var batch = new PhysicsEntityBatch();
            GPGPU.core_memory.unload_broken(raw_broken, broken_count);
            int offset_2 = 0;
            int offset_1 = 0;
            for (int type : raw_broken.entity_types)
            {
                if (type == -1) break;

                float x = raw_broken.positions[offset_2++];
                float y = raw_broken.positions[offset_2++];
                float m = raw_broken.model_ids[offset_1++];

                var substance = SubstanceTools.from_type_index(type);
                
                if (substance instanceof Solid solid)
                {
                    float sz = UniformGrid.BLOCK_SIZE / 2;
                    float offset = (sz / 2) - 2f;

                    // todo: friction and restitution values from broken object should be forwarded to the spawned object
                    if (m == ModelRegistry.BASE_BLOCK_INDEX)
                    {
                        batch.new_block(x - offset, y - offset, sz, 90, 0, 0, Constants.EntityFlags.COLLECTABLE.bits, 0, solid, new int[4]);
                        batch.new_block(x - offset, y + offset, sz, 90, 0, 0, Constants.EntityFlags.COLLECTABLE.bits, 0, solid, new int[4]);
                        batch.new_block(x + offset, y - offset, sz, 90, 0, 0, Constants.EntityFlags.COLLECTABLE.bits, 0, solid, new int[4]);
                        //batch.new_block(true, x + offset, y + offset, sz, 90, 0,0, 0, solid, new int[4]);
                    }
                    else if (m == ModelRegistry.L_SHARD_INDEX)
                    {
                        batch.new_shard(false, false, x - offset, y + offset, sz, Constants.EntityFlags.COLLECTABLE.bits, 0, 30, 0, 0, solid);
                        batch.new_shard(false, true, x + offset, y + offset, sz, Constants.EntityFlags.COLLECTABLE.bits, 0, 30, 0, 0, solid);
                    }
                    else if (m == ModelRegistry.R_SHARD_INDEX)
                    {
                        batch.new_shard(false, true, x - offset, y + offset, sz, Constants.EntityFlags.COLLECTABLE.bits, 0, 30, 0, 0, solid);
                        batch.new_shard(false, false, x + offset, y + offset, sz, Constants.EntityFlags.COLLECTABLE.bits, 0, 30, 0, 0, solid);
                    }
                    else if (m == ModelRegistry.BASE_SPIKE_INDEX)
                    {
                        batch.new_shard(true, false, x - offset, y + offset, sz, Constants.EntityFlags.COLLECTABLE.bits, 0, 30, 0, 0, solid);
                        batch.new_shard(true, false, x + offset, y - offset, sz, Constants.EntityFlags.COLLECTABLE.bits, 0, 30, 0, 0, solid);
                        batch.new_shard(true, false, x + offset, y + offset, sz, Constants.EntityFlags.COLLECTABLE.bits, 0, 30, 0, 0, solid);
                    }
                }
            }

            load_queue.offer(batch);

            // this is required to ensure broken objects from previous frames aren't processed
            Arrays.fill(raw_broken.entity_types, -1);
        }
    }

    @Override
    public void tick(float dt)
    {
        Sector unloading;
        while ((unloading = unload_queue.poll()) != null)
        {
            running_batches.put(unloading, new PhysicsEntityBatch());
        }

        boolean ok = next_dt.offer(dt);
        assert ok : "unable to cycle SectorLoader";
    }

    @Override
    public void shutdown()
    {
        task_thread.interrupt();
    }
}
