package com.controllerface.bvge.ecs.systems.sectors;

import com.controllerface.bvge.cl.GPGPU;
import com.controllerface.bvge.ecs.ECS;
import com.controllerface.bvge.ecs.systems.GameSystem;
import com.controllerface.bvge.geometry.*;
import com.controllerface.bvge.gl.renderers.UniformGridRenderer;
import com.controllerface.bvge.physics.PhysicsEntityBatch;
import com.github.benmanes.caffeine.cache.Cache;

import java.util.*;
import java.util.concurrent.*;

public class SectorUnloader extends GameSystem
{
    private final UnloadedSectorSlice sectors                     = new UnloadedSectorSlice();
    private final Map<Sector, PhysicsEntityBatch> running_batches = new HashMap<>();
    private final BlockingQueue<Float> next_dt                    = new ArrayBlockingQueue<>(1);
    private final Cache<Sector, PhysicsEntityBatch> sector_cache;
    private final Thread task_thread;

    public SectorUnloader(ECS ecs, Cache<Sector, PhysicsEntityBatch> sector_cache)
    {
        super(ecs);
        this.sector_cache = sector_cache;
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
                    unload_sectors(next_dt.take());
                }
                catch (InterruptedException e)
                {
                    Thread.currentThread().interrupt();
                }
            }
        }
    }

    private void unload_sectors(float dt)
    {
        int[] last_counts = GPGPU.core_memory.last_egress_counts();
        int entity_count = (dt == -1f)  ? 0 : last_counts[0];
        if (entity_count > 0)
        {
            sectors.ensure_space(last_counts);
            GPGPU.core_memory.transfer_world_output(sectors, last_counts);
            for (int entity_offset = 0; entity_offset < entity_count; entity_offset++)
            {
                int entity_4_x = entity_offset * 4;
                int entity_4_y = entity_4_x + 1;
                int entity_4_z = entity_4_x + 2;
                int entity_4_w = entity_4_x + 3;

                int entity_2_x = entity_offset * 2;
                int entity_2_y = entity_2_x + 1;

                var entity_x               = sectors.raw_entity[entity_4_x];
                var entity_y               = sectors.raw_entity[entity_4_y];
                var entity_z               = sectors.raw_entity[entity_4_z];
                var entity_w               = sectors.raw_entity[entity_4_w];
                var entity_anim_elapsed_x  = sectors.raw_entity_anim_elapsed[entity_2_x];
                var entity_anim_elapsed_y  = sectors.raw_entity_anim_elapsed[entity_2_y];
                var entity_motion_state_x  = sectors.raw_entity_motion_state[entity_2_x];
                var entity_motion_state_y  = sectors.raw_entity_motion_state[entity_2_y];
                var entity_anim_index_x    = sectors.raw_entity_anim_index[entity_2_x];
                var entity_anim_index_y    = sectors.raw_entity_anim_index[entity_2_y];
                var entity_model_id        = sectors.raw_entity_model_id[entity_offset];
                var entity_model_transform = sectors.raw_entity_model_transform[entity_offset];
                var entity_mass            = sectors.raw_entity_mass[entity_offset];
                var entity_root_hull       = sectors.raw_entity_root_hull[entity_offset];
                var entity_flag            = sectors.raw_entity_flag[entity_offset];
                var entity_bone_table_x    = sectors.raw_entity_bone_table[entity_2_x];
                var entity_bone_table_y    = sectors.raw_entity_bone_table[entity_2_y];
                var entity_hull_table_x    = sectors.raw_entity_hull_table[entity_2_x];
                var entity_hull_table_y    = sectors.raw_entity_hull_table[entity_2_y];

                var entity_bones = new UnloadedEntityBone[entity_bone_table_y - entity_bone_table_x + 1];
                var entity_hulls = new UnloadedHull[entity_hull_table_y - entity_hull_table_x + 1];

                int entity_bone_count = 0;
                for (int entity_bone_offset = entity_bone_table_x; entity_bone_offset <= entity_bone_table_y; entity_bone_offset++)
                {
                    int entity_bone_16_s0 = entity_bone_offset * 16;

                    float[] bone_data = new float[16];
                    bone_data[0]  = sectors.raw_entity_bone[entity_bone_16_s0];
                    bone_data[1]  = sectors.raw_entity_bone[entity_bone_16_s0 + 1];
                    bone_data[2]  = sectors.raw_entity_bone[entity_bone_16_s0 + 2];
                    bone_data[3]  = sectors.raw_entity_bone[entity_bone_16_s0 + 3];
                    bone_data[4]  = sectors.raw_entity_bone[entity_bone_16_s0 + 4];
                    bone_data[5]  = sectors.raw_entity_bone[entity_bone_16_s0 + 5];
                    bone_data[6]  = sectors.raw_entity_bone[entity_bone_16_s0 + 6];
                    bone_data[7]  = sectors.raw_entity_bone[entity_bone_16_s0 + 7];
                    bone_data[8]  = sectors.raw_entity_bone[entity_bone_16_s0 + 8];
                    bone_data[9]  = sectors.raw_entity_bone[entity_bone_16_s0 + 9];
                    bone_data[10] = sectors.raw_entity_bone[entity_bone_16_s0 + 10];
                    bone_data[11] = sectors.raw_entity_bone[entity_bone_16_s0 + 11];
                    bone_data[12] = sectors.raw_entity_bone[entity_bone_16_s0 + 12];
                    bone_data[13] = sectors.raw_entity_bone[entity_bone_16_s0 + 13];
                    bone_data[14] = sectors.raw_entity_bone[entity_bone_16_s0 + 14];
                    bone_data[15] = sectors.raw_entity_bone[entity_bone_16_s0 + 15];
                    int ref_id    = sectors.raw_entity_bone_reference_id[entity_bone_offset];
                    int parent_id = sectors.raw_entity_bone_parent_id[entity_bone_offset];

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

                    var hull_x             = sectors.raw_hull[hull_4_x];
                    var hull_y             = sectors.raw_hull[hull_4_y];
                    var hull_z             = sectors.raw_hull[hull_4_z];
                    var hull_w             = sectors.raw_hull[hull_4_w];
                    var hull_scale_x       = sectors.raw_hull_scale[hull_2_x];
                    var hull_scale_y       = sectors.raw_hull_scale[hull_2_y];
                    var hull_rotation_x    = sectors.raw_hull_rotation[hull_2_x];
                    var hull_rotation_y    = sectors.raw_hull_rotation[hull_2_y];
                    var hull_friction      = sectors.raw_hull_friction[hull_offset];
                    var hull_restitution   = sectors.raw_hull_restitution[hull_offset];
                    var hull_integrity     = sectors.raw_hull_integrity[hull_offset];
                    var hull_mesh_id       = sectors.raw_hull_mesh_id[hull_offset];
                    var hull_entity_id     = sectors.raw_hull_entity_id[hull_offset];
                    var hull_uv_offset     = sectors.raw_hull_uv_offset[hull_offset];
                    var hull_flags         = sectors.raw_hull_flag[hull_offset];
                    var hull_point_table_x = sectors.raw_hull_point_table[hull_2_x];
                    var hull_point_table_y = sectors.raw_hull_point_table[hull_2_y];
                    var hull_edge_table_x  = sectors.raw_hull_edge_table[hull_2_x];
                    var hull_edge_table_y  = sectors.raw_hull_edge_table[hull_2_y];
                    var hull_bone_table_x  = sectors.raw_hull_bone_table[hull_2_x];
                    var hull_bone_table_y  = sectors.raw_hull_bone_table[hull_2_y];

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

                        var point_x                = sectors.raw_point[point_4_x];
                        var point_y                = sectors.raw_point[point_4_y];
                        var point_z                = sectors.raw_point[point_4_z];
                        var point_w                = sectors.raw_point[point_4_w];
                        var point_bone_table_x     = sectors.raw_point_bone_table[point_4_x];
                        var point_bone_table_y     = sectors.raw_point_bone_table[point_4_y];
                        var point_bone_table_z     = sectors.raw_point_bone_table[point_4_z];
                        var point_bone_table_w     = sectors.raw_point_bone_table[point_4_w];
                        var point_vertex_reference = sectors.raw_point_vertex_reference[point_offset];
                        var point_hull_index       = sectors.raw_point_hull_index[point_offset];
                        var point_hit_count        = sectors.raw_point_hit_count[point_offset];
                        var point_flags            = sectors.raw_point_flag[point_offset];

                        hull_points[hull_point_count++] = new UnloadedPoint(point_x, point_y, point_z, point_w,
                            point_bone_table_x, point_bone_table_y, point_bone_table_z, point_bone_table_w,
                            point_vertex_reference, point_hull_index, point_hit_count, point_flags);
                    }

                    int hull_edge_count = 0;
                    for (int edge_offset = hull_edge_table_x; edge_offset <= hull_edge_table_y; edge_offset++)
                    {
                        int edge_2_x = edge_offset * 2;
                        int edge_2_y = edge_2_x + 1;

                        var edge_p1     = sectors.raw_edge[edge_2_x];
                        var edge_p2     = sectors.raw_edge[edge_2_y];
                        var edge_length = sectors.raw_edge_length[edge_offset];
                        var edge_flags  = sectors.raw_edge_flag[edge_offset];

                        edge_p1 = edge_p1 - hull_point_table_x;
                        edge_p2 = edge_p2 - hull_point_table_x;

                        hull_edges[hull_edge_count++] = new UnloadedEdge(edge_p1, edge_p2, edge_length, edge_flags);
                    }

                    int hull_bone_count = 0;
                    for (int hull_bone_offset = hull_bone_table_x; hull_bone_offset <= hull_bone_table_y; hull_bone_offset++)
                    {
                        int hull_bone_16_s0 = hull_bone_offset * 16;

                        float[] bone_data = new float[16];
                        bone_data[0]    = sectors.raw_hull_bone[hull_bone_16_s0];
                        bone_data[1]    = sectors.raw_hull_bone[hull_bone_16_s0 + 1];
                        bone_data[2]    = sectors.raw_hull_bone[hull_bone_16_s0 + 2];
                        bone_data[3]    = sectors.raw_hull_bone[hull_bone_16_s0 + 3];
                        bone_data[4]    = sectors.raw_hull_bone[hull_bone_16_s0 + 4];
                        bone_data[5]    = sectors.raw_hull_bone[hull_bone_16_s0 + 5];
                        bone_data[6]    = sectors.raw_hull_bone[hull_bone_16_s0 + 6];
                        bone_data[7]    = sectors.raw_hull_bone[hull_bone_16_s0 + 7];
                        bone_data[8]    = sectors.raw_hull_bone[hull_bone_16_s0 + 8];
                        bone_data[9]    = sectors.raw_hull_bone[hull_bone_16_s0 + 9];
                        bone_data[10]   = sectors.raw_hull_bone[hull_bone_16_s0 + 10];
                        bone_data[11]   = sectors.raw_hull_bone[hull_bone_16_s0 + 11];
                        bone_data[12]   = sectors.raw_hull_bone[hull_bone_16_s0 + 12];
                        bone_data[13]   = sectors.raw_hull_bone[hull_bone_16_s0 + 13];
                        bone_data[14]   = sectors.raw_hull_bone[hull_bone_16_s0 + 14];
                        bone_data[15]   = sectors.raw_hull_bone[hull_bone_16_s0 + 15];
                        int bind_id     = sectors.raw_hull_bone_bind_pose_id[hull_bone_offset];
                        int inv_bind_id = sectors.raw_hull_bone_inv_bind_pose_id[hull_bone_offset];

                        hull_bones[hull_bone_count++] = new UnloadedHullBone(bone_data, bind_id, inv_bind_id);
                    }

                    entity_hulls[entity_hull_count++] = new UnloadedHull(hull_x, hull_y, hull_z, hull_w,
                        hull_scale_x, hull_scale_y, hull_rotation_x, hull_rotation_y,
                        hull_friction, hull_restitution, hull_integrity,
                        hull_mesh_id, hull_entity_id, hull_uv_offset, hull_flags,
                        hull_points, hull_edges, hull_bones);
                }

                // todo: sector objects can probably be cached since they are immutable records
                var raw_sector = UniformGridRenderer.get_sector_for_point(entity_x, entity_y);
                var sec = new Sector(raw_sector[0], raw_sector[1]);
                var batch = running_batches.computeIfAbsent(sec, PhysicsEntityBatch::new);
                int adjusted_root_hull = entity_root_hull - entity_hull_table_x;

                var unloaded_entity = new UnloadedEntity(entity_x, entity_y, entity_z, entity_w,
                    entity_anim_elapsed_x, entity_anim_elapsed_y,
                    entity_motion_state_x, entity_motion_state_y,
                    entity_anim_index_x, entity_anim_index_y,
                    entity_model_id, entity_model_transform,
                    entity_mass, adjusted_root_hull, entity_flag,
                    entity_hulls, entity_bones);

                batch.new_entity(unloaded_entity);
            }
            for (var entry : running_batches.entrySet())
            {
                sector_cache.put(entry.getKey(), entry.getValue());
            }
            running_batches.clear();
        }
        GPGPU.core_memory.await_sector();
    }

    @Override
    public void tick(float dt)
    {
        boolean ok = next_dt.offer(dt);
        assert ok : "unable to cycle SectorLoader";
    }

    @Override
    public void shutdown()
    {
        task_thread.interrupt();
    }
}
