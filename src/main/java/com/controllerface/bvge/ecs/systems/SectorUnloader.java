package com.controllerface.bvge.ecs.systems;

import com.controllerface.bvge.cl.GPGPU;
import com.controllerface.bvge.ecs.ECS;
import com.controllerface.bvge.game.Sector;
import com.controllerface.bvge.gl.renderers.UniformGridRenderer;
import com.controllerface.bvge.physics.PhysicsEntityBatch;
import com.github.benmanes.caffeine.cache.Cache;

import java.util.*;
import java.util.concurrent.*;

public class SectorUnloader extends GameSystem
{
    private final Cache<Sector, PhysicsEntityBatch> sector_cache;
//    private final Thread unloader;
    private final BlockingQueue<Float> next_dt = new ArrayBlockingQueue<>(1);
    private final UnloadedSectorSlice unloaded_sectors = new UnloadedSectorSlice();
    private final Map<Sector, PhysicsEntityBatch> running_batches = Collections.synchronizedMap(new HashMap<>());

    private Phaser entity_barrier;


    public SectorUnloader(ECS ecs, Cache<Sector, PhysicsEntityBatch> sector_cache)
    {
        super(ecs);
        this.sector_cache = sector_cache;
//        this.unloader = Thread.ofVirtual().start(() ->
//        {
//            while (!Thread.currentThread().isInterrupted())
//            {
//                try
//                {
//                    var dt = next_dt.take();
//                    if (dt == -1f) GPGPU.core_memory.await_sector();
//                    else
//                    {
//                        int[] last_counts = GPGPU.core_memory.last_egress_counts();
//                        int entity_count = last_counts[0];
//                        if (entity_count > 0)
//                        {
//                            unloaded_sectors.ensure_space(last_counts);
//                            GPGPU.core_memory.transfer_world_output(unloaded_sectors, last_counts);
//                            for (int i = 0; i < entity_count - 1; i++)
//                            {
//                                int e_offset = i * 4;
//                                var x = unloaded_sectors.raw_entity[e_offset];
//                                var y = unloaded_sectors.raw_entity[e_offset + 1];
//                                var z = unloaded_sectors.raw_entity[e_offset + 2];
//                                var w = unloaded_sectors.raw_entity[e_offset + 3];
//                                //System.out.println("dump: " + x + " , " + y + " , " + z + " , " + w);
//                            }
//                        }
//                        //GPGPU.core_memory.await_sector();
//                    }
//                }
//                catch (InterruptedException e)
//                {
//                    Thread.currentThread().interrupt();
//                }
//            }
//        });
       // next_dt.offer(-1f);
    }

    @Override
    public void tick(float dt)
    {
        //next_dt.offer(dt);

        int[] last_counts = GPGPU.core_memory.last_egress_counts();
        int entity_count = last_counts[0];
        if (entity_count > 0)
        {
            unloaded_sectors.ensure_space(last_counts);
            GPGPU.core_memory.transfer_world_output(unloaded_sectors, last_counts);
            for (int i = 0; i < entity_count - 1; i++)
            {
                int e_offset = i * 4;
                var x = unloaded_sectors.raw_entity[e_offset];
                var y = unloaded_sectors.raw_entity[e_offset + 1];
                var z = unloaded_sectors.raw_entity[e_offset + 2];
                var w = unloaded_sectors.raw_entity[e_offset + 3];
                //System.out.println("dump: " + x + " , " + y + " , " + z + " , " + w);
            }
        }
    }

    @Override
    public void shutdown()
    {
        //unloader.interrupt();
    }
}
