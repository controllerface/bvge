package com.controllerface.bvge.ecs.systems;

import com.controllerface.bvge.cl.GPGPU;
import com.controllerface.bvge.ecs.ECS;
import com.controllerface.bvge.game.Sector;
import com.controllerface.bvge.physics.PhysicsEntityBatch;
import com.github.benmanes.caffeine.cache.Cache;

import java.util.Arrays;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;

public class SectorUnloader extends GameSystem
{
    private final Cache<Sector, PhysicsEntityBatch> sector_cache;
    private final Thread unloader;
    private final BlockingQueue<Float> next_dt = new ArrayBlockingQueue<>(1);
    private final UnloadedSector unloaded_sector = new UnloadedSector();

    public SectorUnloader(ECS ecs, Cache<Sector, PhysicsEntityBatch> sector_cache)
    {
        super(ecs);
        this.sector_cache = sector_cache;
        this.unloader = Thread.ofVirtual().start(() ->
        {
            while (!Thread.currentThread().isInterrupted())
            {
                try
                {
                    var dt = next_dt.take();
                    if (dt == -1f) GPGPU.core_memory.await_sector();
                    else
                    {
                        int[] last_counts = GPGPU.core_memory.last_egress_counts();
                        if (last_counts[0] > 0)
                        {
                            unloaded_sector.ensure_space(last_counts);
                            GPGPU.core_memory.transfer_world_output(unloaded_sector, last_counts);
                        }
                        GPGPU.core_memory.await_sector();
                    }
                }
                catch (InterruptedException e)
                {
                    Thread.currentThread().interrupt();
                }
            }
        });
        next_dt.offer(-1f);
    }

    @Override
    public void tick(float dt)
    {
        next_dt.offer(dt);
    }

    @Override
    public void shutdown()
    {
        unloader.interrupt();
    }
}
