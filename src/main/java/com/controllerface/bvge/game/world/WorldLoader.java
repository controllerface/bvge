package com.controllerface.bvge.game.world;

import com.controllerface.bvge.cl.GPGPU;
import com.controllerface.bvge.ecs.ECS;
import com.controllerface.bvge.ecs.systems.GameSystem;
import com.controllerface.bvge.game.world.sectors.Sector;
import com.controllerface.bvge.editor.Editor;
import com.controllerface.bvge.physics.PhysicsEntityBatch;
import com.controllerface.bvge.physics.UniformGrid;
import com.github.benmanes.caffeine.cache.Cache;

import java.util.HashSet;
import java.util.Queue;
import java.util.Set;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.Semaphore;

public class WorldLoader extends GameSystem
{
    private final WorldType world                = new EarthLikeWorld();
    private final Set<Sector> old_loaded_sectors = new HashSet<>();
    private final Set<Sector> new_loaded_sectors = new HashSet<>();
    private final UniformGrid uniformGrid;
    private final Cache<Sector, PhysicsEntityBatch> sector_cache;
    private final Queue<PhysicsEntityBatch> load_queue;
    private final Queue<Sector> unload_queue;
    private final Thread task_thread;
    private final Semaphore world_permit;

    private final BlockingQueue<SectorBounds> next_sector_bounds = new ArrayBlockingQueue<>(1);

    private record SectorBounds(float outer_x_origin, float outer_y_origin, float outer_x_corner, float outer_y_corner) { }

    public WorldLoader(ECS ecs,
                       UniformGrid uniformGrid,
                       Cache<Sector, PhysicsEntityBatch> sector_cache_in,
                       Queue<PhysicsEntityBatch> load_queue,
                       Queue<Sector> unload_queue,
                       Semaphore world_permit)
    {
        super(ecs);
        this.uniformGrid = uniformGrid;
        this.sector_cache = sector_cache_in;
        this.load_queue = load_queue;
        this.unload_queue = unload_queue;
        this.world_permit = world_permit;
        this.task_thread = Thread.ofVirtual().start(new SectorLoadTask());
    }

    private class SectorLoadTask implements Runnable
    {
        @Override
        public void run()
        {
            while (!Thread.currentThread().isInterrupted())
            {
                try
                {
                    load_sectors(next_sector_bounds.take());
                    world_permit.release(1);
                    GPGPU.core_memory.await_world_barrier();
                }
                catch (InterruptedException e)
                {
                    Thread.currentThread().interrupt();
                }
            }
        }
    }

    private void load_sectors(SectorBounds sector_bounds)
    {
        var sector_0_key = UniformGrid.get_sector_for_point(sector_bounds.outer_x_origin, sector_bounds.outer_y_origin);
        var sector_2_key = UniformGrid.get_sector_for_point(sector_bounds.outer_x_corner, sector_bounds.outer_y_corner);

        float sector_0_origin_x = (float) sector_0_key[0] * UniformGrid.SECTOR_SIZE;
        float sector_0_origin_y = (float) sector_0_key[1] * UniformGrid.SECTOR_SIZE;

        float sector_2_origin_x = (float) sector_2_key[0] * UniformGrid.SECTOR_SIZE;
        float sector_2_origin_y = (float) sector_2_key[1] * UniformGrid.SECTOR_SIZE;

        old_loaded_sectors.clear();
        old_loaded_sectors.addAll(new_loaded_sectors);
        new_loaded_sectors.clear();

        for (int sx = sector_0_key[0]; sx <= sector_2_key[0]; sx++)
        {
            for (int sy = sector_0_key[1]; sy <= sector_2_key[1]; sy++)
            {
                var sector = new Sector(sx, sy);
                if (old_loaded_sectors.contains(sector))
                {
                    new_loaded_sectors.add(sector);
                    var _ = sector_cache.getIfPresent(sector); // keep sector live in cache by accessing it
                }
                else
                {
                    new_loaded_sectors.add(sector);
                    var sector_batch = sector_cache.get(sector, world::generate_sector);
                    GPGPU.core_memory.load_entity_batch(sector_batch);
                }
            }
        }

        for (var sector : old_loaded_sectors)
        {
            if (!new_loaded_sectors.contains(sector))
            {
                unload_queue.add(sector);
            }
        }

        uniformGrid.update_sector_metrics(new_loaded_sectors, sector_0_origin_x, sector_0_origin_y,
            Math.abs(sector_0_origin_x - (sector_2_origin_x + UniformGrid.SECTOR_SIZE)),
            Math.abs(sector_0_origin_y - (sector_2_origin_y + UniformGrid.SECTOR_SIZE)));

        PhysicsEntityBatch batch;
        while ((batch = load_queue.poll()) != null)
        {
            GPGPU.core_memory.load_entity_batch(batch);
        }
    }

    @Override
    public void tick(float dt)
    {
        float outer_x_origin = uniformGrid.outer_x_origin();
        float outer_y_origin = uniformGrid.outer_y_origin();
        float outer_x_corner = outer_x_origin + uniformGrid.outer_width;
        float outer_y_corner = outer_y_origin + uniformGrid.outer_height;
        try
        {
            next_sector_bounds.put(new SectorBounds(outer_x_origin, outer_y_origin, outer_x_corner, outer_y_corner));
        }
        catch (InterruptedException e)
        {
            throw new RuntimeException(e);
        }
        if (Editor.ACTIVE)
        {
            Editor.queue_event("sector_count", String.valueOf(new_loaded_sectors.size()));
        }
    }

    @Override
    public void shutdown()
    {
        task_thread.interrupt();
    }
}
