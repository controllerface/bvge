package com.controllerface.bvge.ecs.systems;

import com.controllerface.bvge.cl.GPGPU;
import com.controllerface.bvge.ecs.ECS;
import com.controllerface.bvge.game.Sector;
import com.controllerface.bvge.gl.renderers.UniformGridRenderer;
import com.controllerface.bvge.physics.PhysicsEntityBatch;
import com.controllerface.bvge.physics.UniformGrid;
import com.github.benmanes.caffeine.cache.Cache;

import java.util.HashSet;
import java.util.Set;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;

public class SectorLoader extends GameSystem
{
    private final UniformGrid uniformGrid;
    private final Set<Sector> last_loaded_sectors = new HashSet<>();
    private final Set<Sector> loaded_sectors = new HashSet<>();
    private final Set<Sector> deferred_sectors = new HashSet<>();
    private final Cache<Sector, PhysicsEntityBatch> sector_cache;
    private final Thread loader;
    private final BlockingQueue<SectorBounds> next_sector_bounds = new ArrayBlockingQueue<>(1);
    private final WorldType world = new EarthLikeWorld();

    private record SectorBounds(float outer_x_origin, float outer_y_origin, float outer_x_corner, float outer_y_corner) { }

    public SectorLoader(ECS ecs, UniformGrid uniformGrid, Cache<Sector, PhysicsEntityBatch> sector_cache)
    {
        super(ecs);
        this.uniformGrid = uniformGrid;
        this.sector_cache = sector_cache;

        this.loader = Thread.ofVirtual().start(() ->
        {
            while (!Thread.currentThread().isInterrupted())
            {
                try
                {
                    var sector_bounds = next_sector_bounds.take();
                    var sector_0_key = UniformGridRenderer.get_sector_for_point(sector_bounds.outer_x_origin, sector_bounds.outer_y_origin);
                    var sector_2_key = UniformGridRenderer.get_sector_for_point(sector_bounds.outer_x_corner, sector_bounds.outer_y_corner);

                    float sector_0_origin_x = (float) sector_0_key[0] * UniformGrid.SECTOR_SIZE;
                    float sector_0_origin_y = (float) sector_0_key[1] * UniformGrid.SECTOR_SIZE;

                    float sector_2_origin_x = (float) sector_2_key[0] * UniformGrid.SECTOR_SIZE;
                    float sector_2_origin_y = (float) sector_2_key[1] * UniformGrid.SECTOR_SIZE;

                    last_loaded_sectors.clear();
                    last_loaded_sectors.addAll(loaded_sectors);
                    loaded_sectors.clear();

                    // This "slots" count is used to control how many sectors are loaded each tick. Generally,
                    // it should be set to the number of rows in the sector grid, with processing being done
                    // in column-major order. I.e. only a single column of sectors loads each frame
                    //int slots = sector_2_key[1] - sector_0_key[1];

                    for (int sx = sector_0_key[0]; sx <= sector_2_key[0]; sx++)
                    {
                        for (int sy = sector_0_key[1]; sy <= sector_2_key[1]; sy++)
                        {
                            var sector = new Sector(sx, sy);
                            if (last_loaded_sectors.contains(sector))
                            {
                                loaded_sectors.add(sector);
                                var _ = sector_cache.getIfPresent(sector);
                            }
                            else //if (slots-- > 0)
                            {
                                loaded_sectors.add(sector);
                                var sector_batch = sector_cache.get(sector, world::load_sector);
                                GPGPU.core_memory.load_entity_batch(sector_batch);
                            }
//                            else
//                            {
//                                deferred_sectors.add(sector);
//                            }
                        }
                    }
                    last_loaded_sectors.forEach(s->
                    {
                        if (!loaded_sectors.contains(s))
                        {
                            sector_cache.put(s, new PhysicsEntityBatch(s));
                        }
                    });

                    uniformGrid.update_sector_metrics(loaded_sectors, sector_0_origin_x, sector_0_origin_y,
                        Math.abs(sector_0_origin_x - (sector_2_origin_x + UniformGrid.SECTOR_SIZE)),
                        Math.abs(sector_0_origin_y - (sector_2_origin_y + UniformGrid.SECTOR_SIZE)));

                    GPGPU.core_memory.await_sector();
                }
                catch (InterruptedException e)
                {
                    Thread.currentThread().interrupt();
                }
            }
        });
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
    }

    @Override
    public void shutdown()
    {
        loader.interrupt();
    }
}
