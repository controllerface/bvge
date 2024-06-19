package com.controllerface.bvge.ecs.systems;

import com.controllerface.bvge.cl.GPGPU;
import com.controllerface.bvge.ecs.ECS;
import com.controllerface.bvge.game.world.sectors.CollectedObjectBuffer;
import com.controllerface.bvge.game.state.PlayerInventory;

import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;

public class InventorySystem extends GameSystem
{
    private final CollectedObjectBuffer.Raw raw_collected = new CollectedObjectBuffer.Raw();
    private final BlockingQueue<Float> next_dt = new ArrayBlockingQueue<>(1);
    private final PlayerInventory player_inventory;
    private final Thread task_thread;

    public InventorySystem(ECS ecs, PlayerInventory player_inventory)
    {
        super(ecs);
        this.player_inventory = player_inventory;
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
                    if ((dt != -1f))
                    {
                        int[] last_counts = GPGPU.core_memory.last_egress_counts();
                        unload_collected(last_counts);
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

    private void unload_collected(int[] last_counts)
    {
        int collected_count = last_counts[7];
        if (collected_count > 0)
        {
            GPGPU.core_memory.unload_collected(raw_collected, collected_count);
            for (int i = 0; i < collected_count; i++)
            {
                player_inventory.collect_substance(raw_collected.types[i], 1);
            }
        }
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