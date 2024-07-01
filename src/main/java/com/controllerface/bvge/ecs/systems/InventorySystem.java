package com.controllerface.bvge.ecs.systems;

import com.controllerface.bvge.cl.GPGPU;
import com.controllerface.bvge.ecs.ECS;
import com.controllerface.bvge.editor.Editor;
import com.controllerface.bvge.game.world.sectors.CollectedObjectBuffer;
import com.controllerface.bvge.game.state.PlayerInventory;
import com.controllerface.bvge.substances.Solid;
import com.controllerface.bvge.window.EventBus;
import com.controllerface.bvge.window.Window;

import java.util.*;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.logging.Level;
import java.util.logging.Logger;

public class InventorySystem extends GameSystem
{
    private final CollectedObjectBuffer.Raw raw_collected = new CollectedObjectBuffer.Raw();
    private final BlockingQueue<Float> next_dt = new ArrayBlockingQueue<>(1);
    private final PlayerInventory player_inventory;
    private final Thread task_thread;

    private final static Logger LOGGER = Logger.getLogger(InventorySystem.class.getName());
    private final Queue<EventBus.Event> event_queue = new ConcurrentLinkedQueue<>();

    private Solid current_block = null;

    public InventorySystem(ECS ecs, PlayerInventory player_inventory)
    {
        super(ecs);
        this.player_inventory = player_inventory;
        this.task_thread = Thread.ofVirtual().start(new SectorUnloadTask());
        boolean ok = this.next_dt.offer(-1f);
        assert ok : "unable to start SectorLoader";
        Window.get().event_bus().register(event_queue, EventBus.EventType.NEXT_ITEM, EventBus.EventType.PREV_ITEM);
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
                int qty = 1; // todo: pull from world, possibly account for player stat mods
                int type = raw_collected.types[i];
                LOGGER.log(Level.FINE, type + ":" + qty);
                if (Editor.ACTIVE)
                {
                    Editor.inventory(type, qty);
                }
                player_inventory.collect_substance(raw_collected.types[i], 1);
            }
            Window.get().event_bus().report_event(EventBus.inventory(EventBus.EventType.ITEM_CHANGE));
        }
    }


    private Solid findNextItem(Solid currentItem)
    {
        List<Solid> solids = new ArrayList<>(player_inventory.solid_counts().keySet());
        int startIndex = (currentItem == null)
            ? -1
            : solids.indexOf(currentItem);

        for (int i = startIndex + 1; i < solids.size(); i++)
        {
            Solid solid = solids.get(i);
            if (player_inventory.solid_counts().get(solid) > 0)
            {
                return solid;
            }
        }
        return null;
    }

    private Solid findPrevItem(Solid currentItem)
    {
        List<Solid> solids = new ArrayList<>(player_inventory.solid_counts().keySet());
        int startIndex = (currentItem == null)
            ? solids.size()
            : solids.indexOf(currentItem);

        for (int i = startIndex - 1; i >= 0; i--)
        {
            Solid solid = solids.get(i);
            if (player_inventory.solid_counts().get(solid) > 0)
            {
                return solid;
            }
        }
        return null;
    }

    @Override
    public void tick(float dt)
    {
        boolean ok = next_dt.offer(dt);
        assert ok : "unable to cycle SectorLoader";
        var last_block = current_block;
        EventBus.Event next_event;
        while ((next_event = event_queue.poll()) != null)
        {
            if (next_event.type() == EventBus.EventType.NEXT_ITEM)
            {
                current_block = findNextItem(current_block);
            }
            if (next_event.type() == EventBus.EventType.PREV_ITEM)
            {
                current_block = findPrevItem(current_block);
            }
        }
        if (last_block != current_block)
        {
            var name = current_block == null ? "-" : current_block.name();

            Window.get().event_bus()
                .report_event(EventBus.message(EventBus.EventType.ITEM_PLACING, name));
        }
    }

    @Override
    public void shutdown()
    {
        task_thread.interrupt();
    }
}
