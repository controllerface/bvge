package com.controllerface.bvge.ecs.systems;

import com.controllerface.bvge.cl.GPGPU;
import com.controllerface.bvge.ecs.ECS;
import com.controllerface.bvge.ecs.components.Component;
import com.controllerface.bvge.ecs.components.ControlPoints;
import com.controllerface.bvge.ecs.components.GameComponent;
import com.controllerface.bvge.editor.Editor;
import com.controllerface.bvge.game.world.sectors.CollectedObjectBuffer;
import com.controllerface.bvge.game.state.PlayerInventory;
import com.controllerface.bvge.window.EventType;
import com.controllerface.bvge.window.Window;

import java.util.Map;
import java.util.Objects;
import java.util.Queue;
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
    private final Queue<EventType> event_queue = new ConcurrentLinkedQueue<>();

    public InventorySystem(ECS ecs, PlayerInventory player_inventory)
    {
        super(ecs);
        this.player_inventory = player_inventory;
        this.task_thread = Thread.ofVirtual().start(new SectorUnloadTask());
        boolean ok = this.next_dt.offer(-1f);
        assert ok : "unable to start SectorLoader";
        Window.get().event_bus().register(event_queue, EventType.NEXT_ITEM, EventType.PREV_ITEM);
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
                if (Editor.ACTIVE) Editor.inventory(type, qty);
                player_inventory.collect_substance(raw_collected.types[i], 1);
            }
            Window.get().event_bus().report_event(EventType.INVENTORY);
        }
    }

    @Override
    public void tick(float dt)
    {
        boolean ok = next_dt.offer(dt);
        assert ok : "unable to cycle SectorLoader";
        EventType next_event;
        while ((next_event = event_queue.poll()) != null)
        {
            if (next_event == EventType.NEXT_ITEM
                || next_event == EventType.PREV_ITEM)
            {

                // if nothing is currently selected, start from the top/bottom of the
                // inventory and find the next solid that has enough material to spawn

                // if something is selected

                // if no materials are available, return.

                // for the selected object

                System.out.println("event: " + next_event);
            }
        }

    }

    @Override
    public void shutdown()
    {
        task_thread.interrupt();
    }
}
