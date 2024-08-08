package com.controllerface.bvge.memory;

import com.controllerface.bvge.gpu.cl.GPGPU;
import com.controllerface.bvge.gpu.cl.buffers.Destroyable;
import com.controllerface.bvge.gpu.cl.buffers.ResizableBuffer;
import com.controllerface.bvge.gpu.cl.programs.GPUCrud;
import com.controllerface.bvge.gpu.cl.programs.GPUProgram;
import com.controllerface.bvge.ecs.ECS;
import com.controllerface.bvge.ecs.components.ComponentType;
import com.controllerface.bvge.ecs.components.EntityIndex;
import com.controllerface.bvge.editor.Editor;
import com.controllerface.bvge.memory.groups.CoreBufferGroup;
import com.controllerface.bvge.memory.groups.ReferenceBufferGroup;
import com.controllerface.bvge.memory.groups.RenderBufferGroup;
import com.controllerface.bvge.memory.groups.UnorderedCoreBufferGroup;
import com.controllerface.bvge.memory.references.ReferenceController;
import com.controllerface.bvge.memory.sectors.*;
import com.controllerface.bvge.memory.types.CoreBufferType;
import com.controllerface.bvge.memory.types.ReferenceBufferType;
import com.controllerface.bvge.memory.types.RenderBufferType;
import com.controllerface.bvge.models.geometry.ModelRegistry;
import com.controllerface.bvge.physics.PhysicsEntityBatch;
import com.controllerface.bvge.physics.PhysicsObjects;
import com.controllerface.bvge.game.Constants;
import com.controllerface.bvge.core.Window;
import com.controllerface.bvge.events.Event;

import java.util.Queue;
import java.util.concurrent.BrokenBarrierException;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.CyclicBarrier;

import static com.controllerface.bvge.models.geometry.ModelRegistry.*;
import static org.lwjgl.opencl.CL10.clFinish;

public class GPUCoreMemory implements Destroyable
{
    private static final String BUF_NAME_SECTOR          = "Live Sectors";
    private static final String BUF_NAME_RENDER          = "Render Mirror";
    private static final String BUF_NAME_REFERENCE       = "Reference Data";
    private static final String BUF_NAME_SECTOR_EGRESS_A = "Sector Egress A";
    private static final String BUF_NAME_SECTOR_EGRESS_B = "Sector Egress B";
    private static final String BUF_NAME_BROKEN_EGRESS_A = "Broken Egress A";
    private static final String BUF_NAME_BROKEN_EGRESS_B = "Broken Egress B";
    private static final String BUF_NAME_OBJECT_EGRESS_A = "Object Egress A";
    private static final String BUF_NAME_OBJECT_EGRESS_B = "Object Egress B";

    private static final long ENTITY_INIT   = 10_000L;
    private static final long HULL_INIT     = 10_000L;
    private static final long EDGE_INIT     = 24_000L;
    private static final long POINT_INIT    = 50_000L;
    private static final long DELETE_INIT   = 10_000L;

    private final GPUProgram p_gpu_crud = new GPUCrud();

    /**
     * When data is stored in the mirror buffer, the index values of the core memory buffers are cached
     * here, so they can be referenced by rendering tasks, which are using a mirror buffer that may
     * differ in contents. Using these cached index values allows physics and rendering tasks to run
     * concurrently without interfering with each other.
     */
    private int last_hull_index   = 0;
    private int last_point_index  = 0;
    private int last_edge_index   = 0;
    private int last_entity_index = 0;

    private final int[] next_egress_counts = new int[8];
    private final int[] last_egress_counts = new int[8];
    private final OrderedSectorInput sector_ingress_buffer;
    private final FlippableContainer<UnorderedSectorOutput> sector_egress_buffer;
    private final FlippableContainer<BrokenObjectBuffer> broken_egress_buffer;
    private final FlippableContainer<CollectedObjectBuffer> object_egress_buffer;
    private final CoreBufferGroup sector_buffers;
    private final SectorController sector_controller;
    private final RenderBufferGroup render_buffers;
    private final ReferenceBufferGroup reference_buffers;
    private final ReferenceController reference_controller;
    private final SectorCompactor sector_compactor;

    /**
     * This barrier is used to facilitate co-operation between the sector loading thread and the main loop.
     * Each iteration, the sector loader waits on this barrier once it is done loading sectors, and then the
     * main loop does the same, tripping the barrier, which it then immediately resets.
     */
    private final CyclicBarrier world_barrier = new CyclicBarrier(4);
    private final Queue<Event> event_queue = new ConcurrentLinkedQueue<>();
    private final ECS ecs;

    public GPUCoreMemory(ECS ecs)
    {
        this.ecs = ecs;
        Window.get().event_bus().register(event_queue, Event.Type.SELECT_BLOCK);

        p_gpu_crud.init();

        this.sector_buffers       = new CoreBufferGroup(BUF_NAME_SECTOR, GPGPU.ptr_compute_queue, ENTITY_INIT, HULL_INIT, EDGE_INIT, POINT_INIT);
        this.sector_controller    = new SectorController(GPGPU.ptr_compute_queue, this.p_gpu_crud, this.sector_buffers);
        this.render_buffers       = new RenderBufferGroup(BUF_NAME_RENDER, GPGPU.ptr_compute_queue, ENTITY_INIT, HULL_INIT, EDGE_INIT, POINT_INIT);
        this.reference_buffers    = new ReferenceBufferGroup(BUF_NAME_REFERENCE, GPGPU.ptr_compute_queue);
        this.reference_controller = new ReferenceController(GPGPU.ptr_compute_queue, this.p_gpu_crud, this.reference_buffers);
        this.sector_compactor     = new SectorCompactor(GPGPU.ptr_compute_queue, sector_controller, sector_buffers, ENTITY_INIT, HULL_INIT, EDGE_INIT, POINT_INIT, DELETE_INIT);

        var sector_egress_a = new UnorderedSectorOutput(BUF_NAME_SECTOR_EGRESS_A, GPGPU.ptr_sector_queue, this, ENTITY_INIT, HULL_INIT, EDGE_INIT, POINT_INIT);
        var sector_egress_b = new UnorderedSectorOutput(BUF_NAME_SECTOR_EGRESS_B, GPGPU.ptr_sector_queue, this, ENTITY_INIT, HULL_INIT, EDGE_INIT, POINT_INIT);
        var broken_egress_a = new BrokenObjectBuffer(BUF_NAME_BROKEN_EGRESS_A, GPGPU.ptr_sector_queue, this);
        var broken_egress_b = new BrokenObjectBuffer(BUF_NAME_BROKEN_EGRESS_B, GPGPU.ptr_sector_queue, this);
        var object_egress_a = new CollectedObjectBuffer(BUF_NAME_OBJECT_EGRESS_A, GPGPU.ptr_sector_queue, this);
        var object_egress_b = new CollectedObjectBuffer(BUF_NAME_OBJECT_EGRESS_B, GPGPU.ptr_sector_queue, this);

        this.sector_ingress_buffer = new OrderedSectorInput(GPGPU.ptr_sector_queue, this);
        this.sector_egress_buffer  = new FlippableContainer<>(sector_egress_a, sector_egress_b);
        this.broken_egress_buffer  = new FlippableContainer<>(broken_egress_a, broken_egress_b);
        this.object_egress_buffer  = new FlippableContainer<>(object_egress_a, object_egress_b);
    }

    public int last_point()
    {
        return last_point_index;
    }

    public int last_entity()
    {
        return last_entity_index;
    }

    public int last_hull()
    {
        return last_hull_index;
    }

    public int last_edge()
    {
        return last_edge_index;
    }

    public SectorContainer sector_container()
    {
        return sector_controller;
    }

    public ReferenceContainer reference_container()
    {
        return reference_controller;
    }

    public ResizableBuffer get_buffer(ReferenceBufferType referenceBufferType)
    {
        return reference_buffers.buffer(referenceBufferType);
    }

    public ResizableBuffer get_buffer(RenderBufferType renderBufferType)
    {
        return render_buffers.buffer(renderBufferType);
    }

    public ResizableBuffer get_buffer(CoreBufferType coreBufferType)
    {
        return sector_buffers.buffer(coreBufferType);
    }

    public void swap_render_buffers()
    {
        render_buffers.copy_from(sector_buffers);

        last_edge_index   = sector_controller.next_edge();
        last_entity_index = sector_controller.next_entity();
        last_hull_index   = sector_controller.next_hull();
        last_point_index  = sector_controller.next_point();
    }

    public void swap_egress_buffers()
    {
        last_egress_counts[0] = next_egress_counts[0];
        last_egress_counts[1] = next_egress_counts[1];
        last_egress_counts[2] = next_egress_counts[2];
        last_egress_counts[3] = next_egress_counts[3];
        last_egress_counts[4] = next_egress_counts[4];
        last_egress_counts[5] = next_egress_counts[5];
        last_egress_counts[6] = next_egress_counts[6];
        last_egress_counts[7] = next_egress_counts[7];

        sector_egress_buffer.flip();
        broken_egress_buffer.flip();
        object_egress_buffer.flip();
    }

    public void release_world_barrier()
    {
        world_barrier.reset();
    }

    public void await_world_barrier()
    {
        if (world_barrier.isBroken()) return;
        try { world_barrier.await(); }
        catch (InterruptedException _) { }
        catch (BrokenBarrierException e)
        {
            if (!Window.get().is_closing()) throw new RuntimeException(e);
        }
    }

    public void load_entity_batch(PhysicsEntityBatch batch)
    {
        for (var entity : batch.entities)
        {
            PhysicsObjects.load_entity(sector_ingress_buffer, entity);
        }
        for (var block : batch.blocks)
        {
            PhysicsObjects.base_block(sector_ingress_buffer,
                block.x(),
                block.y(),
                block.size(),
                block.mass(),
                block.friction(),
                block.restitution(),
                block.entity_flags(),
                block.hull_flags(),
                block.material(),
                block.hits());
        }
        for (var shard : batch.shards)
        {
            int id = shard.spike()
                ? ModelRegistry.BASE_SPIKE_INDEX
                : shard.flip()
                    ? L_SHARD_INDEX
                    : R_SHARD_INDEX;

            int shard_flags = shard.hull_flags();

            PhysicsObjects.tri(sector_ingress_buffer,
                shard.x(),
                shard.y(),
                shard.size(),
                shard.entity_flags(),
                shard_flags,
                shard.mass(),
                shard.friction(),
                shard.restitution(),
                id,
                shard.material());
        }
        for (var liquid : batch.fluids)
        {
            PhysicsObjects.liquid_particle(sector_ingress_buffer,
                liquid.x(),
                liquid.y(),
                liquid.size(),
                liquid.mass(),
                liquid.friction(),
                liquid.restitution(),
                liquid.entity_flags(),
                liquid.hull_flags(),
                liquid.point_flags(),
                liquid.particle_fluid());
        }
    }

    public int[] last_egress_counts()
    {
        return last_egress_counts;
    }

    public void egress(int[] egress_counts)
    {
        next_egress_counts[0]  = egress_counts[0];
        next_egress_counts[1]  = egress_counts[1];
        next_egress_counts[2]  = egress_counts[2];
        next_egress_counts[3]  = egress_counts[3];
        next_egress_counts[4]  = egress_counts[4];
        next_egress_counts[5]  = egress_counts[5];
        next_egress_counts[6]  = egress_counts[6];
        next_egress_counts[7]  = egress_counts[7];

        int checksum = next_egress_counts[0]
            + next_egress_counts[6]
            + next_egress_counts[7];

        if (checksum == 0) return;

        clFinish(GPGPU.ptr_compute_queue);
        if (next_egress_counts[0] > 0)
        {
            sector_egress_buffer.front().egress(sector_controller.next_entity(), next_egress_counts);
        }
        if (next_egress_counts[6] > 0)
        {
            broken_egress_buffer.front().egress(sector_controller.next_entity(), next_egress_counts[6]);
        }
        if (next_egress_counts[7] > 0)
        {
            object_egress_buffer.front().egress(sector_controller.next_entity(), next_egress_counts[6]);
        }
        clFinish(GPGPU.ptr_sector_queue);
    }

    public void unload_collected(CollectedObjectBuffer.Raw raw, int count)
    {
        raw.ensure_space(count);
        object_egress_buffer.back().unload(raw, count);
        clFinish(GPGPU.ptr_sector_queue);
    }

    public void unload_broken(BrokenObjectBuffer.Raw raw, int count)
    {
        raw.ensure_space(count);
        broken_egress_buffer.back().unload(raw, count);
        clFinish(GPGPU.ptr_sector_queue);
    }

    public void unload_sectors(UnorderedCoreBufferGroup.Raw raw, int[] egress_counts)
    {
        raw.ensure_space(egress_counts);
        sector_egress_buffer.back().unload(raw, egress_counts);
        clFinish(GPGPU.ptr_sector_queue);
    }

    public void swap_ingress_buffers()
    {
        long sd = Editor.ACTIVE ? System.nanoTime() : 0;

        Event next_event;
        while ((next_event = event_queue.poll()) != null)
        {
            EntityIndex block_cursor = ComponentType.BlockCursorId.forEntity(ecs, Constants.PLAYER_ID);
            assert block_cursor != null : "null block selector id";
            if (next_event instanceof Event.SelectBlock(var _, var solid))
            {
                if (solid == null) sector_controller.clear_block_cursor(block_cursor.index());
                else sector_controller.update_block_cursor(block_cursor.index(), solid.mineral_number);
            }
        }

        int point_count         = sector_ingress_buffer.next_point();
        int edge_count          = sector_ingress_buffer.next_edge();
        int hull_count          = sector_ingress_buffer.next_hull();
        int entity_count        = sector_ingress_buffer.next_entity();
        int hull_bone_count     = sector_ingress_buffer.next_hull_bone();
        int armature_bone_count = sector_ingress_buffer.next_entity_bone();

        int total = point_count
            + edge_count
            + hull_count
            + entity_count
            + hull_bone_count
            + armature_bone_count;

        if (total == 0) return;

        int point_capacity         = point_count + sector_controller.next_point();
        int edge_capacity          = edge_count + sector_controller.next_edge();
        int hull_capacity          = hull_count + sector_controller.next_hull();
        int entity_capacity        = entity_count + sector_controller.next_entity();
        int hull_bone_capacity     = hull_bone_count + sector_controller.next_hull_bone();
        int armature_bone_capacity = armature_bone_count + sector_controller.next_entity_bone();

        sector_buffers.ensure_capacity_all(point_capacity,
                edge_capacity,
                hull_capacity,
                entity_capacity,
                hull_bone_capacity,
                armature_bone_capacity);

        clFinish(GPGPU.ptr_compute_queue);
        sector_ingress_buffer.merge_into(this.sector_controller);
        clFinish(GPGPU.ptr_sector_queue);

        sector_controller.expand(point_count, edge_count, hull_count, entity_count, hull_bone_count, armature_bone_count);

        if (Editor.ACTIVE)
        {
            long e = System.nanoTime() - sd;
            Editor.queue_event("sector_load", String.valueOf(e));
        }
    }

    public void update_mouse_position(int entity_index, float x, float y)
    {
        sector_controller.update_mouse_position(entity_index, x, y);
    }

    public void update_block_position(int entity_index, float x, float y)
    {
        sector_controller.update_block_position(entity_index, x, y);
    }

    public void place_block(int src, int dest)
    {
        sector_controller.place_block(src, dest);
    }

    public float[] read_entity_position(int entity_index)
    {
        return sector_controller.read_position(entity_index);
    }

    public void read_entity_info(int entity_index, float[] output)
    {
        sector_controller.read_entity_info(entity_index, output);
    }

    public void write_entity_info(int target,
                                  float[] accel,
                                  float[] current_time,
                                  float[] previous_time,
                                  float[] current_blend,
                                  short[] motion_state,
                                  int[] anim_layers,
                                  int[] anim_previous,
                                  int arm_flag)
    {
        sector_controller.write_entity_info(target,
            accel,
            current_time,
            previous_time,
            current_blend,
            motion_state,
            anim_layers,
            anim_previous,
            arm_flag);
    }

    public int[] count_egress_entities()
    {
        return sector_controller.count_egress_entities();
    }

    public void delete_and_compact()
    {
        sector_compactor.delete_and_compact();
    }

    @Override
    public void destroy()
    {
        sector_buffers.destroy();
        sector_controller.destroy();
        sector_ingress_buffer.destroy();
        sector_egress_buffer.destroy();
        broken_egress_buffer.destroy();
        object_egress_buffer.destroy();
        render_buffers.destroy();
        reference_buffers.destroy();
        p_gpu_crud.destroy();
    }
}
