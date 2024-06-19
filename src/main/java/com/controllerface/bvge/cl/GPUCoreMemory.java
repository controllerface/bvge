package com.controllerface.bvge.cl;

import com.controllerface.bvge.cl.buffers.*;
import com.controllerface.bvge.cl.programs.GPUCrud;
import com.controllerface.bvge.cl.programs.GPUProgram;
import com.controllerface.bvge.game.world.sectors.SectorContainer;
import com.controllerface.bvge.game.world.sectors.*;
import com.controllerface.bvge.geometry.ModelRegistry;
import com.controllerface.bvge.physics.PhysicsEntityBatch;
import com.controllerface.bvge.physics.PhysicsObjects;
import com.controllerface.bvge.util.Constants;
import com.controllerface.bvge.window.Window;

import java.util.concurrent.BrokenBarrierException;
import java.util.concurrent.CyclicBarrier;

import static com.controllerface.bvge.geometry.ModelRegistry.*;
import static org.lwjgl.opencl.CL10.clFinish;

public class GPUCoreMemory implements SectorContainer
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
    private static final long DELETE_1_INIT = 10_000L;

    private final GPUProgram p_gpu_crud = new GPUCrud();

    /**
     * When data is stored in the mirror buffer, the index values of the core memory buffers are cached
     * here, so they can be referenced by rendering tasks, which are using a mirror buffer that may
     * differ in contents. Using these cached index values allows physics and rendering tasks to run
     * concurrently without interfering with each other.
     */
    private int last_hull_index       = 0;
    private int last_point_index      = 0;
    private int last_edge_index       = 0;
    private int last_entity_index     = 0;

    private final int[] next_egress_counts = new int[8];
    private final int[] last_egress_counts = new int[8];
    private final OrderedSectorInput sector_ingress_buffer;
    private final DoubleBuffer<UnorderedSectorOutput> sector_egress_buffer;
    private final DoubleBuffer<BrokenObjectBuffer> broken_egress_buffer;
    private final DoubleBuffer<CollectedObjectBuffer> object_egress_buffer;
    private final SectorBufferGroup sector_buffers;
    private final SectorController sector_controller;
    private final MirrorBufferGroup mirror_buffers;
    private final ReferenceBufferGroup reference_buffers;
    private final ReferenceController reference_controller;
    private final SectorCompactor sector_compactor;

    /**
     * This barrier is used to facilitate co-operation between the sector loading thread and the main loop.
     * Each iteration, the sector loader waits on this barrier once it is done loading sectors, and then the
     * main loop does the same, tripping the barrier, which it then immediately resets.
     */
    private final CyclicBarrier world_barrier = new CyclicBarrier(4);

    public GPUCoreMemory()
    {
        p_gpu_crud.init();

        this.sector_buffers       = new SectorBufferGroup(BUF_NAME_SECTOR, GPGPU.ptr_compute_queue, ENTITY_INIT, HULL_INIT, EDGE_INIT, POINT_INIT);
        this.sector_controller    = new SectorController(GPGPU.ptr_compute_queue, this.p_gpu_crud, this.sector_buffers);
        this.mirror_buffers       = new MirrorBufferGroup(BUF_NAME_RENDER, GPGPU.ptr_compute_queue, ENTITY_INIT, HULL_INIT, EDGE_INIT, POINT_INIT);
        this.reference_buffers    = new ReferenceBufferGroup(BUF_NAME_REFERENCE, GPGPU.ptr_compute_queue);
        this.reference_controller = new ReferenceController(GPGPU.ptr_compute_queue, this.p_gpu_crud, this.reference_buffers);
        this.sector_compactor     = new SectorCompactor(GPGPU.ptr_compute_queue, sector_controller, sector_buffers, ENTITY_INIT, HULL_INIT, EDGE_INIT, POINT_INIT, DELETE_1_INIT);

        var sector_egress_a = new UnorderedSectorOutput(BUF_NAME_SECTOR_EGRESS_A, GPGPU.ptr_sector_queue, this);
        var sector_egress_b = new UnorderedSectorOutput(BUF_NAME_SECTOR_EGRESS_B, GPGPU.ptr_sector_queue, this);
        var broken_egress_a = new BrokenObjectBuffer(BUF_NAME_BROKEN_EGRESS_A, GPGPU.ptr_sector_queue, this);
        var broken_egress_b = new BrokenObjectBuffer(BUF_NAME_BROKEN_EGRESS_B, GPGPU.ptr_sector_queue, this);
        var object_egress_a = new CollectedObjectBuffer(BUF_NAME_OBJECT_EGRESS_A, GPGPU.ptr_sector_queue, this);
        var object_egress_b = new CollectedObjectBuffer(BUF_NAME_OBJECT_EGRESS_B, GPGPU.ptr_sector_queue, this);

        this.sector_ingress_buffer = new OrderedSectorInput(GPGPU.ptr_sector_queue, this);
        this.sector_egress_buffer  = new DoubleBuffer<>(sector_egress_a, sector_egress_b);
        this.broken_egress_buffer  = new DoubleBuffer<>(broken_egress_a, broken_egress_b);
        this.object_egress_buffer  = new DoubleBuffer<>(object_egress_a, object_egress_b);
    }

    public ResizableBuffer buffer(BufferType bufferType)
    {
        return switch (bufferType)
        {
            case BROKEN_POSITIONS,
                 BROKEN_UV_OFFSETS,
                 BROKEN_MODEL_IDS,
                 COLLECTED_UV_OFFSETS,
                 COLLECTED_FLAG,
                 COLLECTED_TYPE -> null;

            case ANIM_POS_CHANNEL,
                 ANIM_ROT_CHANNEL,
                 ANIM_SCL_CHANNEL,
                 VERTEX_REFERENCE,
                 VERTEX_UV_TABLE,
                 VERTEX_WEIGHT,
                 VERTEX_TEXTURE_UV,
                 MODEL_TRANSFORM,
                 ANIM_TIMING_INDEX,
                 MESH_VERTEX_TABLE,
                 MESH_FACE_TABLE,
                 MESH_FACE,
                 BONE_REFERENCE,
                 BONE_ANIM_CHANNEL_TABLE,
                 BONE_BIND_POSE,
                 ANIM_DURATION,
                 ANIM_TICK_RATE,
                 ANIM_KEY_FRAME,
                 ANIM_FRAME_TIME -> reference_buffers.get_buffer(bufferType);

            case MIRROR_POINT,
                 MIRROR_POINT_ANTI_GRAV,
                 MIRROR_POINT_HIT_COUNT,
                 MIRROR_POINT_VERTEX_REFERENCE,
                 MIRROR_EDGE,
                 MIRROR_EDGE_FLAG,
                 MIRROR_HULL,
                 MIRROR_HULL_AABB,
                 MIRROR_HULL_ENTITY_ID,
                 MIRROR_HULL_FLAG,
                 MIRROR_HULL_MESH_ID,
                 MIRROR_HULL_UV_OFFSET,
                 MIRROR_HULL_INTEGRITY,
                 MIRROR_HULL_POINT_TABLE,
                 MIRROR_HULL_ROTATION,
                 MIRROR_HULL_SCALE,
                 MIRROR_ENTITY,
                 MIRROR_ENTITY_FLAG,
                 MIRROR_ENTITY_MODEL_ID,
                 MIRROR_ENTITY_ROOT_HULL -> mirror_buffers.get_buffer(bufferType);

            case POINT,
                 POINT_HIT_COUNT,
                 POINT_FLAG,
                 POINT_HULL_INDEX,
                 POINT_VERTEX_REFERENCE,
                 POINT_BONE_TABLE,
                 POINT_ANTI_GRAV,
                 HULL,
                 HULL_ROTATION,
                 HULL_UV_OFFSET,
                 HULL_MESH_ID,
                 HULL_RESTITUTION,
                 HULL_INTEGRITY,
                 HULL_FRICTION,
                 HULL_FLAG,
                 HULL_EDGE_TABLE,
                 HULL_POINT_TABLE,
                 HULL_BONE_INV_BIND_POSE,
                 HULL_BONE_BIND_POSE,
                 HULL_BONE_TABLE,
                 HULL_ENTITY_ID,
                 HULL_BONE,
                 HULL_SCALE,
                 HULL_AABB,
                 HULL_AABB_INDEX,
                 HULL_AABB_KEY_TABLE,
                 EDGE,
                 EDGE_LENGTH,
                 EDGE_FLAG,
                 ENTITY,
                 ENTITY_TRANSFORM_ID,
                 ENTITY_ROOT_HULL,
                 ENTITY_MODEL_ID,
                 ENTITY_MASS,
                 ENTITY_HULL_TABLE,
                 ENTITY_BONE_TABLE,
                 ENTITY_TYPE,
                 ENTITY_FLAG,
                 ENTITY_BONE_PARENT_ID,
                 ENTITY_BONE_REFERENCE_ID,
                 ENTITY_BONE,
                 ENTITY_ANIM_INDEX,
                 ENTITY_MOTION_STATE,
                 ENTITY_ACCEL,
                 ENTITY_ANIM_BLEND,
                 ENTITY_ANIM_ELAPSED -> sector_buffers.get_buffer(bufferType);
        };
    }

    public void mirror_render_buffers()
    {
        mirror_buffers.mirror(sector_buffers);

        last_edge_index     = sector_controller.edge_index();
        last_entity_index   = sector_controller.entity_index();
        last_hull_index     = sector_controller.hull_index();
        last_point_index    = sector_controller.point_index();
    }

    // index methods

    public int next_mesh()
    {
        return reference_controller.mesh_index();
    }

    @Override
    public int next_entity()
    {
        return sector_controller.entity_index();
    }

    @Override
    public int next_hull()
    {
        return sector_controller.hull_index();
    }

    @Override
    public int next_point()
    {
        return sector_controller.point_index();
    }

    @Override
    public int next_edge()
    {
        return sector_controller.edge_index();
    }

    @Override
    public int next_hull_bone()
    {
        return sector_controller.hull_bone_index();
    }

    @Override
    public int next_armature_bone()
    {
        return sector_controller.entity_bone_index();
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

    public void flip_egress_buffers()
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
            if (block.dynamic())
            {
                PhysicsObjects.base_block(sector_ingress_buffer, block.x(), block.y(), block.size(), block.mass(), block.friction(), block.restitution(), block.flags(), block.material(), block.hits());
            }
            else
            {
                int flags = block.flags() | Constants.HullFlags.IS_STATIC._int;
                PhysicsObjects.base_block(sector_ingress_buffer, block.x(), block.y(), block.size(), block.mass(), block.friction(), block.restitution(), flags, block.material(), block.hits());
            }
        }
        for (var shard : batch.shards)
        {
            int id = shard.spike()
                ? ModelRegistry.BASE_SPIKE_INDEX
                : shard.flip()
                    ? L_SHARD_INDEX
                    : R_SHARD_INDEX;

            int shard_flags = shard.flags();

            PhysicsObjects.tri(sector_ingress_buffer, shard.x(), shard.y(), shard.size(), shard_flags, shard.mass(), shard.friction(), shard.restitution(), id, shard.material());
        }
        for (var liquid : batch.liquids)
        {
            PhysicsObjects.liquid_particle(sector_ingress_buffer, liquid.x(), liquid.y(), liquid.size(), liquid.mass(), liquid.friction(), liquid.restitution(), liquid.flags(), liquid.point_flags(), liquid.particle_fluid());
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
            sector_egress_buffer.front().egress(sector_controller.entity_index(), next_egress_counts);
        }
        if (next_egress_counts[6] > 0)
        {
            broken_egress_buffer.front().egress(sector_controller.entity_index(), next_egress_counts[6]);
        }
        if (next_egress_counts[7] > 0)
        {
            object_egress_buffer.front().egress(sector_controller.entity_index(), next_egress_counts[6]);
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

    public void unload_sectors(UnorderedSectorBufferGroup.Raw raw, int[] egress_counts)
    {
        raw.ensure_space(egress_counts);
        sector_egress_buffer.back().unload(raw, egress_counts);
        clFinish(GPGPU.ptr_sector_queue);
    }

    public void transfer_world_input()
    {
        int point_count         = sector_ingress_buffer.next_point();
        int edge_count          = sector_ingress_buffer.next_edge();
        int hull_count          = sector_ingress_buffer.next_hull();
        int entity_count        = sector_ingress_buffer.next_entity();
        int hull_bone_count     = sector_ingress_buffer.next_hull_bone();
        int armature_bone_count = sector_ingress_buffer.next_armature_bone();

        int total = point_count
            + edge_count
            + hull_count
            + entity_count
            + hull_bone_count
            + armature_bone_count;

        if (total == 0) return;

        int point_capacity         = point_count + next_point();
        int edge_capacity          = edge_count + next_edge();
        int hull_capacity          = hull_count + next_hull();
        int entity_capacity        = entity_count + next_entity();
        int hull_bone_capacity     = hull_bone_count + next_hull_bone();
        int armature_bone_capacity = armature_bone_count + next_armature_bone();

        sector_buffers.ensure_capacity_all(point_capacity,
                edge_capacity,
                hull_capacity,
                entity_capacity,
                hull_bone_capacity,
                armature_bone_capacity);

        clFinish(GPGPU.ptr_compute_queue);
        sector_ingress_buffer.merge_into_parent(this);
        clFinish(GPGPU.ptr_sector_queue);

        sector_controller.expand(point_count, edge_count, hull_count, entity_count, hull_bone_count, armature_bone_count);
    }

    public int new_animation_timings(float duration, float tick_rate)
    {
        return reference_controller.new_animation_timings(duration, tick_rate);
    }

    public int new_bone_channel(int anim_timing_index, int[] pos_table, int[] rot_table, int[] scl_table)
    {
        return reference_controller.new_bone_channel(anim_timing_index, pos_table, rot_table, scl_table);
    }

    public int new_keyframe(float[] frame, float time)
    {
        return reference_controller.new_keyframe(frame, time);
    }

    public int new_texture_uv(float u, float v)
    {
        return reference_controller.new_texture_uv(u, v);
    }

    public int new_mesh_reference(int[] vertex_table, int[] face_table)
    {
        return reference_controller.new_mesh_reference(vertex_table, face_table);
    }

    public int new_mesh_face(int[] face)
    {
        return reference_controller.new_mesh_face(face);
    }

    public int new_vertex_reference(float x, float y, float[] weights, int[] uv_table)
    {
        return reference_controller.new_vertex_reference(x, y, weights, uv_table);
    }

    public int new_bone_bind_pose(float[] bone_data)
    {
        return reference_controller.new_bone_bind_pose(bone_data);
    }

    public int new_bone_reference(float[] bone_data)
    {
        return reference_controller.new_bone_reference(bone_data);
    }

    public int new_model_transform(float[] transform_data)
    {
        return reference_controller.new_model_transform(transform_data);
    }

    public void set_bone_channel_table(int bind_pose_target, int[] channel_table)
    {
        reference_controller.set_bone_channel_table(bind_pose_target, channel_table);
    }

    public void update_position(int entity_index, float x, float y)
    {
        sector_controller.update_position(entity_index, x, y);
    }

    public float[] read_position(int entity_index)
    {
        return sector_controller.read_position(entity_index);
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
    public int new_point(float[] position, int[] bone_ids, int vertex_index, int hull_index, int hit_count, int flags)
    {
        return sector_controller.create_point(position, bone_ids, vertex_index, hull_index, hit_count, flags);
    }

    @Override
    public int new_edge(int p1, int p2, float l, int flags)
    {
        return sector_controller.create_edge(p1, p2, l, flags);
    }

    @Override
    public int new_hull(int mesh_id,
                        float[] position,
                        float[] scale,
                        float[] rotation,
                        int[] point_table,
                        int[] edge_table,
                        int[] bone_table,
                        float friction,
                        float restitution,
                        int entity_id,
                        int uv_offset,
                        int flags)
    {
        return sector_controller.create_hull(mesh_id, position, scale, rotation, point_table, edge_table, bone_table, friction, restitution, entity_id, uv_offset, flags);
    }

    @Override
    public int new_hull_bone(float[] bone_data, int bind_pose_id, int inv_bind_pose_id)
    {
        return sector_controller.create_hull_bone(bone_data, bind_pose_id, inv_bind_pose_id);
    }

    @Override
    public int new_entity(float x, float y, float z, float w,
                          int[] hull_table,
                          int[] bone_table,
                          float mass,
                          int anim_index,
                          float anim_time,
                          int root_hull,
                          int model_id,
                          int model_transform_id,
                          int type,
                          int flags)
    {
        return sector_controller.create_entity(x, y, z, w, hull_table, bone_table, mass, anim_index, anim_time, root_hull, model_id, model_transform_id, type, flags);
    }

    @Override
    public int new_armature_bone(int bone_reference, int bone_parent_id, float[] bone_data)
    {
        return sector_controller.create_entity_bone(bone_reference, bone_parent_id, bone_data);
    }

    @Override
    public void destroy()
    {
        sector_buffers.destroy();
        sector_controller.destroy();
        sector_ingress_buffer.destroy();
        sector_egress_buffer.front().destroy();
        sector_egress_buffer.back().destroy();
        broken_egress_buffer.front().destroy();
        broken_egress_buffer.back().destroy();
        object_egress_buffer.front().destroy();
        object_egress_buffer.back().destroy();
        mirror_buffers.destroy();
        reference_buffers.destroy();
        p_gpu_crud.destroy();
    }
}
