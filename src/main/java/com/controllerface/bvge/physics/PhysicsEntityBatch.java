package com.controllerface.bvge.physics;

import com.controllerface.bvge.geometry.UnloadedEntity;
import com.controllerface.bvge.substances.Liquid;
import com.controllerface.bvge.substances.Solid;

import java.util.ArrayList;
import java.util.List;

public class PhysicsEntityBatch
{
    public record Block(float x, float y, float size, float mass, float friction, float restitution, int entity_flags, int hull_flags, Solid material, int[] hits) { }
    public record Shard(boolean spike, boolean flip, float x, float y, float size, int entity_flags, int hull_flags, float mass, float friction, float restitution, Solid material) { }
    public record Fluid(float x, float y, float size, float mass, float friction, float restitution, int entity_flags, int hull_flags, int point_flags, com.controllerface.bvge.substances.Liquid particle_fluid) { }

    public final List<Block> blocks = new ArrayList<>();
    public final List<Shard> shards = new ArrayList<>();
    public final List<Fluid> fluids = new ArrayList<>();

    public final List<UnloadedEntity> entities = new ArrayList<>();

    public PhysicsEntityBatch() {}

    public void new_block(float x, float y, float size, float mass, float friction, float restitution, int entity_flags, int hull_flags, Solid block_material, int[] hits)
    {
        blocks.add(new Block(x, y, size, mass, friction, restitution, entity_flags, hull_flags, block_material, hits));
    }

    public void new_shard(boolean spike, boolean flip, float x, float y, float size, int entity_flags, int hull_flags, float mass, float friction, float restitution, Solid material)
    {
        shards.add(new Shard(spike, flip, x, y, size, entity_flags, hull_flags, mass, friction, restitution, material));
    }

    public void new_liquid(float x, float y, float size, float mass, float friction, float restitution, int entity_flags, int hull_flags, int point_flags, Liquid particle_fluid)
    {
        fluids.add(new Fluid(x, y, size, mass, friction, restitution, entity_flags, hull_flags, point_flags, particle_fluid));
    }

    public void new_entity(UnloadedEntity entity)
    {
        entities.add(entity);
    }
}
