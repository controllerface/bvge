package com.controllerface.bvge.physics;

import com.controllerface.bvge.game.Sector;
import com.controllerface.bvge.substances.Solid;

import java.util.ArrayList;
import java.util.List;

public class PhysicsEntityBatch
{
    public final Sector sector;

    public record Block(boolean dynamic, float x, float y, float size, float mass, float friction, float restitution, int flags, Solid material) { }
    public record Shard(boolean spike, float x, float y, float size, int flags, float mass, float friction, float restitution, Solid material) { }
    public record Liquid(float x, float y, float size, float mass, float friction, float restitution, int flags, int point_flags, com.controllerface.bvge.substances.Liquid particle_fluid) { }

    public record Entity(int model_id, float x, float y, float size, float mass, float friction, float restitution, int flags, int uv_offset) { }

    public final List<Block> blocks = new ArrayList<>();
    public final List<Shard> shards = new ArrayList<>();
    public final List<Liquid> liquids = new ArrayList<>();

    public final List<Entity> entities = new ArrayList<>();

    public PhysicsEntityBatch(Sector sector)
    {
        this.sector = sector;
    }

    public void new_block(boolean dynamic, float x, float y, float size, float mass, float friction, float restitution, int flags, Solid block_material)
    {
        blocks.add(new Block(dynamic, x, y, size, mass, friction, restitution, flags, block_material));
    }

    public void new_shard(boolean spike, float x, float y, float size, int flags, float mass, float friction, float restitution, Solid material)
    {
        shards.add(new Shard(spike, x, y, size, flags, mass, friction, restitution, material));
    }

    public void new_liquid(float x, float y, float size, float mass, float friction, float restitution, int flags, int point_flags, com.controllerface.bvge.substances.Liquid particle_fluid)
    {
        liquids.add(new Liquid(x, y, size, mass, friction, restitution, flags, point_flags, particle_fluid));
    }

    public void new_entity(int model_id, float x, float y, float size, float mass, float friction, float restitution, int flags, int uv_offset)
    {
        entities.add(new Entity(model_id, x, y, size, mass, friction, restitution, flags, uv_offset));
    }
}
