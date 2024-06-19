package com.controllerface.bvge.physics;

import com.controllerface.bvge.geometry.UnloadedEntity;
import com.controllerface.bvge.substances.Solid;

import java.util.ArrayList;
import java.util.List;

public class PhysicsEntityBatch
{
    //public final Sector sector;

    public record Block(boolean dynamic, float x, float y, float size, float mass, float friction, float restitution, int flags, Solid material, int[] hits) { }
    public record Shard(boolean spike, boolean flip, float x, float y, float size, int flags, float mass, float friction, float restitution, Solid material) { }
    public record Liquid(float x, float y, float size, float mass, float friction, float restitution, int flags, int point_flags, com.controllerface.bvge.substances.Liquid particle_fluid) { }


    public final List<Block> blocks = new ArrayList<>();
    public final List<Shard> shards = new ArrayList<>();
    public final List<Liquid> liquids = new ArrayList<>();

    public final List<UnloadedEntity> entities = new ArrayList<>();

    public PhysicsEntityBatch()
    {
        //this.sector = sector;
    }

    public void new_block(boolean dynamic, float x, float y, float size, float mass, float friction, float restitution, int flags, Solid block_material, int[] hits)
    {
        blocks.add(new Block(dynamic, x, y, size, mass, friction, restitution, flags, block_material, hits));
    }

    public void new_shard(boolean spike, boolean flip, float x, float y, float size, int flags, float mass, float friction, float restitution, Solid material)
    {
        shards.add(new Shard(spike, flip, x, y, size, flags, mass, friction, restitution, material));
    }

    public void new_liquid(float x, float y, float size, float mass, float friction, float restitution, int flags, int point_flags, com.controllerface.bvge.substances.Liquid particle_fluid)
    {
        liquids.add(new Liquid(x, y, size, mass, friction, restitution, flags, point_flags, particle_fluid));
    }

    public void new_entity(UnloadedEntity entity)
    {
        entities.add(entity);
    }
}
