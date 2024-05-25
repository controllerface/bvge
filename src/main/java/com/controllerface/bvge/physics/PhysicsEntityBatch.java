package com.controllerface.bvge.physics;

import com.controllerface.bvge.game.Sector;
import com.controllerface.bvge.substances.Liquid;
import com.controllerface.bvge.substances.Solid;

import java.util.ArrayList;
import java.util.List;

public class PhysicsEntityBatch
{
    public final Sector sector;

    public record BlockEntity(boolean dynamic, float x, float y, float size, float mass, float friction, float restitution, int flags, Solid block_material) { }
    public record LiquidEntity(float x, float y, float size, float mass, float friction, float restitution, int flags, int point_flags, Liquid particle_fluid) { }
    public record Tri(float x, float y, float size, int flags, float mass, float friction, float restitution) { }

    public final List<BlockEntity> blocks = new ArrayList<>();
    public final List<LiquidEntity> liquids = new ArrayList<>();
    public final List<Tri> tris = new ArrayList<>();

    public PhysicsEntityBatch(Sector sector)
    {
        this.sector = sector;
    }

    public void new_block(boolean dynamic, float x, float y, float size, float mass, float friction, float restitution, int flags, Solid block_material)
    {
        blocks.add(new BlockEntity(dynamic, x, y, size, mass, friction, restitution, flags, block_material));
    }

    public void new_liquid(float x, float y, float size, float mass, float friction, float restitution, int flags, int point_flags, Liquid particle_fluid)
    {
        liquids.add(new LiquidEntity(x, y, size, mass, friction, restitution, flags, point_flags, particle_fluid));
    }

    public void new_tri(float x, float y, float size, int flags, float mass, float friction, float restitution)
    {
        tris.add(new Tri(x, y, size, flags, mass, friction, restitution));
    }
}
