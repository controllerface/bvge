package com.controllerface.bvge.ecs.systems;

import com.controllerface.bvge.game.Sector;
import com.controllerface.bvge.physics.PhysicsEntityBatch;

public interface WorldType
{
    PhysicsEntityBatch load_sector(Sector sector);
}
