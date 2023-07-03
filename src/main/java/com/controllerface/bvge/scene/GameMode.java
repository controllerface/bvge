package com.controllerface.bvge.scene;

import com.controllerface.bvge.ecs.systems.physics.SpatialMapEX;

public abstract class GameMode
{
    public GameMode()
    {

    }

    abstract public void start();

    abstract public void update(float dt);


    abstract public void resizeSpatialMap(int width, int height);

    abstract public SpatialMapEX getSpatialMap();

    abstract public void load();
}