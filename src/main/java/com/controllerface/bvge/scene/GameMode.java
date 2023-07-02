package com.controllerface.bvge.scene;

import com.controllerface.bvge.ecs.systems.physics.SpatialMapEX;

public abstract class GameMode
{
    //protected Renderer renderer = new Renderer();
    protected Camera camera;

    public GameMode()
    {

    }

    public void start()
    {

    }

    abstract public void update(float dt);

    public Camera camera()
    {
        return this.camera;
    }


    abstract public void resizeSpatialMap(int width, int height);

    abstract public SpatialMapEX getSpatialMap();

    public void saveExit()
    {
        // empty for now
    }

    public void load()
    {
        // empty for now
    }
}