package com.controllerface.bvge.scene;

import com.controllerface.bvge.Camera;

public abstract class GameMode
{
    //protected Renderer renderer = new Renderer();
    protected Camera camera;
    private boolean isRunning = false;

    protected boolean levelLoaded = false;

    public GameMode()
    {

    }

    public void start()
    {

    }



    public abstract void update(float dt);

    public Camera camera()
    {
        return this.camera;
    }


    public void saveExit()
    {
        // empty for now
    }

    public void load()
    {
        // empty for now
    }
}