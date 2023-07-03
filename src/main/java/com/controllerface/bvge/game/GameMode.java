package com.controllerface.bvge.game;

public abstract class GameMode
{
    public GameMode()
    {

    }

    abstract public void start();

    abstract public void update(float dt);

    abstract public void resizeSpatialMap(int width, int height);

    abstract public void load();
}