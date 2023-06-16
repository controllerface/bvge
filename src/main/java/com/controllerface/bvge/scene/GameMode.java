package com.controllerface.bvge.scene;

import com.controllerface.bvge.Camera;
import com.controllerface.bvge.GameObject;
import com.controllerface.bvge.Transform;

import java.util.ArrayList;
import java.util.List;

public abstract class GameMode
{
    //protected Renderer renderer = new Renderer();
    protected Camera camera;
    private boolean isRunning = false;
    protected List<GameObject> gameObjects = new ArrayList<>();

    protected boolean levelLoaded = false;

    public GameMode()
    {

    }

    public void start()
    {

    }

    public void addGameObjectToScene(GameObject go)
    {
//        if (!isRunning)
//        {
//            gameObjects.add(go);
//        }
//        else
//        {
//            gameObjects.add(go);
//            go.start();
//            this.renderer.add(go);
//        }
    }

    public GameObject getGameObject(int gameObjectId)
    {
        return this.gameObjects.stream()
            .filter(gameObject -> gameObject.getUid() == gameObjectId)
            .findFirst()
            .orElse(null);
    }

    public abstract void update(float dt);

    public Camera camera()
    {
        return this.camera;
    }


    public GameObject createGameObject(String name)
    {
        GameObject go = new GameObject(name);
        go.addComponent(new Transform());
        go.transform = go.getComponent(Transform.class);
        return go;
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