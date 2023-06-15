package com.controllerface.bvge.scene;

import com.controllerface.bvge.Camera;
import com.controllerface.bvge.GameObject;
import com.controllerface.bvge.Transform;
import com.controllerface.bvge.rendering.GridLines;
import com.controllerface.bvge.rendering.SpriteRenderer;
import org.joml.Vector2f;
import org.joml.Vector4f;

public class GenericScene extends Scene
{
    GameObject sceneData = this.createGameObject("generic stuff");

    public GenericScene()
    {
    }

    @Override
    public void init()
    {
        loadResources();
        this.camera = new Camera(new Vector2f(-250, 0));
        sceneData.addComponent(new GridLines());
        sceneData.start();


        GameObject obj1 = this.createGameObject("Object 1");
        obj1.getComponent(Transform.class).position.x = 200;
        obj1.getComponent(Transform.class).position.y = 200;
        obj1.getComponent(Transform.class).scale.x = 32;
        obj1.getComponent(Transform.class).scale.y = 32;

//            new GameObject("Object 1",
//            new Transform(new Vector2f(200, 100), new Vector2f(256, 256)), 2);
        //obj1.addComponent(new SpriteRenderer(new Sprite(AssetPool.getTexture("assets/images/blendImage1.png"))));


        SpriteRenderer obj1sprite = new SpriteRenderer();
        obj1sprite.setColor(new Vector4f(1,0,0,1));
        obj1.addComponent(obj1sprite);
        //obj1.addComponent(new RigidBody());

        this.addGameObjectToScene(obj1);
    }

    private void loadResources()
    {
        // nothing here yet
    }

    @Override
    public void update(float dt)
    {
        sceneData.update(dt);
        this.camera.adjustProjection();
        for (GameObject go : this.gameObjects)
        {
            go.update(dt);
        }
    }

    @Override
    public void render()
    {
        this.renderer.render();
    }

    @Override
    public void imgui()
    {

    }
}
