package com.controllerface.bvge.scene;

import com.controllerface.bvge.Camera;
import com.controllerface.bvge.GameObject;
import com.controllerface.bvge.ecs.ComponentType;
import com.controllerface.bvge.ecs.ECS;
import com.controllerface.bvge.rendering.GridLines;
import com.controllerface.bvge.rendering.Sprite;
import com.controllerface.bvge.rendering.SpriteRenderer;
import com.controllerface.bvge.util.AssetPool;
import org.joml.Vector2f;
import org.joml.Vector4f;

public class GenericScene extends Scene
{
    GameObject sceneData = this.createGameObject("generic stuff");
    private ECS ecs = new ECS();

    public GenericScene()
    {
    }

    @Override
    public void load()
    {
        ecs.registerSystem((dt) ->
        {
            var message = "testing: " + dt;
            var comps = ecs.getComponents(ComponentType.SpriteRenderer);
            message += " c: " + comps.size();
            //System.out.println(message);
        });

        var player = ecs.registerEntity("player");
        ecs.attachComponent(player, ComponentType.SpriteRenderer, new SpriteRenderer());
        var comp = ecs.getComponentFor(player, ComponentType.SpriteRenderer);
        SpriteRenderer r = ComponentType.SpriteRenderer.coerce(comp);
        System.out.println("DEBUG: component = " + r);

    }

    @Override
    public void init()
    {
        this.camera = new Camera(new Vector2f(-250, 0));
        sceneData.addComponent(new GridLines());
        sceneData.start();

        // a single game object
        GameObject obj1 = this.createGameObject("Box");

        // places the object in world co-ordinates
        obj1.transform.position.x = 200;
        obj1.transform.position.y = 200;

        // scale must be applied or the object is effectively infinitely small
        obj1.transform.scale.x = 32;
        obj1.transform.scale.y = 32;

        // load in a basic square sprite
        Sprite sprite = new Sprite();
        sprite.setTexture(AssetPool.getTexture("assets/images/blendImage1.png"));
        SpriteRenderer obj1sprite = new SpriteRenderer();
        obj1sprite.setSprite(sprite);
        obj1sprite.setColor(new Vector4f(1,0,0,1));
        obj1.addComponent(obj1sprite);

        this.addGameObjectToScene(obj1);
    }

    @Override
    public void update(float dt)
    {
        ecs.run(dt);

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
