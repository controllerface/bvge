package com.controllerface.bvge.scene;

import com.controllerface.bvge.Camera;
import com.controllerface.bvge.GameObject;
import com.controllerface.bvge.TransformEX;
import com.controllerface.bvge.ecs.ComponentType;
import com.controllerface.bvge.ecs.ECS;
import com.controllerface.bvge.rendering.GridLines;
import com.controllerface.bvge.rendering.Sprite;
import com.controllerface.bvge.rendering.SpriteComponentOLD;
import com.controllerface.bvge.rendering.SpriteComponentEX;
import com.controllerface.bvge.util.AssetPool;
import org.joml.Vector2f;
import org.joml.Vector4f;

public class GenericScene extends Scene
{
    //GameObject sceneData = this.createGameObject("generic stuff");
    private final ECS ecs;

    public GenericScene(ECS ecs)
    {
        this.ecs = ecs;
    }

    @Override
    public void load()
    {
        // test entity
        var player = ecs.registerEntity("player");
        var scomp = new SpriteComponentEX();
        var sprite = new Sprite();
        var transform =  new TransformEX();
        transform.scale.x = 32f;
        transform.scale.y = 32f;
        transform.position.x = 5.0f;
        transform.position.y = 5.0f;
        sprite.setHeight(32);
        sprite.setWidth(32);
        scomp.setColor(new Vector4f(0,0,1,1));
        ecs.attachComponent(player, ComponentType.SpriteComponent, scomp);
        ecs.attachComponent(player, ComponentType.Transform, transform);

        //var comp = ecs.getComponentFor(player, ComponentType.SpriteComponent);
        //SpriteComponentEX r = ComponentType.SpriteComponent.coerce(comp);
        //System.out.println("DEBUG: component = " + r);
    }

    @Override
    public void init()
    {
        this.camera = new Camera(new Vector2f(-250, 0));
        //sceneData.addComponent(new GridLines());
        //sceneData.start();

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
        SpriteComponentOLD obj1sprite = new SpriteComponentOLD();
        obj1sprite.setSprite(sprite);
        obj1sprite.setColor(new Vector4f(1,0,0,1));
        obj1.addComponent(obj1sprite);

        this.addGameObjectToScene(obj1);
    }

    @Override
    public void update(float dt)
    {
        //sceneData.update(dt);
        this.camera.adjustProjection();
    }

    @Override
    public void imgui()
    {

    }
}
