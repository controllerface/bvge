package com.controllerface.bvge.scene;

import com.controllerface.bvge.Camera;
import com.controllerface.bvge.TransformEX;
import com.controllerface.bvge.ecs.Component;
import com.controllerface.bvge.ecs.ControlPoints;
import com.controllerface.bvge.ecs.ECS;
import com.controllerface.bvge.ecs.RigidBody2D;
import com.controllerface.bvge.rendering.Sprite;
import com.controllerface.bvge.rendering.SpriteComponentEX;
import org.joml.Vector2f;
import org.joml.Vector4f;

public class GameRunning extends GameMode
{
    private final ECS ecs;

    public GameRunning(ECS ecs)
    {
        this.ecs = ecs;
    }

    @Override
    public void load()
    {
        // player entity
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
        ecs.attachComponent(player, Component.SpriteComponent, scomp);
        ecs.attachComponent(player, Component.Transform, transform);
        ecs.attachComponent(player, Component.ControlPoints, new ControlPoints());
        ecs.attachComponent(player, Component.RigidBody2D, RigidBody2D.simpleBox());
    }

    @Override
    public void start()
    {
        this.camera = new Camera(new Vector2f(0, 0));
    }

    @Override
    public void update(float dt)
    {
        this.camera.adjustProjection();
    }

}
