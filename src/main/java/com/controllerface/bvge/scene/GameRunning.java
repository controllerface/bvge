package com.controllerface.bvge.scene;

import com.controllerface.bvge.ecs.components.Transform;
import com.controllerface.bvge.ecs.components.Component;
import com.controllerface.bvge.ecs.components.ControlPoints;
import com.controllerface.bvge.ecs.ECS;
import com.controllerface.bvge.ecs.components.RigidBody2D;
import com.controllerface.bvge.ecs.Sprite;
import com.controllerface.bvge.ecs.components.SpriteComponent;
import com.controllerface.bvge.util.AssetPool;
import com.controllerface.bvge.ecs.components.QuadRectangle;
import org.joml.Random;
import org.joml.Vector2f;
import org.joml.Vector4f;

public class GameRunning extends GameMode
{
    private final ECS ecs;

    public GameRunning(ECS ecs)
    {
        this.ecs = ecs;
    }

    private int testBoxSize = 30;

    private void genNPCs(float spacing, float size)
    {
        var rand = new Random();
        for (int i = 0; i < testBoxSize; i++)
        {
            for (int j = 0; j < testBoxSize; j++)
            {
                float r = rand.nextFloat() / 3.0f;
                float g = rand.nextFloat() / 2.0f;
                float b = rand.nextFloat();

                float x = 100 + i * spacing;
                float y = 100 + j * spacing;

                var npc = ecs.registerEntity(null);
                var scomp2 = new SpriteComponent();
                var sprite2 = new Sprite();
                var tex2 = AssetPool.getTexture("assets/images/blendImage2.png");
                sprite2.setTexture(tex2);
                var transform2 = new Transform();
                transform2.scale.x = size;
                transform2.scale.y = size;
                transform2.position.x = x;
                transform2.position.y = y;
                sprite2.setHeight(32);
                sprite2.setWidth(32);
                //scomp2.setSprite(sprite2);
                scomp2.setColor(new Vector4f(r,g,b,1));
                ecs.attachComponent(npc, Component.SpriteComponent, scomp2);
                ecs.attachComponent(npc, Component.Transform, transform2);
                ecs.attachComponent(npc, Component.RigidBody2D, RigidBody2D.simpleBox(x, y, size, npc));
                ecs.attachComponent(npc, Component.BoundingBox, new QuadRectangle(0,0,0,0));

            }
        }
    }

    @Override
    public void load()
    {
        // player entity
        var player = ecs.registerEntity("player");
        var scomp = new SpriteComponent();
        var sprite = new Sprite();
        var tex = AssetPool.getTexture("assets/images/blendImage1.png");
        sprite.setTexture(tex);
        var transform =  new Transform();
        transform.scale.x = 32f;
        transform.scale.y = 32f;
        transform.position.x = 500f;
        transform.position.y = 50f;
        sprite.setHeight(16);
        sprite.setWidth(16);
        scomp.setSprite(sprite);
        //scomp.setColor(new Vector4f(0,0,0,1));
        ecs.attachComponent(player, Component.SpriteComponent, scomp);
        ecs.attachComponent(player, Component.Transform, transform);
        ecs.attachComponent(player, Component.ControlPoints, new ControlPoints());
        ecs.attachComponent(player, Component.RigidBody2D, RigidBody2D.simpleBox(500,50, 32, player));
        ecs.attachComponent(player, Component.BoundingBox, new QuadRectangle(0,0,0,0));

        genNPCs(2, 2);
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
