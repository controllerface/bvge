package com.controllerface.bvge.scene;

import com.controllerface.bvge.Camera;
import com.controllerface.bvge.Transform;
import com.controllerface.bvge.ecs.Component;
import com.controllerface.bvge.ecs.ControlPoints;
import com.controllerface.bvge.ecs.ECS;
import com.controllerface.bvge.ecs.RigidBody2D;
import com.controllerface.bvge.rendering.Sprite;
import com.controllerface.bvge.rendering.SpriteComponent;
import com.controllerface.bvge.util.AssetPool;
import com.controllerface.bvge.util.quadtree.QuadRectangle;
import org.joml.Vector2f;

public class GameRunning extends GameMode
{
    private final ECS ecs;

    public GameRunning(ECS ecs)
    {
        this.ecs = ecs;
    }

    private int testBoxSize = 25;

    private void genNPCs(int spacing, int size)
    {
        for (int i = 0; i < testBoxSize; i++)
        {
            for (int j = 0; j < testBoxSize; j++)
            {
                var npc = ecs.registerEntity(null);
                var scomp2 = new SpriteComponent();
                var sprite2 = new Sprite();
                var tex2 = AssetPool.getTexture("assets/images/blendImage2.png");
                sprite2.setTexture(tex2);
                var transform2 = new Transform();
                transform2.scale.x = 8f;
                transform2.scale.y = 8f;
                transform2.position.x = 0f;
                transform2.position.y = 0f;
                sprite2.setHeight(32);
                sprite2.setWidth(32);
                scomp2.setSprite(sprite2);
                //scomp.setColor(new Vector4f(0,0,0,1));
                ecs.attachComponent(npc, Component.SpriteComponent, scomp2);
                ecs.attachComponent(npc, Component.Transform, transform2);
                ecs.attachComponent(npc, Component.RigidBody2D,
                    RigidBody2D.simpleBox(200 + i * spacing, 200 + j * spacing, size, npc));
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
        transform.position.x = 50f;
        transform.position.y = 50f;
        sprite.setHeight(16);
        sprite.setWidth(16);
        scomp.setSprite(sprite);
        //scomp.setColor(new Vector4f(0,0,0,1));
        ecs.attachComponent(player, Component.SpriteComponent, scomp);
        ecs.attachComponent(player, Component.Transform, transform);
        ecs.attachComponent(player, Component.ControlPoints, new ControlPoints());
        ecs.attachComponent(player, Component.RigidBody2D, RigidBody2D.simpleBox(50,50, 32, player));
        ecs.attachComponent(player, Component.BoundingBox, new QuadRectangle(0,0,0,0));

        genNPCs(15, 15);
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
