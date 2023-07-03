package com.controllerface.bvge.scene;

import com.controllerface.bvge.data.PhysicsObjects;
import com.controllerface.bvge.ecs.ECS;
import com.controllerface.bvge.ecs.Sprite;
import com.controllerface.bvge.ecs.components.*;
import com.controllerface.bvge.ecs.systems.physics.SpatialMapEX;
import com.controllerface.bvge.util.AssetPool;
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

    private final SpatialMapEX spatialMap = new SpatialMapEX();

    private int testBoxSize = 64;

    private void genNPCs(float spacing, float size)
    {
        System.out.println("generating: " + testBoxSize * testBoxSize + " NPCs..");
        var rand = new Random();
        for (int i = 0; i < testBoxSize; i++)
        {
            for (int j = 0; j < testBoxSize; j++)
            {
                float r = rand.nextFloat() / 5.0f;
                float g = rand.nextFloat() / 5.0f;
                float b = rand.nextFloat();

                float x = 100 + i * spacing;
                float y = 100 + j * spacing;

                var npc = ecs.registerEntity(null);
                var scomp2 = new SpriteComponent();
                var sprite2 = new Sprite();
                var tex2 = AssetPool.getTexture("assets/images/blendImage2.png");
                sprite2.setTexture(tex2);
                sprite2.setHeight(32);
                sprite2.setWidth(32);
                //scomp2.setSprite(sprite2);
                scomp2.setColor(new Vector4f(r,g,b,1));
                var physicsObject = PhysicsObjects.simpleBox(x, y, size, npc);
                ecs.attachComponent(npc, Component.SpriteComponent, scomp2);
                ecs.attachComponent(npc, Component.Transform, physicsObject.transform());
                ecs.attachComponent(npc, Component.RigidBody2D, physicsObject);
                ecs.attachComponent(npc, Component.BoundingBox, physicsObject.bounds());
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
        sprite.setHeight(16);
        sprite.setWidth(16);
        scomp.setSprite(sprite);
        //scomp.setColor(new Vector4f(0,0,0,1));
        var physicsObject = PhysicsObjects.polygon1(500,50, 32, player);
        ecs.attachComponent(player, Component.SpriteComponent, scomp);
        ecs.attachComponent(player, Component.Transform, physicsObject.transform());
        ecs.attachComponent(player, Component.ControlPoints, new ControlPoints());
        ecs.attachComponent(player, Component.RigidBody2D, physicsObject);
        ecs.attachComponent(player, Component.BoundingBox, physicsObject.bounds());

        genNPCs(3f, 3f);
    }

    @Override
    public void start()
    {
        //this.camera = new Camera(new Vector2f(0, 0));
    }

    @Override
    public void update(float dt)
    {
        //this.camera.adjustProjection();
    }

    @Override
    public void resizeSpatialMap(int width, int height) {
        spatialMap.resize(width, height);
    }

    public SpatialMapEX getSpatialMap() {
        return spatialMap;
    }
}
