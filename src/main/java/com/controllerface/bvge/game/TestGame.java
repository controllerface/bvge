package com.controllerface.bvge.game;

import com.controllerface.bvge.data.PhysicsObjects;
import com.controllerface.bvge.ecs.ECS;
import com.controllerface.bvge.ecs.Sprite;
import com.controllerface.bvge.ecs.components.*;
import com.controllerface.bvge.ecs.systems.CameraTracking;
import com.controllerface.bvge.ecs.systems.physics.SpatialMap;
import com.controllerface.bvge.ecs.systems.physics.VerletPhysics;
import com.controllerface.bvge.ecs.systems.renderers.BoundingBoxRenderer;
import com.controllerface.bvge.ecs.systems.renderers.LineRenderer;
import com.controllerface.bvge.ecs.systems.renderers.SpacePartitionRenderer;
import com.controllerface.bvge.ecs.systems.renderers.SpriteRenderer;
import com.controllerface.bvge.util.AssetPool;
import org.joml.Random;
import org.joml.Vector4f;

public class TestGame extends GameMode
{
    private final ECS ecs;

    public TestGame(ECS ecs)
    {
        this.ecs = ecs;
    }

    private final SpatialMap spatialMap = new SpatialMap();

    private void genNPCs(float spacing, float size)
    {
        int testBoxSize = 64;

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

    private void genPlayer()
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
        var physicsObject = PhysicsObjects.polygon1(0,0, 32, player);
        ecs.attachComponent(player, Component.ControlPoints, new ControlPoints());
        ecs.attachComponent(player, Component.CameraFocus, new CameraFocus());
        ecs.attachComponent(player, Component.SpriteComponent, scomp);
        ecs.attachComponent(player, Component.Transform, physicsObject.transform());
        ecs.attachComponent(player, Component.RigidBody2D, physicsObject);
        ecs.attachComponent(player, Component.BoundingBox, physicsObject.bounds());
    }

    // note: order of adding systems is important
    private void loadSystems()
    {
        ecs.registerSystem(new VerletPhysics(ecs, spatialMap));
        ecs.registerSystem(new CameraTracking(ecs));
        ecs.registerSystem(new SpacePartitionRenderer(ecs, spatialMap));
        //ecs.registerSystem(new SpriteRenderer(ecs));
        ecs.registerSystem(new LineRenderer(ecs));
        //ecs.registerSystem(new BoundingBoxRenderer(ecs));
    }

    @Override
    public void load()
    {
        genPlayer();
        genNPCs(3f, 3f);
        loadSystems();
    }

    @Override
    public void start()
    {
    }

    @Override
    public void update(float dt)
    {
    }

    @Override
    public void resizeSpatialMap(int width, int height)
    {
        spatialMap.resize(width, height);
    }
}
