package com.controllerface.bvge.game;

import com.controllerface.bvge.data.BodyIndex;
import com.controllerface.bvge.data.LinearForce;
import com.controllerface.bvge.data.PhysicsObjects;
import com.controllerface.bvge.ecs.ECS;
import com.controllerface.bvge.ecs.Sprite;
import com.controllerface.bvge.ecs.components.*;
import com.controllerface.bvge.ecs.systems.CameraTracking;
import com.controllerface.bvge.ecs.systems.GameSystem;
import com.controllerface.bvge.ecs.systems.physics.SpatialPartition;
import com.controllerface.bvge.ecs.systems.physics.VerletPhysics;
import com.controllerface.bvge.ecs.systems.renderers.*;
import com.controllerface.bvge.util.AssetPool;
import org.joml.Random;
import org.joml.Vector4f;

public class TestGame extends GameMode
{
    private final GameSystem screenBlankSystem;
    public TestGame(ECS ecs, GameSystem screenBlankSystem)
    {
        super(ecs);
        this.screenBlankSystem = screenBlankSystem;
    }

    private final SpatialPartition spatialPartition = new SpatialPartition();

    private void genNPCs(int box_size, float spacing, float size, float start_x, float start_y)
    {
        // trivial change for new commit
        System.out.println("generating: " + box_size * box_size + " NPCs..");
        var rand = new Random();
        for (int i = 0; i < box_size; i++)
        {
            for (int j = 0; j < box_size; j++)
            {
                float r = rand.nextFloat() / 5.0f;
                float g = rand.nextFloat() / 5.0f;
                float b = rand.nextFloat();

                float x = start_x + i * spacing;
                float y = start_y + j * spacing;

                var npc = ecs.registerEntity(null);
                var scomp2 = new SpriteComponent();
                var sprite2 = new Sprite();
                var tex2 = AssetPool.getTexture("assets/images/blendImage2.png");
                sprite2.setTexture(tex2);
                sprite2.setHeight(32);
                sprite2.setWidth(32);
                scomp2.setSprite(sprite2);
                scomp2.setColor(new Vector4f(r,g,b,1));
                var body_index = PhysicsObjects.dynamic_Box(x, y, size);
                ecs.attachComponent(npc, Component.SpriteComponent, scomp2);
                ecs.attachComponent(npc, Component.RigidBody2D, new BodyIndex(body_index));
            }
        }
    }

    private void genFloor(int floor_size, float spacing, float size, float start_x, float start_y)
    {
        System.out.println("generating floor size: " + floor_size);
        var rand = new Random();
        for (int i = 0; i < floor_size; i++)
        {
            float r = rand.nextFloat() / 5.0f;
            float g = rand.nextFloat() / 5.0f;
            float b = rand.nextFloat();
            float x = start_x + i * spacing;
            float y = start_y;

            var npc = ecs.registerEntity(null);
            var scomp2 = new SpriteComponent();
            var sprite2 = new Sprite();
            var tex2 = AssetPool.getTexture("assets/images/blendImage2.png");
            sprite2.setTexture(tex2);
            sprite2.setHeight(32);
            sprite2.setWidth(32);
            //scomp2.setSprite(sprite2);
            scomp2.setColor(new Vector4f(r,g,b,1));
            var body_index = PhysicsObjects.static_box(x, y, size);
            ecs.attachComponent(npc, Component.SpriteComponent, scomp2);
            ecs.attachComponent(npc, Component.RigidBody2D, new BodyIndex(body_index));
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

        // todo: instead of a body, just a reference/index needs to be stored
        var body_index = PhysicsObjects.polygon1(600,100, 32);
        ecs.attachComponent(player, Component.ControlPoints, new ControlPoints());
        ecs.attachComponent(player, Component.CameraFocus, new CameraFocus());
        ecs.attachComponent(player, Component.SpriteComponent, scomp);
        ecs.attachComponent(player, Component.RigidBody2D, new BodyIndex(body_index));
        ecs.attachComponent(player, Component.LinearForce, new LinearForce(1500));
    }

    // note: order of adding systems is important
    private void loadSystems()
    {
        // do physics first
        ecs.registerSystem(new VerletPhysics(ecs, spatialPartition));

        // camera movement must be handled before rendering occurs
        ecs.registerSystem(new CameraTracking(ecs, spatialPartition));

        // rendering passes happen at the end
        // todo: if things are moved to a unified vbo, there could be a VBO prep step here before
        //  the screen blank so the render time is a fast as possible, and the time from blank to
        //  render is minimized.
        // blank screen before rendering
        ecs.registerSystem(screenBlankSystem);
        ecs.registerSystem(new EdgeRenderer(ecs));
        ecs.registerSystem(new BoxRenderer(ecs));

        //ecs.registerSystem(new SpacePartitionRenderer(ecs, spatialPartition));
        //ecs.registerSystem(new SpriteRenderer(ecs));
        //ecs.registerSystem(new BoundingBoxRenderer(ecs, spatialPartition));


    }

    @Override
    public void load()
    {
        genPlayer();

//        genNPCs(100, 10f, 10f, 2100, 2100);
//        genNPCs(100, 10f, 10f, 1000, -1000);
//        genNPCs(100, 10f, 10f, -1500, -1500);
//        genNPCs(100, 10f, 10f, 40, 2500);

        //genNPCs(100, 10f, 10f, 2100, 2100);
        //genNPCs(100, 10f, 10f, 1000, -1000);
        genNPCs(100, 10f, 10f, -1500, -1500);
        genNPCs(100, 10f, 10f, 40, 500);

        genNPCs(3, 20f, 15f, 100, 300);
        genFloor(500, 5f, 5f, -500, -40);

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
        // todo: buffer resize operations and then apply ONLY after a frame is done rendering
        spatialPartition.resize(width, height);
    }
}
