package com.controllerface.bvge.game;

import com.controllerface.bvge.data.HullIndex;
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
import com.controllerface.bvge.util.Assets;
import org.joml.Random;
import org.joml.Vector4f;

import static com.controllerface.bvge.data.PhysicsObjects.*;
import static com.controllerface.bvge.geometry.Models.POLYGON1_MODEL;
import static com.controllerface.bvge.geometry.Models.TEST_MODEL_INDEX;


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
                var tex2 = Assets.texture("assets/images/blendImage2.png");
                sprite2.setTexture(tex2);
                sprite2.setHeight(32);
                sprite2.setWidth(32);
                scomp2.setSprite(sprite2);
                scomp2.setColor(new Vector4f(r,g,b,1));
                var hull_index = PhysicsObjects.dynamic_Box(x, y, size);
                ecs.attachComponent(npc, Component.SpriteComponent, scomp2);
                ecs.attachComponent(npc, Component.RigidBody2D, new HullIndex(hull_index));
            }
        }
    }

    private void genCircles(int box_size, float spacing, float size, float start_x, float start_y)
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
                var hull_index = PhysicsObjects.particle(x, y, size);
                ecs.attachComponent(npc, Component.RigidBody2D, new HullIndex(hull_index));
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
            var tex2 = Assets.texture("assets/images/blendImage2.png");
            sprite2.setTexture(tex2);
            sprite2.setHeight(32);
            sprite2.setWidth(32);
            //scomp2.setSprite(sprite2);
            scomp2.setColor(new Vector4f(r,g,b,1));
            var hull_index = PhysicsObjects.static_box(x, y, size);
            ecs.attachComponent(npc, Component.SpriteComponent, scomp2);
            ecs.attachComponent(npc, Component.RigidBody2D, new HullIndex(hull_index));
        }
    }

    private void genTestCircle(float size, float x, float y)
    {
        // circle entity
        var npc = ecs.registerEntity(null);
        var hull_index = PhysicsObjects.particle(x, y, size);
        ecs.attachComponent(npc, Component.RigidBody2D, new HullIndex(hull_index));
    }

    private void genTestFigure(float size, float x, float y)
    {
        // circle entity
        var npc = ecs.registerEntity(null);
        var hull_index = PhysicsObjects.wrap_model(TEST_MODEL_INDEX, x, y, size, FLAG_STATIC_OBJECT | FLAG_POLYGON);
        ecs.attachComponent(npc, Component.RigidBody2D, new HullIndex(hull_index));
    }


    private void genPlayer()
    {
        // player entity
        var player = ecs.registerEntity("player");

        //var hull_index = PhysicsObjects.polygon1(0,0, 32);

        var hull_index = PhysicsObjects.wrap_model(POLYGON1_MODEL,0,0, 32, FLAG_NONE | FLAG_POLYGON);


        ecs.attachComponent(player, Component.ControlPoints, new ControlPoints());
        ecs.attachComponent(player, Component.CameraFocus, new CameraFocus());
        ecs.attachComponent(player, Component.RigidBody2D, new HullIndex(hull_index));
        ecs.attachComponent(player, Component.LinearForce, new LinearForce(1500));
    }

    // note: order of adding systems is important
    private void loadSystems()
    {
        // all physics calculations should be done first
        ecs.registerSystem(new VerletPhysics(ecs, spatialPartition));

        // camera movement must be handled before rendering occurs, but after objects are in position
        ecs.registerSystem(new CameraTracking(ecs, spatialPartition));

        // blank screen before rendering, rendering passes happen after screen blanking
        ecs.registerSystem(screenBlankSystem);

        // these are debug-level renderers for visualizing the modeled physics boundaries
        ecs.registerSystem(new EdgeRenderer(ecs));
        ecs.registerSystem(new CircleRenderer(ecs));
        ecs.registerSystem(new BoundingBoxRenderer(ecs));


        // main renderers go here, one for each model type that can be rendered
        //ecs.registerSystem(new CrateRenderer(ecs));


        //ecs.registerSystem(new SpacePartitionRenderer(ecs, spatialPartition));
        //ecs.registerSystem(new SpriteRenderer(ecs));



    }

    @Override
    public void load()
    {
        genPlayer();
        genTestFigure(10, 500, 500);
        //genTestCircle(20,0, 50);
        //genTestCircle(100,100, 100);
        //genTestCircle(20,0, 100);
        genTestCircle(30,20, 55);

        //genCircles(100, 10f, 10f, 0, 2500);

        //genCircles(100, 80f, 100f, 2100, 2100);

//        genNPCs(100, 10f, 10f, 2100, 2100);
//        genNPCs(100, 10f, 10f, 1000, -1000);
//        genNPCs(100, 10f, 10f, -1500, -1500);
        //genNPCs(100, 9f, 10f, 40, 500);

        //genNPCs(100, 7f, 10f, -1000, 0);
        //genNPCs(100, 7f, 10f, 0, -1000);
        //genNPCs(100, 7f, 10f, 0, 1000);
        //genNPCs(100, 10f, 10f, 0, 500);

        genNPCs(1, 41f, 40f, 100, 300);
        genFloor(200, 150f, 150f, -4000, -100);
        //genFloor(50, 25f, 25f, -500, 150);
        //genFloor(50, 25f, 25f, -500, 1000);

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
