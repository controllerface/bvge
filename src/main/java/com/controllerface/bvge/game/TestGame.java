package com.controllerface.bvge.game;

import com.controllerface.bvge.ecs.components.ArmatureIndex;
import com.controllerface.bvge.ecs.components.LinearForce;
import com.controllerface.bvge.ecs.systems.physics.PhysicsObjects;
import com.controllerface.bvge.ecs.ECS;
import com.controllerface.bvge.ecs.components.CameraFocus;
import com.controllerface.bvge.ecs.components.Component;
import com.controllerface.bvge.ecs.components.ControlPoints;
import com.controllerface.bvge.ecs.systems.CameraTracking;
import com.controllerface.bvge.ecs.systems.GameSystem;
import com.controllerface.bvge.ecs.systems.physics.UniformGrid;
import com.controllerface.bvge.ecs.systems.physics.PhysicsSimulation;
import com.controllerface.bvge.ecs.systems.renderers.*;
import com.controllerface.bvge.geometry.Meshes;
import com.controllerface.bvge.geometry.Models;

import static com.controllerface.bvge.ecs.systems.physics.PhysicsObjects.*;
import static com.controllerface.bvge.geometry.Models.*;


public class TestGame extends GameMode
{
    private final GameSystem screenBlankSystem;

    public TestGame(ECS ecs, GameSystem screenBlankSystem)
    {
        super(ecs);

        Meshes.init();
        Models.init();

        this.screenBlankSystem = screenBlankSystem;
    }

    private final UniformGrid uniformGrid = new UniformGrid();

    private void genCrates(int box_size, float spacing, float size, float start_x, float start_y)
    {
        System.out.println("generating: " + box_size * box_size + " Crates..");
        for (int i = 0; i < box_size; i++)
        {
            for (int j = 0; j < box_size; j++)
            {
                float x = start_x + i * spacing;
                float y = start_y + j * spacing;
                var npc = ecs.registerEntity(null);
                var armature_index = PhysicsObjects.dynamic_Box(x, y, size);
                ecs.attachComponent(npc, Component.Armature, new ArmatureIndex(armature_index));
            }
        }
    }

    private void genCrates2(int box_size, float spacing, float size, float start_x, float start_y)
    {
        System.out.println("generating: " + box_size * box_size + " Crates..");
        for (int i = 0; i < box_size; i++)
        {
            for (int j = 0; j < box_size; j++)
            {
                float x = start_x + i * spacing;
                float y = start_y + j * spacing;
                genBoxModelNPC(size, x, y);
            }
        }
    }

    private void genTriangles(int box_size, float spacing, float size, float start_x, float start_y)
    {
        System.out.println("generating: " + box_size * box_size + " Triangles..");
        for (int i = 0; i < box_size; i++)
        {
            for (int j = 0; j < box_size; j++)
            {
                float x = start_x + i * spacing;
                float y = start_y + j * spacing;
                var npc = ecs.registerEntity(null);
                var armature_index = PhysicsObjects.tri(x, y, size, 0);
                ecs.attachComponent(npc, Component.Armature, new ArmatureIndex(armature_index));
            }
        }
    }

    private void genCircles(int box_size, float spacing, float size, float start_x, float start_y)
    {
        System.out.println("generating: " + box_size * box_size + " Particles..");
        for (int i = 0; i < box_size; i++)
        {
            for (int j = 0; j < box_size; j++)
            {
                float x = start_x + i * spacing;
                float y = start_y + j * spacing;
                var npc = ecs.registerEntity(null);
                var armature_index = PhysicsObjects.particle(x, y, size);
                ecs.attachComponent(npc, Component.Armature, new ArmatureIndex(armature_index));
            }
        }
    }

    private void genFloor(int floor_size, float spacing, float size, float start_x, float start_y)
    {
        System.out.println("generating floor size: " + floor_size);
        for (int i = 0; i < floor_size; i++)
        {
            float x = start_x + i * spacing;
            float y = start_y;
            var npc = ecs.registerEntity(null);
            var armature_index = PhysicsObjects.static_box(x, y, size);
            ecs.attachComponent(npc, Component.Armature, new ArmatureIndex(armature_index));
        }
    }

    private void genWall(int floor_size, float spacing, float size, float start_x, float start_y)
    {
        System.out.println("generating wall size: " + floor_size);
        for (int i = 0; i < floor_size; i++)
        {
            float x = start_x;
            float y = start_y + i * spacing;
            var npc = ecs.registerEntity(null);
            var armature_index = PhysicsObjects.static_box(x, y, size);
            ecs.attachComponent(npc, Component.Armature, new ArmatureIndex(armature_index));
        }
    }

    private void genTestCircle(float size, float x, float y)
    {
        var npc = ecs.registerEntity(null);
        var armature_index = PhysicsObjects.particle(x, y, size);
        ecs.attachComponent(npc, Component.Armature, new ArmatureIndex(armature_index));
    }

    private void genTestCrate(float size, float x, float y)
    {
        var npc = ecs.registerEntity(null);
        var armature_index = PhysicsObjects.dynamic_Box(x, y, size);
        ecs.attachComponent(npc, Component.Armature, new ArmatureIndex(armature_index));
    }

    private void genTestTriangle(float size, float x, float y)
    {
        var npc = ecs.registerEntity(null);
        var armature_index = PhysicsObjects.tri(x, y, size, 0);
        ecs.attachComponent(npc, Component.Armature, new ArmatureIndex(armature_index));
    }

    private void genTestFigure(float size, float x, float y)
    {
        // circle entity
        var figure = ecs.registerEntity("player");

        var armature_index = PhysicsObjects.wrap_model(TEST_MODEL_INDEX, x, y, size, FLAG_NONE | FLAG_POLYGON);
        ecs.attachComponent(figure, Component.ControlPoints, new ControlPoints());
        ecs.attachComponent(figure, Component.CameraFocus, new CameraFocus());
        ecs.attachComponent(figure, Component.Armature, new ArmatureIndex(armature_index));
        ecs.attachComponent(figure, Component.LinearForce, new LinearForce(1500));
    }

    private void genTestFigureNPC(float size, float x, float y)
    {
        var figure = ecs.registerEntity(null);
        var armature_index = PhysicsObjects.wrap_model(TEST_MODEL_INDEX, x, y, size, FLAG_NONE | FLAG_POLYGON);
        ecs.attachComponent(figure, Component.Armature, new ArmatureIndex(armature_index));
    }

    private void genBoxModelNPC(float size, float x, float y)
    {
        var figure = ecs.registerEntity(null);
        var armature_index = PhysicsObjects.wrap_model(TEST_SQUARE_INDEX, x, y, size, FLAG_NONE | FLAG_POLYGON);
        ecs.attachComponent(figure, Component.Armature, new ArmatureIndex(armature_index));
    }


    private void genPlayer()
    {
        // player entity
        var player = ecs.registerEntity("player");
        var armature_index = PhysicsObjects.wrap_model(POLYGON1_MODEL,0,0, 32, FLAG_NONE | FLAG_POLYGON | FLAG_NO_BONES);
        ecs.attachComponent(player, Component.ControlPoints, new ControlPoints());
        ecs.attachComponent(player, Component.CameraFocus, new CameraFocus());
        ecs.attachComponent(player, Component.Armature, new ArmatureIndex(armature_index));
        ecs.attachComponent(player, Component.LinearForce, new LinearForce(1500));
    }

    // note: order of adding systems is important
    private void loadSystems()
    {
        // all physics calculations should be done first
        ecs.registerSystem(new PhysicsSimulation(ecs, uniformGrid));

        // camera movement must be handled before rendering occurs, but after objects are in position
        ecs.registerSystem(new CameraTracking(ecs, uniformGrid));

        // the blanking system clears the screen before rendering passes
        ecs.registerSystem(screenBlankSystem);

        // these are debug-level renderers for visualizing the modeled physics boundaries
        ecs.registerSystem(new EdgeRenderer(ecs));
        ecs.registerSystem(new CircleRenderer(ecs));
        //ecs.registerSystem(new BoundingBoxRenderer(ecs));
        ecs.registerSystem(new BoneRenderer(ecs));

        // main renderers go here, one for each model type that can be rendered
        ecs.registerSystem(new CrateRenderer(ecs));
    }

    @Override
    public void load()
    {
        //genPlayer();
        genTestFigure(1, 500, 0);

        //genTestTriangle(20f, 190, 250);

//        genTestTriangle(10f, 190, 250);
//        genTestTriangle(10f, 190, 200);
//        genTestTriangle(10f, 190, 150);
//        genTestTriangle(10f, 190, 100);
//        genTestTriangle(10f, 190, 50);

//        genTestCrate(10f, 190, 250);
//        genTestCrate(10f, 190, 200);
//        genTestCrate(10f, 190, 150);
//        genTestCrate(10f, 190, 100);
//        genTestCrate(10f, 190, 50);
//
//        genTestCrate(10f, 200, 250);
//        genTestCrate(10f, 200, 200);
//        genTestCrate(10f, 200, 150);
//        genTestCrate(10f, 200, 100);
//        genTestCrate(10f, 200, 50);
//
//        genTestCrate(10f, 210, 250);
//        genTestCrate(10f, 210, 200);
//        genTestCrate(10f, 210, 150);
//        genTestCrate(10f, 210, 100);
//        genTestCrate(10f, 210, 50);
//
//        genBoxModelNPC(0.05f, 200, 150);
//        genBoxModelNPC(0.05f, 200, 125);
//        genBoxModelNPC(0.05f, 200, 100);
//        genBoxModelNPC(0.05f, 200, 75);
//        genBoxModelNPC(0.05f, 200, 50);
        //genTestFigureNPC(1, 200, 50);
        //genTestCircle(20,0, 50);

        //genTestCircle(100,100, 100);
        //genTestCircle(20,0, 100);
        //genTestCircle(30,20, 55);



        //genCircles(100, 80f, 100f, 2100, 2100);

//        genNPCs(100, 10f, 10f, 2100, 2100);
//        genNPCs(100, 10f, 10f, 1000, -1000);
//        genNPCs(100, 10f, 10f, -1500, -1500);
       // genNPCs(100, 9f, 10f, 40, 500);

        //genNPCs(100, 7f, 10f, -1000, 0);
        //genNPCs(100, 7f, 10f, 0, -1000);
        //genNPCs(100, 7f, 10f, 0, 1000);



        genCircles(100, 1f, 2f, 0, 100);
        //genCrates(100,  6f, 5f, 100, 3000);
        //genCrates2(50,  6, 0.015f, 0, 2000);
        //genTriangles(100,  6f, 5f, 0, 1000);



//        genTriangles(50, 5f, 5f, 100, 1000);a
//        genTriangles(50, 5f, 5f, 200, 1100);
//        genTriangles(50, 5f, 5f, 300, 1200);
//        genTriangles(50, 5f, 5f, 400, 1300);
//        genTriangles(50, 5f, 5f, 500, 1400);
//        genTriangles(50, 5f, 5f, 600, 1500);
//        genTriangles(50, 5f, 5f, 700, 1600);
//        genTriangles(50, 5f, 5f, 800, 1700);
//        genTriangles(50, 5f, 5f, 900, 1800);
//        genTriangles(50, 5f, 5f, 0, 1900);


        //genNPCs(1, 41f, 40f, 100, 300);
        genFloor(10, 120f, 150f, -100, -100);
        genWall(5, 120f, 150f, -220, -100);
        genWall(5, 120f, 150f, 1100, -100);

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
        uniformGrid.resize(width, height);
    }
}
