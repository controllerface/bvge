package com.controllerface.bvge.game;

import com.controllerface.bvge.data.ArmatureIndex;
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

    private void genTestCircle(float size, float x, float y)
    {
        // circle entity
        var npc = ecs.registerEntity(null);
        var armature_index = PhysicsObjects.particle(x, y, size);
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
        // circle entity
        var figure = ecs.registerEntity(null);
        var armature_index = PhysicsObjects.wrap_model(TEST_MODEL_INDEX, x, y, size, FLAG_NONE | FLAG_POLYGON);
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
        // todo: write bone animator system, it should put all bones in their current positions

        // all physics calculations should be done first
        ecs.registerSystem(new VerletPhysics(ecs, spatialPartition));

        // camera movement must be handled before rendering occurs, but after objects are in position
        ecs.registerSystem(new CameraTracking(ecs, spatialPartition));

        // the blanking system clears the screen before rendering passes
        ecs.registerSystem(screenBlankSystem);

        // these are debug-level renderers for visualizing the modeled physics boundaries
        ecs.registerSystem(new EdgeRenderer(ecs));
        ecs.registerSystem(new CircleRenderer(ecs));
        //ecs.registerSystem(new BoundingBoxRenderer(ecs));
        ecs.registerSystem(new BoneRenderer(ecs));

        // main renderers go here, one for each model type that can be rendered
        //ecs.registerSystem(new CrateRenderer(ecs));
    }

    @Override
    public void load()
    {
        //genPlayer();
        genTestFigure(1, 100, 0);

        genTestFigureNPC(1, 200, 50);
        //genTestCircle(20,0, 50);
        //genTestCircle(100,100, 100);
        //genTestCircle(20,0, 100);
        //genTestCircle(30,20, 55);

        genCircles(100, 10f, 10f, 0, 2500);

        //genCircles(100, 80f, 100f, 2100, 2100);

//        genNPCs(100, 10f, 10f, 2100, 2100);
//        genNPCs(100, 10f, 10f, 1000, -1000);
//        genNPCs(100, 10f, 10f, -1500, -1500);
       // genNPCs(100, 9f, 10f, 40, 500);

        //genNPCs(100, 7f, 10f, -1000, 0);
        //genNPCs(100, 7f, 10f, 0, -1000);
        //genNPCs(100, 7f, 10f, 0, 1000);
        genNPCs(100, 10f, 10f, 0, 500);

        //genNPCs(1, 41f, 40f, 100, 300);
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
