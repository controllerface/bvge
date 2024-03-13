package com.controllerface.bvge.game;

import com.controllerface.bvge.ecs.ECS;
import com.controllerface.bvge.ecs.components.*;
import com.controllerface.bvge.ecs.systems.CameraTracking;
import com.controllerface.bvge.ecs.systems.GameSystem;
import com.controllerface.bvge.geometry.Meshes;
import com.controllerface.bvge.geometry.Models;
import com.controllerface.bvge.gl.renderers.*;
import com.controllerface.bvge.physics.PhysicsObjects;
import com.controllerface.bvge.physics.PhysicsSimulation;
import com.controllerface.bvge.physics.UniformGrid;

import java.util.EnumSet;

import static com.controllerface.bvge.geometry.Models.*;
import static com.controllerface.bvge.physics.PhysicsObjects.*;


public class TestGame extends GameMode
{
    private final GameSystem screenBlankSystem;

    private enum RenderType
    {
        MODELS, // normal objects
        HULLS,   // physics hulls
        BOUNDS,  // bounding boxes
        POINTS,  // model vertices
    }

    private static final EnumSet<RenderType> ACTIVE_RENDERERS =
        EnumSet.of(
            RenderType.MODELS,
            RenderType.HULLS);

//    private static final EnumSet<RenderType> ACTIVE_RENDERERS =
//        EnumSet.allOf(RenderType.class);

    private final UniformGrid uniformGrid = new UniformGrid();

    public TestGame(ECS ecs, GameSystem screenBlankSystem)
    {
        super(ecs);

        Meshes.init();
        Models.init();

        this.screenBlankSystem = screenBlankSystem;
    }

    private void genSquares(int box_size, float spacing, float size, float start_x, float start_y)
    {
        System.out.println("generating: " + box_size * box_size + " Crates..");
        for (int i = 0; i < box_size; i++)
        {
            for (int j = 0; j < box_size; j++)
            {
                float x = start_x + i * spacing;
                float y = start_y + j * spacing;
                //var npc = ecs.registerEntity(null);
                var armature_index = PhysicsObjects.dynamic_Box(x, y, size, .1f);
                //ecs.attachComponent(npc, Component.Armature, new ArmatureIndex(armature_index));
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
                //var npc = ecs.registerEntity(null);
                var armature_index = PhysicsObjects.tri(x, y, size, 0, .1f);
                //ecs.attachComponent(npc, Component.Armature, new ArmatureIndex(armature_index));
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
                //var npc = ecs.registerEntity(null);
                var armature_index = PhysicsObjects.particle(x, y, size, 1f);
                //ecs.attachComponent(npc, Component.Armature, new ArmatureIndex(armature_index));
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
            //var npc = ecs.registerEntity(null);
            var armature_index = PhysicsObjects.static_box(x, y, size, 0);
            //ecs.attachComponent(npc, Component.Armature, new ArmatureIndex(armature_index));
        }
    }

    private void genWall(int floor_size, float spacing, float size, float start_x, float start_y)
    {
        System.out.println("generating wall size: " + floor_size);
        for (int i = 0; i < floor_size; i++)
        {
            float x = start_x;
            float y = start_y + i * spacing;
            //var npc = ecs.registerEntity(null);
            var armature_index = PhysicsObjects.static_box(x, y, size, 0);
            //ecs.attachComponent(npc, Component.Armature, new ArmatureIndex(armature_index));
        }
    }

    private void genTestCircle(float size, float x, float y)
    {
        //var npc = ecs.registerEntity(null);
        var armature_index = PhysicsObjects.particle(x, y, size, .1f);
        //ecs.attachComponent(npc, Component.Armature, new ArmatureIndex(armature_index));
    }

    private void genTestCrate(float size, float x, float y)
    {
        //var npc = ecs.registerEntity(null);
        var armature_index = PhysicsObjects.dynamic_Box(x, y, size, .1f);
        //ecs.attachComponent(npc, Component.Armature, new ArmatureIndex(armature_index));
    }

    private void genTestTriangle(float size, float x, float y)
    {
       // var npc = ecs.registerEntity(null);
        var armature_index = PhysicsObjects.tri(x, y, size, 0, .1f);
        //ecs.attachComponent(npc, Component.Armature, new ArmatureIndex(armature_index));
    }

    private void genTestFigure(float size, float x, float y)
    {
        // circle entity
        var figure = ecs.registerEntity("player");

        var armature_index = PhysicsObjects.wrap_model(TEST_MODEL_INDEX, x, y, size, FLAG_NONE | FLAG_POLYGON, 100.5f);
        ecs.attachComponent(figure, Component.ControlPoints, new ControlPoints());
        ecs.attachComponent(figure, Component.CameraFocus, new CameraFocus());
        // todo: determine if a different ID may be used for identifying entities that is not tied to the
        //  armature index directly. Now that objects can be deleted, this value can change frequently
        //  and there is not a mechanism to keep ECS entities updated to compensate. Instead, some unique
        //  monotonically increasing value could be used, which doesn't change during entity life time
        ecs.attachComponent(figure, Component.Armature, new ArmatureIndex(armature_index));
        ecs.attachComponent(figure, Component.LinearForce, new LinearForce(1500));
    }

    private void genTestFigureNPC(float size, float x, float y)
    {
        //var figure = ecs.registerEntity(null);
        var armature_index = PhysicsObjects.wrap_model(TEST_MODEL_INDEX, x, y, size, FLAG_NONE | FLAG_POLYGON, 50);
        //ecs.attachComponent(figure, Component.Armature, new ArmatureIndex(armature_index));
    }

    private void genBoxModelNPC(float size, float x, float y)
    {
        //var figure = ecs.registerEntity(null);
        var armature_index = PhysicsObjects.wrap_model(TEST_SQUARE_INDEX, x, y, size, FLAG_NONE | FLAG_POLYGON, .1f);
        //ecs.attachComponent(figure, Component.Armature, new ArmatureIndex(armature_index));
    }


    private void genPlayer()
    {
        // player entity
        var player = ecs.registerEntity("player");
        var armature_index = PhysicsObjects.wrap_model(POLYGON1_MODEL,0,0, 32, FLAG_NONE | FLAG_POLYGON | FLAG_NO_BONES, 1);
        ecs.attachComponent(player, Component.ControlPoints, new ControlPoints());
        ecs.attachComponent(player, Component.CameraFocus, new CameraFocus());
        ecs.attachComponent(player, Component.Armature, new ArmatureIndex(armature_index));
        ecs.attachComponent(player, Component.LinearForce, new LinearForce(1500));
    }

    // note: order of adding systems is important
    private void loadSystems()
    {
        // all physics calculations should be done immediately after animation
        ecs.registerSystem(new PhysicsSimulation(ecs, uniformGrid));

        // camera movement must be handled before rendering occurs, but after collision has been resolved
        ecs.registerSystem(new CameraTracking(ecs, uniformGrid));

        // the blanking system clears the screen before rendering passes
        ecs.registerSystem(screenBlankSystem);

        // main renderers go here, one for each model type that can be rendered

        if (ACTIVE_RENDERERS.contains(RenderType.MODELS))
        {
            ecs.registerSystem(new CrateRenderer(ecs));
            ecs.registerSystem(new HumanoidRenderer(ecs));
        }

        // these are debug-level renderers for visualizing the modeled physics boundaries

        if (ACTIVE_RENDERERS.contains(RenderType.HULLS))
        {
            ecs.registerSystem(new EdgeRenderer(ecs));
            ecs.registerSystem(new CircleRenderer(ecs));
        }

        if (ACTIVE_RENDERERS.contains(RenderType.BOUNDS))
        {
            ecs.registerSystem(new BoundingBoxRenderer(ecs));
        }

        if (ACTIVE_RENDERERS.contains(RenderType.POINTS))
        {
            ecs.registerSystem(new PointRenderer(ecs));
        }

        ecs.registerSystem(new UniformGridRenderer(ecs, uniformGrid));
    }

    @Override
    public void load()
    {
        // player character
        genTestFigure(1f, 300, 0);

//        genTestFigureNPC(1f, 200, 0);
//        genTestFigureNPC(1f, 200, 100);
//        genTestFigureNPC(1f, 200, 250);
//        genTestFigureNPC(1f, 100, 50);

        genCircles(30, 6f, 5f, 0, 100);
        genSquares(30,  6f, 5f, -120, 200);
        genCrates2(30, 7f, 0.025f, 200, 100);
        genTriangles(30,  6f, 5f, 120, 100);

        genFloor(8, 150f, 150f, -70, -100);
        genWall(5, 150f, 150f, -220, -100);
        genWall(5, 150f, 150f, 1130, -100);

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
}
