package com.controllerface.bvge.game;

import com.controllerface.bvge.ecs.ECS;
import com.controllerface.bvge.ecs.components.*;
import com.controllerface.bvge.ecs.systems.CameraTracking;
import com.controllerface.bvge.ecs.systems.GameSystem;
import com.controllerface.bvge.geometry.MeshRegistry;
import com.controllerface.bvge.geometry.ModelRegistry;
import com.controllerface.bvge.gl.renderers.*;
import com.controllerface.bvge.physics.PhysicsObjects;
import com.controllerface.bvge.physics.PhysicsSimulation;
import com.controllerface.bvge.physics.UniformGrid;
import com.controllerface.bvge.substances.Liquid;
import com.controllerface.bvge.substances.Solid;
import com.controllerface.bvge.window.Window;

import java.util.EnumSet;
import java.util.Random;

import static com.controllerface.bvge.geometry.ModelRegistry.*;
import static com.controllerface.bvge.util.Constants.*;

public class TestGame extends GameMode
{
    private final GameSystem screenBlankSystem;

    private enum RenderType
    {
        MODELS,     // normal objects
        HULLS,      // physics hulls
        BOUNDS,     // bounding boxes
        POINTS,     // model vertices
        ARMATURES,  // armature roots
        GRID,       // uniform grid

    }
    private static final EnumSet<RenderType> ACTIVE_RENDERERS =
        EnumSet.of(
            RenderType.HULLS,
//            RenderType.POINTS,
//            RenderType.ARMATURES,
//            RenderType.BOUNDS,
//            RenderType.GRID,
            RenderType.MODELS);

//    private static final EnumSet<RenderType> ACTIVE_RENDERERS =
//        EnumSet.allOf(RenderType.class);

    private final UniformGrid uniformGrid = new UniformGrid(Window.get().width(), Window.get().height());

    private final Random random = new Random();

    public TestGame(ECS ecs, GameSystem screenBlankSystem)
    {
        super(ecs);

        MeshRegistry.init();
        ModelRegistry.init();

        this.screenBlankSystem = screenBlankSystem;
    }


    public float rando_float(float baseNumber, float percentage)
    {

        float upperBound = baseNumber * percentage;
        return baseNumber + random.nextFloat() * (upperBound - baseNumber);
    }

    public int rando_int(int min, int max)
    {
        return random.nextInt(min, max);
    }


//    private void genBlocks(int box_size, float spacing, float size, float start_x, float start_y, Solid block_mineral)
//    {
//        System.out.println("generating: " + box_size * box_size + " Blocks..");
//        for (int i = 0; i < box_size; i++)
//        {
//            for (int j = 0; j < box_size; j++)
//            {
//                float x = start_x + i * spacing;
//                float y = start_y + j * spacing;
//                //var npc = ecs.registerEntity(null);
//                var armature_index = PhysicsObjects.dynamic_block(x, y, size, 500f, 0.05f, 0.0003f, block_mineral);
//                //ecs.attachComponent(npc, Component.Armature, new ArmatureIndex(armature_index));
//            }
//        }
//    }


    private void genBlocks(int box_size, float spacing, float size, float start_x, float start_y, Solid ... minerals)
    {
        System.out.println("generating: " + box_size * box_size + " Blocks..");
        for (int i = 0; i < box_size; i++)
        {
            for (int j = 0; j < box_size; j++)
            {
                float x = start_x + i * spacing;
                float y = start_y + j * spacing;
                int rx = rando_int(0, minerals.length);
                PhysicsObjects.dynamic_block(x, y, size, 90f, 0.03f, 0.0003f, minerals[rx]);
            }
        }
    }

    private void genSquaresRando(int box_size, float spacing, float size, float percentage, float start_x, float start_y, Solid ... minerals)
    {
        System.out.println("generating: " + box_size * box_size + " Blocks..");
        for (int i = 0; i < box_size; i++)
        {
            for (int j = 0; j < box_size; j++)
            {
                float x = start_x + i * spacing;
                float y = start_y + j * spacing;
                int rx = rando_int(0, minerals.length);
                PhysicsObjects.dynamic_block(x, y, rando_float(size, percentage), 90f, 0.03f, 0.0003f, minerals[rx]);
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
                var armature_index = PhysicsObjects.tri(x, y, size, 0, 1f, 0.02f, 0.0003f);
                //ecs.attachComponent(npc, Component.Armature, new ArmatureIndex(armature_index));
            }
        }
    }

    private void genWater(int box_size, float spacing, float size, float start_x, float start_y, Liquid ... liquids)
    {
        boolean flip = false;
        System.out.println("generating: " + box_size * box_size + " water particles..");
        for (int i = 0; i < box_size; i++)
        {
            for (int j = 0; j < box_size; j++)
            {
                float x = start_x + i * spacing;
                float y = start_y + j * spacing;
                int flags = flip
                    ? PointFlags.FLOW_LEFT.bits
                    : 0;
                flip = !flip;
                int rx = rando_int(0, liquids.length);
                var armature_index = PhysicsObjects.liquid_particle(x, y, size,
                    .1f, 0.0f, 0.00000f,
                    HullFlags.IS_LIQUID._int,
                    flags,
                    liquids[rx]);
                //ecs.attachComponent(npc, Component.Armature, new ArmatureIndex(armature_index));
            }
        }
    }

    private void genFloor(int floor_size, float spacing, float size, float start_x, float start_y, float friction, Solid solid)
    {
        System.out.println("generating floor size: " + floor_size);
        for (int i = 0; i < floor_size; i++)
        {
            float x = start_x + i * spacing;
            float y = start_y;
            //var npc = ecs.registerEntity(null);
            var armature_index = PhysicsObjects.static_box(x, y, size, 0, friction, 0.0003f, solid);
            //ecs.attachComponent(npc, Component.Armature, new ArmatureIndex(armature_index));
        }
    }

    private void genWall(int floor_size, float spacing, float size, float start_x, float start_y, Solid solid)
    {
        System.out.println("generating wall size: " + floor_size);
        for (int i = 0; i < floor_size; i++)
        {
            float x = start_x;
            float y = start_y + i * spacing;
            //var npc = ecs.registerEntity(null);
            var armature_index = PhysicsObjects.static_box(x, y, size, 0, 0.0f, 0.0f, solid);
            //ecs.attachComponent(npc, Component.Armature, new ArmatureIndex(armature_index));
        }
    }

    private void genTestCrate(float size, float x, float y)
    {
        //var npc = ecs.registerEntity(null);
        var armature_index = PhysicsObjects.dynamic_block(x, y, size, .1f, 0.02f, 0.0001f, Solid.ANDESITE);
        //ecs.attachComponent(npc, Component.Armature, new ArmatureIndex(armature_index));
    }

    private void genTestTriangle(float size, float x, float y)
    {
       // var npc = ecs.registerEntity(null);
        var armature_index = PhysicsObjects.tri(x, y, size, 0, .1f, 0.02f, 0.0003f);
        //ecs.attachComponent(npc, Component.Armature, new ArmatureIndex(armature_index));
    }

    private void genTestFigure(float size, float x, float y)
    {
        // circle entity
        var figure = ecs.registerEntity("player");

        var armature_index = PhysicsObjects.wrap_model(TEST_MODEL_INDEX, x, y, size, HullFlags.IS_POLYGON._int, 100.5f, 0.05f, 0,0);
        ecs.attachComponent(figure, Component.ControlPoints, new ControlPoints());
        ecs.attachComponent(figure, Component.CameraFocus, new CameraFocus());
        // todo: determine if a different ID may be used for identifying entities that is not tied to the
        //  armature index directly. Now that objects can be deleted, this value can change frequently
        //  and there is not a mechanism to keep ECS entities updated to compensate. Instead, some unique
        //  monotonically increasing value could be used, which doesn't change during entity life time
        ecs.attachComponent(figure, Component.Armature, new ArmatureIndex(armature_index));
        ecs.attachComponent(figure, Component.LinearForce, new LinearForce(1600));
    }

    private void genCursor(float size, float x, float y)
    {
        // circle entity
        var figure = ecs.registerEntity("mouse");
        var armature_index = PhysicsObjects.circle_cursor(x, y, size);
        ecs.attachComponent(figure, Component.Armature, new ArmatureIndex(armature_index));
    }

    private void genTestFigureNPC_2(float size, float x, float y)
    {
        //var figure = ecs.registerEntity(null);
        var armature_index = PhysicsObjects.wrap_model(TEST_MODEL_INDEX_2, x, y, size, HullFlags.IS_POLYGON._int, 50, 0.02f, 0, 0);
        //ecs.attachComponent(figure, Component.Armature, new ArmatureIndex(armature_index));
    }

    private void genTestFigureNPC(float size, float x, float y)
    {
        //var figure = ecs.registerEntity(null);
        var armature_index = PhysicsObjects.wrap_model(TEST_MODEL_INDEX, x, y, size, HullFlags.IS_POLYGON._int, 50, 0.02f, 0,0);
        //ecs.attachComponent(figure, Component.Armature, new ArmatureIndex(armature_index));
    }

    private void genBoxModelNPC(float size, float x, float y)
    {
        //var figure = ecs.registerEntity(null);
        var armature_index = PhysicsObjects.wrap_model(TEST_SQUARE_INDEX, x, y, size, HullFlags.IS_POLYGON._int, .1f, 0.02f, 0,0);
        //ecs.attachComponent(figure, Component.Armature, new ArmatureIndex(armature_index));
    }

    // note: order of adding systems is important
    private void loadSystems()
    {
        // all physics calculations should be done first
        ecs.registerSystem(new PhysicsSimulation(ecs, uniformGrid));

        // camera movement must be handled before rendering occurs, but after collision has been resolved
        ecs.registerSystem(new CameraTracking(ecs, uniformGrid));

        // the blanking system clears the screen before rendering passes
        ecs.registerSystem(screenBlankSystem);

        // main renderers go here, one for each model type that can be rendered

        ecs.registerSystem(new BackgroundRenderer(ecs));

        if (ACTIVE_RENDERERS.contains(RenderType.MODELS))
        {
            //ecs.registerSystem(new CrateRenderer(ecs));
            ecs.registerSystem(new ModelRenderer(ecs, "block_model.glsl", TEST_MODEL_INDEX));
            ecs.registerSystem(new ModelRenderer(ecs, "block_model.glsl", BASE_BLOCK_INDEX));
            ecs.registerSystem(new ModelRenderer(ecs, "block_model.glsl", BASE_TRI_INDEX));
            ecs.registerSystem(new LiquidRenderer(ecs));
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

        if (ACTIVE_RENDERERS.contains(RenderType.GRID))
        {
            ecs.registerSystem(new UniformGridRenderer(ecs, uniformGrid));
        }

        if (ACTIVE_RENDERERS.contains(RenderType.ARMATURES))
        {
            ecs.registerSystem(new ArmatureRenderer(ecs));
        }

        ecs.registerSystem(new MouseRenderer(ecs));
    }

    @Override
    public void load()
    {
        // player character
        genTestFigure(1f, 2000, 3200);
        genCursor(20, 2000, 3200);
        //genTestFigureNPC_2(1f, 100, 500);

//        genTestFigureNPC(1f, 200, 0);
//        genSquares(1,  25f, 25f, 420, 200);
//        genTestFigureNPC(1f, 200, 100);
//        genSquares(1,  25f, 25f, 420, 200);
//        genTestFigureNPC(1f, 200, 250);
//        genSquares(1,  25f, 25f, 420, 200);
//        genTestFigureNPC(1f, 100, 50);

        //genCircles(150, 6f, 5f, 0, 100);

        genWater(100, 15f, 15f, 0, 3000, Liquid.WATER);
        genSquaresRando(40,  32f, 32f, 0.8f,-50, 200, Solid.CLAYSTONE, Solid.SOAPSTONE, Solid.MUDSTONE);
        genBlocks(40,  32f, 32f, 2500, 200, Solid.GREENSCHIST, Solid.SCHIST, Solid.BLUESCHIST, Solid.WHITESCHIST);
        //genBlocks(40,  32f, 32f, 2500, 3800, Solid.QUARTZITE, Solid.QUARTZ_DIORITE, Solid.QUARTZ_MONZONITE);

        //genSquaresRando(50,  25f, 25f, 0.8f, 2500, 200);
        //genSquares(1,  25f, 25f, 420, 200);

        //genCrates2(20, 5f, 0.025f, 100, 100);
        //genTriangles(130,  6f, 5f, -120, 200);
        genTriangles(50,  32f, 32f, 2500, 3800);

        //PhysicsObjects.static_tri(0,-25, 150, 1, 0.02f);
        //PhysicsObjects.static_box(0,0,10,10, 0f);

        genFloor(16, 150f, 150f, -70, -100, 0.5f, Solid.ANDESITE);
        genFloor(32, 150f, 150f, 1700, -100, 0.5f, Solid.ANDESITE);
        genFloor(32, 150f, 150f, 1700, 2200, 0.5f, Solid.DIORITE);

        genWall(15, 150f, 150f, -220, -100, Solid.ANDESITE);
        genWall(5, 150f, 150f, 2000, 1500, Solid.DIORITE);
        genWall(5, 150f, 150f, 4880, -100, Solid.ANDESITE);

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
