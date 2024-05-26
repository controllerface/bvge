package com.controllerface.bvge.game;

import com.controllerface.bvge.cl.GPGPU;
import com.controllerface.bvge.ecs.ECS;
import com.controllerface.bvge.ecs.components.*;
import com.controllerface.bvge.ecs.systems.CameraTracking;
import com.controllerface.bvge.ecs.systems.GameSystem;
import com.controllerface.bvge.geometry.MeshRegistry;
import com.controllerface.bvge.geometry.ModelRegistry;
import com.controllerface.bvge.gl.renderers.*;
import com.controllerface.bvge.physics.PhysicsEntityBatch;
import com.controllerface.bvge.physics.PhysicsObjects;
import com.controllerface.bvge.physics.PhysicsSimulation;
import com.controllerface.bvge.physics.UniformGrid;
import com.controllerface.bvge.substances.Liquid;
import com.controllerface.bvge.substances.Solid;
import com.controllerface.bvge.util.FastNoiseLite;
import com.controllerface.bvge.window.Window;

import java.util.*;

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
        ENTITIES,   // entity roots
        GRID,       // uniform grid

    }
    private static final EnumSet<RenderType> ACTIVE_RENDERERS =
        EnumSet.of(
//            RenderType.HULLS,
//            RenderType.POINTS,
//            RenderType.ENTITIES,
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

    private void genBlocks(int box_size, float spacing, float size, float start_x, float start_y, Solid ... minerals)
    {
        //System.out.println("generating: " + box_size * box_size + " Blocks..");
        for (int i = 0; i < box_size; i++)
        {
            for (int j = 0; j < box_size; j++)
            {
                float x = start_x + i * spacing;
                float y = start_y + j * spacing;
                int rx = rando_int(0, minerals.length);
                PhysicsObjects.dynamic_block(x, y, size, 90f, 0.03f, 0.0003f, 0, minerals[rx]);
            }
        }
    }

    private void genNoiseBlocks(boolean dynamic, int box_size, float spacing, float size, float start_x, float start_y, Solid ... minerals)
    {
        //System.out.println("generating: " + box_size * box_size + " Blocks..");
        for (int i = 0; i < box_size; i++)
        {
            for (int j = 0; j < box_size; j++)
            {
                float x = start_x + i * spacing;
                float y = start_y + j * spacing;
                int rx = rando_int(0, minerals.length);
                if (dynamic) PhysicsObjects.dynamic_block(x, y, size, 90f, 0.03f, 0.0003f, 0, minerals[rx]);
                else PhysicsObjects.static_box(x, y, size, 90f, 0.03f, 0.0003f, 0, minerals[rx]);
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
                PhysicsObjects.dynamic_block(x, y, rando_float(size, percentage), 90f, 0.03f, 0.0003f, 0, minerals[rx]);
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
                PhysicsObjects.tri(x, y, size, 0, 20f, 0.02f, 0.0003f);
            }
        }
    }

    private void genWater(int box_size, float spacing, float size, float start_x, float start_y, Liquid ... liquids)
    {
        boolean flip = false;
        //System.out.println("generating: " + box_size * box_size + " water particles..");
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
                PhysicsObjects.liquid_particle(x, y, size,
                    .1f, 0.0f, 0.00000f,
                    HullFlags.IS_LIQUID._int,
                    flags,
                    liquids[rx]);
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
            PhysicsObjects.static_box(x, y, size, 0, friction, 0.0003f, 0, solid);
        }
    }

    private void genWall(int floor_size, float spacing, float size, float start_x, float start_y, Solid solid)
    {
        System.out.println("generating wall size: " + floor_size);
        for (int i = 0; i < floor_size; i++)
        {
            float x = start_x;
            float y = start_y + i * spacing;
            PhysicsObjects.static_box(x, y, size, 0, 0.0f, 0.0f, 0, solid);
        }
    }

    private void genTestCrate(float size, float x, float y)
    {
        PhysicsObjects.dynamic_block(x, y, size, .1f, 0.02f, 0.0001f, 0, Solid.ANDESITE);
    }

    private void genTestTriangle(float size, float x, float y)
    {
       PhysicsObjects.tri(x, y, size, 0, .1f, 0.02f, 0.0003f);
    }

    private void genPlayer(float size, float x, float y)
    {
        var player = ecs.registerEntity("player");
        var entity_id = PhysicsObjects.wrap_model(PLAYER_MODEL_INDEX, x, y, size, HullFlags.IS_POLYGON._int, 100.5f, 0.05f, 0,0);
        var cursor_id = PhysicsObjects.circle_cursor(0,0, 10);

        ecs.attachComponent(player, Component.EntityId, new EntityIndex(entity_id));
        ecs.attachComponent(player, Component.CursorId, new EntityIndex(cursor_id));
        ecs.attachComponent(player, Component.ControlPoints, new ControlPoints());
        ecs.attachComponent(player, Component.CameraFocus, new CameraFocus());
        ecs.attachComponent(player, Component.LinearForce, new LinearForce(1600));
    }

    private void genTestFigureNPC_2(float size, float x, float y)
    {
        PhysicsObjects.wrap_model(TEST_MODEL_INDEX_2, x, y, size, HullFlags.IS_POLYGON._int, 50, 0.02f, 0, 0);
    }

    private void genTestFigureNPC(float size, float x, float y)
    {
        PhysicsObjects.wrap_model(PLAYER_MODEL_INDEX, x, y, size, HullFlags.IS_POLYGON._int, 50, 0.02f, 0,0);
    }

    private void genBoxModelNPC(float size, float x, float y)
    {
        PhysicsObjects.wrap_model(TEST_SQUARE_INDEX, x, y, size, HullFlags.IS_POLYGON._int, .1f, 0.02f, 0,0);
    }

    // note: order of adding systems is important
    private void loadSystems()
    {
        ecs.registerSystem(new PhysicsSimulation(ecs, uniformGrid)); // all physics calculations should be done first
        ecs.registerSystem(new CameraTracking(ecs, uniformGrid)); // camera is handled before rendering occurs, but after collision has been resolved
        ecs.registerSystem(screenBlankSystem); // the blanking system clears the screen before rendering passes

        // renderers are added in the order in which they will render

        ecs.registerSystem(new BackgroundRenderer(ecs));

        if (ACTIVE_RENDERERS.contains(RenderType.MODELS))
        {
            //ecs.registerSystem(new CrateRenderer(ecs));
            ecs.registerSystem(new ModelRenderer(ecs, "block_model.glsl", PLAYER_MODEL_INDEX));
            ecs.registerSystem(new ModelRenderer(ecs, "block_model.glsl", BASE_BLOCK_INDEX));
            ecs.registerSystem(new ModelRenderer(ecs, "block_model.glsl", BASE_TRI_INDEX));
            ecs.registerSystem(new LiquidRenderer(ecs));
        }

        ecs.registerSystem(new MouseRenderer(ecs));

        // the following are debug renderers

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

        if (ACTIVE_RENDERERS.contains(RenderType.ENTITIES))
        {
            ecs.registerSystem(new EntityRenderer(ecs));
        }
    }

    @Override
    public void load()
    {
        // player character
        genPlayer(1f, 0, 2500);
        //genCursor(10, 0, 0);
        //genTestFigureNPC_2(1f, 100, 500);

//        genTestFigureNPC(1f, 200, 1200);
//        genSquares(1,  25f, 25f, 420, 200);
//        genTestFigureNPC(1f, 200, 100);
//        genSquares(1,  25f, 25f, 420, 200);
//        genTestFigureNPC(1f, 200, 250);
//        genSquares(1,  25f, 25f, 420, 200);
//        genTestFigureNPC(1f, 100, 50);

//        genWater(100, 15f, 15f, 0, 3000, Liquid.WATER);
//        genSquaresRando(40,  32f, 32f, 0.8f,-50, 200, Solid.CLAYSTONE, Solid.SOAPSTONE, Solid.MUDSTONE);
//        genBlocks(40,  32f, 32f, 2500, 200, Solid.GREENSCHIST, Solid.SCHIST, Solid.BLUESCHIST, Solid.WHITESCHIST);
//        genTriangles(50,  24f, 24f, 2500, 3800);

        //PhysicsObjects.static_tri(0,-25, 150, 1, 0.02f);
        //PhysicsObjects.static_box(0,0,10,10, 0f);

//        genFloor(16, 150f, 150f, -70, -100, 0.5f, Solid.ANDESITE);
//        genFloor(32, 150f, 150f, 1700, -100, 0.5f, Solid.ANDESITE);
//        genFloor(32, 150f, 150f, 1700, 2200, 0.5f, Solid.DIORITE);
//
//        genWall(15, 150f, 150f, -220, -100, Solid.ANDESITE);
//        genWall(5, 150f, 150f, 2000, 1500, Solid.DIORITE);
//        genWall(5, 150f, 150f, 4880, -100, Solid.ANDESITE);

        loadSystems();
    }

    FastNoiseLite noise = new FastNoiseLite();

    @Override
    public void start()
    {
        noise.SetNoiseType(FastNoiseLite.NoiseType.Perlin);
        noise.SetFractalType(FastNoiseLite.FractalType.FBm);
    }

    private Set<Sector> last_loaded_sectors = new HashSet<>();
    private Set<Sector> loaded_sectors = new HashSet<>();

    @Override
    public void update(float dt)
    {
        float o_xo = uniformGrid.outer_x_origin();
        float o_yo = uniformGrid.outer_y_origin();

        float o_xu = o_xo + uniformGrid.outer_width;
        float o_yu = o_yo + uniformGrid.outer_height;

        var sector_0_key = UniformGridRenderer.get_sector_for_point(o_xo, o_yo);
        var sector_2_key = UniformGridRenderer.get_sector_for_point(o_xu, o_yu);

        float sector_size = UniformGrid.SECTOR_SIZE;

        float sector_0_origin_x = (float)sector_0_key[0] * sector_size;
        float sector_0_origin_y = (float)sector_0_key[1] * sector_size;

        float sector_2_origin_x = (float)sector_2_key[0] * sector_size;
        float sector_2_origin_y = (float)sector_2_key[1] * sector_size;

        last_loaded_sectors.clear();
        last_loaded_sectors.addAll(loaded_sectors);
        loaded_sectors.clear();

        boolean load_changed = false;
        for (int sx = sector_0_key[0]; sx <= sector_2_key[0]; sx ++)
        {
            for (int sy = sector_0_key[1]; sy <= sector_2_key[1]; sy++)
            {
                var sector = new Sector(sx, sy);
                loaded_sectors.add(sector);
                if (!last_loaded_sectors.contains(sector))
                {
                    //System.out.println("loading sector: ["+sx+","+sy+"]");
                    load_sector(sector);
                    load_changed = true;
                }
            }
        }

        for (var last : last_loaded_sectors)
        {
            if (!loaded_sectors.contains(last))
            {
                //System.out.println("unloading sector: ["+last.x+","+last.y+"]");
                load_changed = true;
            }
        }

        if (load_changed)
        {
            //System.out.println(loaded_sectors.size() + " sectors loaded");
        }

        uniformGrid.update_sector_metrics(loaded_sectors, sector_0_origin_x, sector_0_origin_y,
            Math.abs(sector_0_origin_x - (sector_2_origin_x + sector_size)),
            Math.abs(sector_0_origin_y - (sector_2_origin_y + sector_size)));
    }

    private float map(float x, float in_min, float in_max, float out_min, float out_max)
    {
        return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
    }

    private Solid[] block_pallette = new Solid[]
        {
            Solid.MUDSTONE,
            Solid.CLAYSTONE,
            Solid.WHITESCHIST,
            Solid.GREENSCHIST,
            Solid.BLUESCHIST,
            Solid.SCHIST,
            Solid.WHITESCHIST,
            Solid.GREENSCHIST,
            Solid.BLUESCHIST,
            Solid.SCHIST,
            Solid.WHITESCHIST,
            Solid.GREENSCHIST,
            Solid.BLUESCHIST,
            Solid.SCHIST,
        };

    private static final float block_range_floor = -0.03f;

    private final int m(float n)
    {
        return (int)map(n, block_range_floor, 1f, 0f, (float)block_pallette.length);
    }

    private void load_sector(Sector sector)
    {
        float x_offset = sector.x() * (int)UniformGrid.SECTOR_SIZE;
        float y_offset = sector.y() * (int)UniformGrid.SECTOR_SIZE;

        var batch = new PhysicsEntityBatch(sector);

        boolean flip = false;
        for (int x = 0; x < UniformGrid.BLOCK_COUNT; x++)
        {
            for (int y = 0; y < UniformGrid.BLOCK_COUNT; y++)
            {

                float world_x = (x * UniformGrid.BLOCK_SIZE) + x_offset;
                float world_x_block = world_x + (UniformGrid.BLOCK_SIZE / 2f);
                float world_y = (y * UniformGrid.BLOCK_SIZE) + y_offset;
                float world_y_block = world_y + (UniformGrid.BLOCK_SIZE / 2f);

                float block_x = world_x_block / UniformGrid.BLOCK_SIZE;
                float block_y = world_y / UniformGrid.BLOCK_SIZE;

                float n = noise.GetNoise(block_x, block_y);
                int[] nn = new int[8];
                nn[0] = m(noise.GetNoise(block_x - 1, block_y - 1));
                nn[1] = m(noise.GetNoise(block_x, block_y - 1));
                nn[2] = m(noise.GetNoise(block_x + 1, block_y-1));

                nn[0] = m(noise.GetNoise(block_x - 1, block_y));
                nn[0] = m(noise.GetNoise(block_x + 1, block_y));

                nn[0] = m(noise.GetNoise(block_x - 1, block_y + 1));
                nn[1] = m(noise.GetNoise(block_x, block_y + 1));
                nn[2] = m(noise.GetNoise(block_x + 1, block_y + 1));

                boolean gen_block = n >= block_range_floor;
                boolean gen_dyn = false;

                float sz = UniformGrid.BLOCK_SIZE + 1;
                float szw = rando_float(UniformGrid.BLOCK_SIZE * 0.75f , .85f);

                if (gen_block)
                {
                    int block = m(n);
                    var solid = block_pallette[block];
                    batch.new_block(gen_dyn, world_x_block, world_y_block, sz, 90f, 0.03f, 0.0003f, HullFlags.OUT_OF_BOUNDS._int, solid);
                }
                else if (n < -.2)
                {
                    int flags = flip
                        ? PointFlags.FLOW_LEFT.bits
                        : 0;
                    flip = !flip;
                    batch.new_liquid(world_x, world_y,  szw, .1f, 0.0f, 0.00000f,
                        HullFlags.IS_LIQUID._int | HullFlags.OUT_OF_BOUNDS._int,
                        flags, Liquid.WATER);
                }
                else if (n < -.15)
                {
                    batch.new_tri(world_x, world_y,  sz, HullFlags.OUT_OF_BOUNDS._int,.1f, 0.0f, 0.00000f);
                }
            }
        }
        GPGPU.core_memory.new_batch(batch);
    }
}
