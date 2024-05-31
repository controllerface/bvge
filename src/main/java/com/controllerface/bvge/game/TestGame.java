package com.controllerface.bvge.game;

import com.controllerface.bvge.cl.GPGPU;
import com.controllerface.bvge.ecs.ECS;
import com.controllerface.bvge.ecs.components.*;
import com.controllerface.bvge.ecs.systems.CameraTracking;
import com.controllerface.bvge.ecs.systems.GameSystem;
import com.controllerface.bvge.ecs.systems.SectorLoader;
import com.controllerface.bvge.editor.Editor;
import com.controllerface.bvge.geometry.MeshRegistry;
import com.controllerface.bvge.geometry.ModelRegistry;
import com.controllerface.bvge.gl.renderers.*;
import com.controllerface.bvge.physics.PhysicsEntityBatch;
import com.controllerface.bvge.physics.PhysicsObjects;
import com.controllerface.bvge.physics.PhysicsSimulation;
import com.controllerface.bvge.physics.UniformGrid;
import com.controllerface.bvge.substances.Liquid;
import com.controllerface.bvge.substances.Solid;
import com.controllerface.bvge.util.Constants;
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


    public TestGame(ECS ecs, GameSystem screenBlankSystem)
    {
        super(ecs);

        MeshRegistry.init();
        ModelRegistry.init();

        this.screenBlankSystem = screenBlankSystem;
    }

    private void genPlayer(float size, float x, float y)
    {
        var player = ecs.registerEntity("player");
        var entity_id = PhysicsObjects.wrap_model(GPGPU.core_memory, PLAYER_MODEL_INDEX, x, y, size, HullFlags.IS_POLYGON._int, 100.5f, 0.05f, 0,0);
        var cursor_id = PhysicsObjects.circle_cursor(GPGPU.core_memory, 0,0, 10);

        ecs.attachComponent(player, Component.EntityId, new EntityIndex(entity_id));
        ecs.attachComponent(player, Component.CursorId, new EntityIndex(cursor_id));
        ecs.attachComponent(player, Component.ControlPoints, new ControlPoints());
        ecs.attachComponent(player, Component.CameraFocus, new CameraFocus());
        ecs.attachComponent(player, Component.LinearForce, new LinearForce(1600));
    }


    // note: order of adding systems is relevant
    private void loadSystems()
    {
        ecs.registerSystem(new SectorLoader(ecs, uniformGrid));
        ecs.registerSystem(new PhysicsSimulation(ecs, uniformGrid));
        ecs.registerSystem(new CameraTracking(ecs, uniformGrid));

        ecs.registerSystem(screenBlankSystem);

        ecs.registerSystem(new BackgroundRenderer(ecs));
        ecs.registerSystem(new MouseRenderer(ecs));

        if (ACTIVE_RENDERERS.contains(RenderType.MODELS))
        {
            ecs.registerSystem(new ModelRenderer(ecs, PLAYER_MODEL_INDEX, BASE_BLOCK_INDEX, BASE_SPIKE_INDEX, BASE_SHARD_INDEX));
            ecs.registerSystem(new LiquidRenderer(ecs));
        }


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
        genPlayer(1f, 0, 3000);
        loadSystems();
    }

    @Override
    public void start()
    {}

    @Override
    public void update(float dt)
    {

    }
}
