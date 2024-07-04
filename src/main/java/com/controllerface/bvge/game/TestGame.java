package com.controllerface.bvge.game;

import com.controllerface.bvge.cl.GPGPU;
import com.controllerface.bvge.ecs.ECS;
import com.controllerface.bvge.ecs.components.*;
import com.controllerface.bvge.ecs.systems.CameraTracking;
import com.controllerface.bvge.ecs.systems.GameSystem;
import com.controllerface.bvge.ecs.systems.InventorySystem;
import com.controllerface.bvge.game.world.sectors.Sector;
import com.controllerface.bvge.game.world.WorldLoader;
import com.controllerface.bvge.game.world.WorldUnloader;
import com.controllerface.bvge.game.state.PlayerInventory;
import com.controllerface.bvge.geometry.MeshRegistry;
import com.controllerface.bvge.geometry.ModelRegistry;
import com.controllerface.bvge.gl.renderers.*;
import com.controllerface.bvge.physics.PhysicsEntityBatch;
import com.controllerface.bvge.physics.PhysicsObjects;
import com.controllerface.bvge.physics.PhysicsSimulation;
import com.controllerface.bvge.physics.UniformGrid;
import com.controllerface.bvge.util.Constants;
import com.github.benmanes.caffeine.cache.Cache;
import com.github.benmanes.caffeine.cache.Caffeine;

import java.time.Duration;
import java.time.temporal.ChronoUnit;
import java.util.*;
import java.util.concurrent.LinkedBlockingDeque;

import static com.controllerface.bvge.geometry.ModelRegistry.*;

public class TestGame extends GameMode
{
    private final GameSystem blanking_system;
    private final int GRID_WIDTH = 3840;
    private final int GRID_HEIGHT = 2160;

    private final Cache<Sector, PhysicsEntityBatch> sector_cache;
    private final Queue<PhysicsEntityBatch> load_queue;
    private final Queue<Sector> unload_queue;
    private final PlayerInventory player_inventory;

    private enum RenderType
    {
        GAME,       // normal objects
        HULLS,      // physics hulls
        BOUNDS,     // bounding boxes
        POINTS,     // model vertices
        ENTITIES,   // entity roots
        GRID,       // uniform grid
    }

    private static final EnumSet<RenderType> ACTIVE_RENDERERS =
        EnumSet.of(RenderType.GAME
//            ,RenderType.HULLS
//            ,RenderType.POINTS
//            ,RenderType.ENTITIES
//            ,RenderType.BOUNDS
//            ,RenderType.GRID
        );

    private final UniformGrid uniformGrid = new UniformGrid(GRID_WIDTH, GRID_HEIGHT);


    public TestGame(ECS ecs, GameSystem blanking_system)
    {
        super(ecs);

        MeshRegistry.init();
        ModelRegistry.init();

        this.blanking_system = blanking_system;
        this.load_queue = new LinkedBlockingDeque<>();
        this.unload_queue = new LinkedBlockingDeque<>();
        this.player_inventory = new PlayerInventory();
        this.sector_cache = Caffeine.newBuilder()
            .expireAfterAccess(Duration.of(1, ChronoUnit.HOURS))
            .build();

    }

    private void gen_player(float size, float x, float y)
    {
        var player = ecs.register_entity(Constants.PLAYER_ID);

        var entity_id = PhysicsObjects.wrap_model(GPGPU.core_memory.sector_container(),
                PLAYER_MODEL_INDEX, x, y, size,
                100.5f, 0.05f, 0, 0,
                Constants.EntityFlags.CAN_COLLECT.bits);

        var cursor_id = PhysicsObjects.circle_cursor(GPGPU.core_memory.sector_container(),
                0, 0, 10, entity_id[1]);

        var block_cursor = PhysicsObjects.block_cursor(GPGPU.core_memory.sector_container(), x, y);

        ecs.attach_component(player, Component.EntityId,      new EntityIndex(entity_id[0]));
        ecs.attach_component(player, Component.MouseCursorId, new EntityIndex(cursor_id));
        ecs.attach_component(player, Component.BlockCursorId, new EntityIndex(block_cursor));
        ecs.attach_component(player, Component.LinearForce,   new LinearForce(1600));
        ecs.attach_component(player, Component.InputState,    new InputState());
    }

    private void load_systems(float x, float y)
    {
        ecs.register_system(new WorldLoader(ecs, uniformGrid, sector_cache, load_queue, unload_queue));
        ecs.register_system(new PhysicsSimulation(ecs, uniformGrid));
        ecs.register_system(new WorldUnloader(ecs, sector_cache, load_queue, unload_queue));
        ecs.register_system(new CameraTracking(ecs, uniformGrid, x, y));
        ecs.register_system(new InventorySystem(ecs, player_inventory));
        ecs.register_system(blanking_system);

        ecs.register_system(new BackgroundRenderer(ecs));
        ecs.register_system(new MouseRenderer(ecs));
        ecs.register_system(new HUDRenderer(ecs, player_inventory));

        if (ACTIVE_RENDERERS.contains(RenderType.GAME))
        {
            ecs.register_system(new ModelRenderer(ecs, uniformGrid,
                    PLAYER_MODEL_INDEX, BASE_BLOCK_INDEX, BASE_SPIKE_INDEX, R_SHARD_INDEX, L_SHARD_INDEX));
            ecs.register_system(new LiquidRenderer(ecs, uniformGrid));
        }
        
        // debug renderers

        if (ACTIVE_RENDERERS.contains(RenderType.HULLS))
        {
            ecs.register_system(new EdgeRenderer(ecs));
            ecs.register_system(new CircleRenderer(ecs));
        }

        if (ACTIVE_RENDERERS.contains(RenderType.BOUNDS))
        {
            ecs.register_system(new BoundingBoxRenderer(ecs));
        }

        if (ACTIVE_RENDERERS.contains(RenderType.POINTS))
        {
            ecs.register_system(new PointRenderer(ecs));
        }

        if (ACTIVE_RENDERERS.contains(RenderType.GRID))
        {
            ecs.register_system(new UniformGridRenderer(ecs, uniformGrid));
        }

        if (ACTIVE_RENDERERS.contains(RenderType.ENTITIES))
        {
            ecs.register_system(new EntityPositionRenderer(ecs));
        }
    }

    @Override
    public void init()
    {
        float player_size = 1f;
        float player_spawn_x = -250;
        float player_spawn_y = 1500;
        gen_player(player_size, player_spawn_x, player_spawn_y);
        load_systems(player_spawn_x, player_spawn_y);
    }
}
