package com.controllerface.bvge.game;

import com.controllerface.bvge.ecs.ECS;
import com.controllerface.bvge.ecs.components.*;
import com.controllerface.bvge.game.state.PlayerController;
import com.controllerface.bvge.game.state.PlayerInventory;
import com.controllerface.bvge.game.world.WorldLoader;
import com.controllerface.bvge.game.world.WorldUnloader;
import com.controllerface.bvge.gpu.GPU;
import com.controllerface.bvge.memory.sectors.Sector;
import com.controllerface.bvge.models.geometry.MeshRegistry;
import com.controllerface.bvge.models.geometry.ModelRegistry;
import com.controllerface.bvge.physics.PhysicsEntityBatch;
import com.controllerface.bvge.physics.PhysicsObjects;
import com.controllerface.bvge.physics.PhysicsSimulation;
import com.controllerface.bvge.physics.UniformGrid;
import com.controllerface.bvge.rendering.RenderingSystem;
import com.controllerface.bvge.substances.Solid;
import com.github.benmanes.caffeine.cache.Cache;
import com.github.benmanes.caffeine.cache.Caffeine;

import java.time.Duration;
import java.time.temporal.ChronoUnit;
import java.util.Queue;
import java.util.concurrent.LinkedBlockingDeque;
import java.util.concurrent.Semaphore;

import static com.controllerface.bvge.models.geometry.ModelRegistry.PLAYER_MODEL_INDEX;

public class TestGame extends GameMode
{
    private final Runnable window_upkeep;
    private final int GRID_WIDTH = 3840;
    private final int GRID_HEIGHT = 2160;

    private final Cache<Sector, PhysicsEntityBatch> sector_cache;
    private final Queue<PhysicsEntityBatch> load_queue;
    private final Queue<Sector> unload_queue;
    private final PlayerInventory player_inventory;
    private PlayerController player_controller;

    private final UniformGrid uniformGrid = new UniformGrid(GRID_WIDTH, GRID_HEIGHT);

    public TestGame(ECS ecs, Runnable window_upkeep)
    {
        super(ecs);

        this.window_upkeep = window_upkeep;
        this.load_queue = new LinkedBlockingDeque<>();
        this.unload_queue = new LinkedBlockingDeque<>();
        this.player_inventory = new PlayerInventory();

        // todo: look into a custom weigher instead of a flat timeout. weight should be determined by proximity to
        //  sectors nearest to the player.
        this.sector_cache = Caffeine.newBuilder()
            .expireAfterAccess(Duration.of(1, ChronoUnit.HOURS))
            .build();

        MeshRegistry.init();
        ModelRegistry.init();
    }

    private void gen_player(float size, float x, float y)
    {
        var player = ecs.register_entity(Constants.PLAYER_ID);

        var entity_id = PhysicsObjects.wrap_model(GPU.memory.sector_container(),
                PLAYER_MODEL_INDEX, x, y, size,
                100.5f, 0.05f, 0, 0,
                Constants.EntityFlags.CAN_COLLECT.bits);

        var cursor_id = PhysicsObjects.circle_cursor(GPU.memory.sector_container(),
                0, 0, 10, entity_id[1]);

        var block_cursor = PhysicsObjects.block_cursor(GPU.memory.sector_container(), x, y);

        ecs.attach_component(player, ComponentType.Position,      new Position(x, y));
        ecs.attach_component(player, ComponentType.EntityId,      new EntityIndex(entity_id[0]));
        ecs.attach_component(player, ComponentType.MouseCursorId, new EntityIndex(cursor_id));
        ecs.attach_component(player, ComponentType.BlockCursorId, new EntityIndex(block_cursor));
        ecs.attach_component(player, ComponentType.MovementForce, new FloatValue(1000));
        ecs.attach_component(player, ComponentType.JumpForce,     new FloatValue(9.8f * 1000));
        ecs.attach_component(player, ComponentType.InputState,    new PlayerInput());
        ecs.attach_component(player, ComponentType.BlockCursor,   new BlockCursor());

        player_controller = new PlayerController(ecs, player_inventory);
    }

    private void gen_test_wall(int height, float x, float y)
    {
        float size = 1f;
        float y_offset = 0;
        for (int i = 0; i <= height; i++)
        {
            PhysicsObjects.base_block(GPU.memory.sector_container(),
                x, y + y_offset, size, 100, 0, 0, 0, Constants.HullFlags.IS_STATIC.bits, Solid.OBSIDIAN, new int[4]);
            y_offset+=size;
        }
    }

    private void load_systems(float x, float y)
    {
        var world_permit = new Semaphore(0);
        ecs.register_system(new WorldLoader(ecs, uniformGrid, sector_cache, load_queue, unload_queue, world_permit));
        ecs.register_system(new PhysicsSimulation(ecs, uniformGrid, player_controller));
        ecs.register_system(new WorldUnloader(ecs, sector_cache, load_queue, unload_queue, world_permit));
        ecs.register_system(new CameraTracking(ecs, uniformGrid, x, y));
        ecs.register_system(new InventorySystem(ecs, player_inventory));
        ecs.register_system(new RenderingSystem(ecs, player_inventory, uniformGrid, window_upkeep));
    }

    @Override
    public void init()
    {
        float player_size = 1f;
        float player_spawn_x = 0;
        float player_spawn_y = 550;
        gen_player(player_size, player_spawn_x, player_spawn_y);
        //gen_test_wall(500, 0, 500);
        load_systems(player_spawn_x, player_spawn_y);
    }

    @Override
    public void destroy()
    {
        player_controller.release();
    }
}
