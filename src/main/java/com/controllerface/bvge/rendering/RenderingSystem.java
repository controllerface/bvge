package com.controllerface.bvge.rendering;

import com.controllerface.bvge.ecs.ECS;
import com.controllerface.bvge.ecs.GameSystem;
import com.controllerface.bvge.game.state.PlayerInventory;
import com.controllerface.bvge.physics.UniformGrid;
import com.controllerface.bvge.rendering.renderers.*;

import java.util.ArrayList;
import java.util.EnumSet;
import java.util.List;

import static com.controllerface.bvge.models.geometry.ModelRegistry.*;

public class RenderingSystem extends GameSystem
{
    private final List<Renderer> renderers = new ArrayList<>();
    private final Runnable window_upkeep;

    private enum RenderType
    {
        GAME,       // normal objects
        HULLS,      // physics hulls
        BOUNDS,     // bounding boxes
        POINTS,     // model vertices
        ENTITIES,   // entity roots
        GRID,       // uniform grid
    }


    private static final RenderType[] debug_renderers =
        {
//            RenderType.HULLS,
//            RenderType.POINTS,
//            RenderType.ENTITIES,
            RenderType.BOUNDS,
//            RenderType.GRID
        };

    private static final EnumSet<RenderType> ACTIVE_RENDERERS = EnumSet.of(RenderType.GAME, debug_renderers);

    public RenderingSystem(ECS ecs, PlayerInventory player_inventory, UniformGrid uniform_grid, Runnable window_upkeep)
    {
        super(ecs);
        this.window_upkeep = window_upkeep;

        if (ACTIVE_RENDERERS.contains(RenderType.GAME))
        {
            renderers.add(new BackgroundRenderer());
            renderers.add(new MouseRenderer(ecs));
            renderers.add(new HUDRenderer(player_inventory));
            int[] model_ids = new int[]{ PLAYER_MODEL_INDEX, BASE_BLOCK_INDEX, BASE_SPIKE_INDEX, R_SHARD_INDEX, L_SHARD_INDEX };
            renderers.add(new ModelRenderer(ecs, uniform_grid, model_ids));
            renderers.add(new LiquidRenderer(ecs, uniform_grid));
        }

        // debug renderers

        if (ACTIVE_RENDERERS.contains(RenderType.HULLS))
        {
            renderers.add(new EdgeRenderer());
            renderers.add(new CircleRenderer());
        }

        if (ACTIVE_RENDERERS.contains(RenderType.BOUNDS))
        {
            renderers.add(new BoundingBoxRenderer());
        }

        if (ACTIVE_RENDERERS.contains(RenderType.POINTS))
        {
            renderers.add(new PointRenderer());
        }

        if (ACTIVE_RENDERERS.contains(RenderType.GRID))
        {
            renderers.add(new UniformGridRenderer(ecs, uniform_grid));
        }

        if (ACTIVE_RENDERERS.contains(RenderType.ENTITIES))
        {
            renderers.add(new EntityPositionRenderer());
        }
    }

    @Override
    public void tick(float dt)
    {
        window_upkeep.run();
        for (var renderer : renderers)
        {
            renderer.render();
        }
    }

    @Override
    public void shutdown()
    {
        for (var renderer : renderers)
        {
            renderer.destroy();
        }
    }
}
