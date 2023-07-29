package com.controllerface.bvge.ecs.systems.renderers;

import com.controllerface.bvge.Main;
import com.controllerface.bvge.ecs.ECS;
import com.controllerface.bvge.ecs.systems.GameSystem;
import com.controllerface.bvge.gl.Shader;
import com.controllerface.bvge.ecs.systems.renderers.batches.EdgeRenderBatch;
import com.controllerface.bvge.util.AssetPool;
import com.controllerface.bvge.util.Constants;

import java.util.ArrayList;
import java.util.List;

/**
 * Manages rendering of edge constraints. All edges that are defined in the currently
 * loaded physics state are rendered as lines.
 */
public class EdgeRenderer extends GameSystem
{
    private final Shader shader;
    private final List<EdgeRenderBatch> batches;
    private int last_count = 0;

    public EdgeRenderer(ECS ecs)
    {
        super(ecs);
        this.batches = new ArrayList<>();
        this.shader = AssetPool.getShader("object_outline.glsl");
    }

    private void render()
    {
        for (EdgeRenderBatch batch : batches)
        {
            batch.render();
            batch.clear();
        }
    }

    @Override
    public void run(float dt)
    {
        // todo: right now, this check only adds batches, never reducing them if the count goes
        //  low enough that some batches would be unneeded. This will leak memory resources
        //  so should be adjusted when deleting bodies is added.

        var edge_count = Main.Memory.edgesCount();
        var needed_batches = edge_count / Constants.Rendering.MAX_BATCH_SIZE;
        var r = edge_count % Constants.Rendering.MAX_BATCH_SIZE;
        if (r > 0)
        {
            needed_batches++;
        }
        while (needed_batches > batches.size())
        {
            var b = new EdgeRenderBatch(shader);
            batches.add(b);
            b.start();
        }

        int next = 0;
        int offset = 0;
        for (int i = edge_count; i > 0; i -= Constants.Rendering.MAX_BATCH_SIZE)
        {
            int count = Math.min(Constants.Rendering.MAX_BATCH_SIZE, i);
            var b = batches.get(next++);
            b.setLineCount(count);
            b.setOffset(offset);
            offset += count;
        }
        render();
    }

    @Override
    public void shutdown()
    {

    }
}