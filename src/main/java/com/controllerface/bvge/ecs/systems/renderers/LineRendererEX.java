package com.controllerface.bvge.ecs.systems.renderers;

import com.controllerface.bvge.Main;
import com.controllerface.bvge.ecs.ECS;
import com.controllerface.bvge.ecs.systems.GameSystem;
import com.controllerface.bvge.gl.Shader;
import com.controllerface.bvge.gl.batches.LineRenderBatchEX;
import com.controllerface.bvge.util.AssetPool;
import com.controllerface.bvge.util.Constants;

import java.util.ArrayList;
import java.util.List;

public class LineRendererEX extends GameSystem
{
    private Shader shader;
    private List<LineRenderBatchEX> batches;

    public LineRendererEX(ECS ecs)
    {
        super(ecs);
        this.batches = new ArrayList<>();
        this.shader = AssetPool.getShader("object_outline.glsl");
    }

    private void render()
    {
        for (LineRenderBatchEX batch : batches)
        {
            batch.render();
            batch.clear();
        }
    }

    @Override
    public void run(float dt)
    {
        var edge_count = Main.Memory.edgesCount();
        var needed_batches = edge_count / Constants.Rendering.MAX_BATCH_SIZE;
        var r = edge_count % Constants.Rendering.MAX_BATCH_SIZE;
        if (r > 0)
        {
            needed_batches++;
        }
        while (needed_batches > batches.size())
        {
            var b = new LineRenderBatchEX(0, shader);
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


//        for (Map.Entry<String, GameComponent> entry : ecs.getComponents(Component.RigidBody2D).entrySet())
//        {
//            GameComponent component = entry.getValue();
//            FBody2D body = Component.RigidBody2D.coerce(component);
//            for (FEdge2D edge : body.edges())
//            {
//                add(edge);
//            }
//        }
        render();
    }

    @Override
    public void shutdown()
    {

    }
}