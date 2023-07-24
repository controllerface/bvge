package com.controllerface.bvge.ecs.systems.renderers;

import com.controllerface.bvge.Main;
import com.controllerface.bvge.cl.OpenCL;
import com.controllerface.bvge.data.FBody2D;
import com.controllerface.bvge.data.FEdge2D;
import com.controllerface.bvge.ecs.ECS;
import com.controllerface.bvge.ecs.components.Component;
import com.controllerface.bvge.ecs.components.GameComponent;
import com.controllerface.bvge.ecs.systems.GameSystem;
import com.controllerface.bvge.gl.Shader;
import com.controllerface.bvge.gl.batches.LineRenderBatchEX;
import com.controllerface.bvge.util.AssetPool;
import com.controllerface.bvge.util.Constants;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

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

    private void render(int current_edge_count)
    {
        for (LineRenderBatchEX batch : batches)
        {
            batch.render(current_edge_count);
            batch.clear();
        }
    }

    @Override
    public void run(float dt)
    {
        // todo: query the number of lines and divide that into batches
        //  -> check current batch count and add more if needed
        //  -> run CL jobs to batch in the edge vertices
        //  -> on last batch, if less than batch max, set length to read the right number of vertices

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
        for (int i = edge_count; i > 0; i -= Constants.Rendering.MAX_BATCH_SIZE)
        {
            int count = Math.min(Constants.Rendering.MAX_BATCH_SIZE, i);
            var b = batches.get(next++);
            b.setLineCount(count);
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
        render(edge_count);
    }

    @Override
    public void shutdown()
    {

    }
}