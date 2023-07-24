package com.controllerface.bvge.ecs.systems.renderers;

import com.controllerface.bvge.data.FBody2D;
import com.controllerface.bvge.data.FEdge2D;
import com.controllerface.bvge.ecs.ECS;
import com.controllerface.bvge.ecs.components.Component;
import com.controllerface.bvge.ecs.components.GameComponent;
import com.controllerface.bvge.ecs.systems.GameSystem;
import com.controllerface.bvge.gl.Shader;
import com.controllerface.bvge.gl.batches.LineRenderBatch;
import com.controllerface.bvge.util.AssetPool;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import static com.controllerface.bvge.data.PhysicsObjects.FLAG_STATIC;

public class LineRenderer extends GameSystem
{
    private Shader shader;
    private List<LineRenderBatch> batches;

    public LineRenderer(ECS ecs)
    {
        super(ecs);
        this.batches = new ArrayList<>();
        this.shader = AssetPool.getShader("debugLine2D.glsl");
    }

    private void add(FEdge2D line)
    {
        boolean added = false;
        for (LineRenderBatch batch : batches)
        {
            if (batch.hasRoom())
            {
                batch.addLine(line);
                added = true;
                break;
            }
        }

        if (!added)
        {
            LineRenderBatch newBatch = new LineRenderBatch(0, shader);
            newBatch.start();
            batches.add(newBatch);
            newBatch.addLine(line);
        }
    }

    private void render()
    {
        for (LineRenderBatch batch : batches)
        {
            batch.render();
            batch.clear();
        }
    }

    @Override
    public void run(float dt)
    {
        for (Map.Entry<String, GameComponent> entry : ecs.getComponents(Component.RigidBody2D).entrySet())
        {
            GameComponent component = entry.getValue();
            FBody2D body = Component.RigidBody2D.coerce(component);
            for (FEdge2D edge : body.edges())
            {
                add(edge);
            }
        }
        render();
    }

    @Override
    public void shutdown()
    {

    }
}