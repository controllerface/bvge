package com.controllerface.bvge.ecs.systems;

import com.controllerface.bvge.ecs.*;
import com.controllerface.bvge.rendering.BoxRenderBatch;
import com.controllerface.bvge.rendering.Shader;
import com.controllerface.bvge.util.AssetPool;
import com.controllerface.bvge.util.quadtree.QuadRectangle;
import com.controllerface.bvge.util.quadtree.QuadTree;
import org.joml.Vector3f;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class BoundingBoxRendering extends GameSystem
{
    private Shader shader;
    private List<BoxRenderBatch> batches;
    private final Vector3f color = new Vector3f(1.0f,1.0f,1.0f);

    public BoundingBoxRendering(ECS ecs)
    {
        super(ecs);
        this.batches = new ArrayList<>();
        this.shader = AssetPool.getShader("debugLine2D.glsl");
    }

    private void add(QuadRectangle box)
    {
        boolean added = false;
        for (BoxRenderBatch batch : batches)
        {
            if (batch.hasRoom())
            {
                batch.addLine(box, color);
                added = true;
                break;
            }
        }

        if (!added)
        {
            BoxRenderBatch newBatch = new BoxRenderBatch(0, shader);
            newBatch.start();
            batches.add(newBatch);
            newBatch.addLine(box, color);
        }
    }

    private void render()
    {
        for (BoxRenderBatch batch : batches)
        {
            batch.render();
            batch.clear();
        }
    }

    @Override
    public void run(float dt)
    {
        for (Map.Entry<String, GameComponent> entry : ecs.getComponents(Component.BoundingBox).entrySet())
        {
            GameComponent component = entry.getValue();
            QuadRectangle box = Component.BoundingBox.coerce(component);
            this.add(box);
        }
        render();
    }
}