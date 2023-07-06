package com.controllerface.bvge.ecs.systems.renderers;

import com.controllerface.bvge.data.FBounds2D;
import com.controllerface.bvge.ecs.ECS;
import com.controllerface.bvge.ecs.components.Component;
import com.controllerface.bvge.ecs.components.GameComponent;
import com.controllerface.bvge.ecs.systems.GameSystem;
import com.controllerface.bvge.gl.Shader;
import com.controllerface.bvge.gl.batches.BoxRenderBatch;
import com.controllerface.bvge.util.AssetPool;
import org.joml.Vector3f;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class BoundingBoxRenderer extends GameSystem
{
    private Shader shader;
    private List<BoxRenderBatch> batches;
    private final Vector3f color = new Vector3f(0.8f,0.8f,0.2f);

    public BoundingBoxRenderer(ECS ecs)
    {
        super(ecs);
        this.batches = new ArrayList<>();
        this.shader = AssetPool.getShader("debugLine2D.glsl");
    }

    private void add(FBounds2D box)
    {
        boolean added = false;
        for (BoxRenderBatch batch : batches)
        {
            if (batch.hasRoom())
            {
                batch.addBox(box, color);
                added = true;
                break;
            }
        }

        if (!added)
        {
            BoxRenderBatch newBatch = new BoxRenderBatch(0, shader);
            newBatch.start();
            batches.add(newBatch);
            newBatch.addBox(box, color);
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
            FBounds2D box = Component.BoundingBox.coerce(component);
            this.add(box);
        }
        render();
    }
}