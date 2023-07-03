package com.controllerface.bvge.ecs.systems.renderers;

import com.controllerface.bvge.ecs.ECS;
import com.controllerface.bvge.ecs.systems.GameSystem;
import com.controllerface.bvge.ecs.systems.physics.SpatialMap;
import com.controllerface.bvge.gl.Shader;
import com.controllerface.bvge.gl.batches.RectRenderBatch;
import com.controllerface.bvge.util.AssetPool;
import org.joml.Vector3f;

import java.util.ArrayList;
import java.util.List;

public class SpacePartitionRenderer extends GameSystem
{
    private Shader shader;
    private List<RectRenderBatch> batches;
    private final Vector3f color = new Vector3f(0f,0f,1f);
    private final Vector3f color2 = new Vector3f(.5f,0.1f,0.1f);

    private final SpatialMap spatialMap;

    public SpacePartitionRenderer(ECS ecs, SpatialMap spatialMap)
    {
        super(ecs);
        this.spatialMap = spatialMap;
        this.batches = new ArrayList<>();
        this.shader = AssetPool.getShader("debugLine2D.glsl");
    }

    private void add(float x, float y, float w, float h, Vector3f colorToUse)
    {
        //var colorToUse = color; //box.playerTouch ? color2 : color;
        boolean added = false;
        for (RectRenderBatch batch : batches)
        {
            if (batch.hasRoom())
            {
                batch.addBox(x, y, w, h, colorToUse);
                added = true;
                break;
            }
        }

        if (!added)
        {
            RectRenderBatch newBatch = new RectRenderBatch(0, shader);
            newBatch.start();
            batches.add(newBatch);
            newBatch.addBox(x, y, w, h, colorToUse);
        }
    }

    private void render()
    {
        for (RectRenderBatch batch : batches)
        {
            batch.render();
            batch.clear();
        }
    }

    private boolean hasSet = false;

    @Override
    public void run(float dt)
    {
        if (spatialMap == null) return;
        for (float i = 0; i < spatialMap.getWidth(); i += spatialMap.getX_spacing())
        {
            for (float j = 0; j < spatialMap.getHeight(); j += spatialMap.getY_spacing())
            {
//                if (i + spatialMap.getX_spacing() > spatialMap.getWidth()
//                        || j + spatialMap.getY_spacing() > spatialMap.getHeight())
//                {
//                    continue;
//                }
                this.add(i, j, spatialMap.getX_spacing(), spatialMap.getY_spacing(), color2);
            }
        }
        this.add(0,0, spatialMap.getWidth(), spatialMap.getHeight(), color);
        render();
    }
}