package com.controllerface.bvge.ecs.systems.renderers;

import com.controllerface.bvge.ecs.ECS;
import com.controllerface.bvge.ecs.systems.GameSystem;
import com.controllerface.bvge.ecs.systems.physics.SpatialPartition;
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
    private final Vector3f color3 = new Vector3f(1f,0.1f,0.1f);

    private final SpatialPartition spatialPartition;

    public SpacePartitionRenderer(ECS ecs, SpatialPartition spatialPartition)
    {
        super(ecs);
        this.spatialPartition = spatialPartition;
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

    private Vector3f getColor(float x, float y)
    {
        if (accum < 1) return color2;

        var c = spatialPartition.countAtIndex(x, y);
        if (c > 0)
        {
            return color3;
        }
        return color2;
    }

    float accum = 0;
    @Override
    public void run(float dt)
    {
        if (spatialPartition == null) return;
        var x_max = spatialPartition.getX_origin() + spatialPartition.getWidth();
        var y_max = spatialPartition.getY_origin() + spatialPartition.getHeight();

        for (float i = spatialPartition.getX_origin(); i < x_max; i += spatialPartition.getX_spacing())
        {
            for (float j = spatialPartition.getY_origin(); j < y_max; j += spatialPartition.getY_spacing())
            {
                var c = getColor(i, j);
                this.add(i, j, spatialPartition.getX_spacing(), spatialPartition.getY_spacing(), c);
            }
        }


        accum+=dt;

        this.add(spatialPartition.getX_origin(),
                spatialPartition.getY_origin(),
                spatialPartition.getWidth(),
                spatialPartition.getHeight(), color);
        render();
    }
}