package com.controllerface.bvge.ecs.systems;

import com.controllerface.bvge.ecs.ECS;
import com.controllerface.bvge.ecs.RigidBody2D;
import com.controllerface.bvge.ecs.systems.physics.VerletPhysics;
import com.controllerface.bvge.rendering.BoxRenderBatch;
import com.controllerface.bvge.rendering.Shader;
import com.controllerface.bvge.util.AssetPool;
import com.controllerface.bvge.util.quadtree.QuadRectangle;
import com.controllerface.bvge.util.quadtree.QuadTree;
import org.joml.Vector3f;

import java.util.ArrayList;
import java.util.List;

public class SpacePartionRendering extends GameSystem
{
    private Shader shader;
    private List<BoxRenderBatch> batches;
    private final Vector3f color = new Vector3f(0.3f,0.2f,0.8f);

    public SpacePartionRendering(ECS ecs)
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
            //batch.clear();
        }
    }

    private VerletPhysics.SpatialMap spatialMap;

    public void setSpatialMap( VerletPhysics.SpatialMap spatialMap)
    {
        this.spatialMap = spatialMap;
    }

    private boolean hasSet = false;

    @Override
    public void run(float dt)
    {
        if (spatialMap == null) return;

        if (!hasSet)
        {
            for (QuadRectangle rect : spatialMap.rects)
            {
                add(rect);
            }
            hasSet = true;
        }

        render();
    }
}