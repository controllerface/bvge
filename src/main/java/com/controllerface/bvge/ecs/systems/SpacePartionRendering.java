package com.controllerface.bvge.ecs.systems;

import com.controllerface.bvge.ecs.ECS;
import com.controllerface.bvge.ecs.systems.physics.SpatialMap;
import com.controllerface.bvge.rendering.BoxRenderBatch;
import com.controllerface.bvge.rendering.Shader;
import com.controllerface.bvge.util.AssetPool;
import com.controllerface.bvge.util.quadtree.QuadRectangle;
import org.joml.Vector3f;

import java.util.ArrayList;
import java.util.List;

public class SpacePartionRendering extends GameSystem
{
    private Shader shader;
    private List<BoxRenderBatch> batches;
    private final Vector3f color = new Vector3f(0f,0f,1f);
    private final Vector3f color2 = new Vector3f(1f,0f,0f);


    public SpacePartionRendering(ECS ecs)
    {
        super(ecs);
        this.batches = new ArrayList<>();
        this.shader = AssetPool.getShader("debugLine2D.glsl");
    }

    private void add(QuadRectangle box)
    {
        var colorToUse = box.playerTouch ? color2 : color;
        boolean added = false;
        for (BoxRenderBatch batch : batches)
        {
            if (batch.hasRoom())
            {
                batch.addBox(box, colorToUse);
                added = true;
                break;
            }
        }

        if (!added)
        {
            BoxRenderBatch newBatch = new BoxRenderBatch(0, shader);
            newBatch.start();
            batches.add(newBatch);
            newBatch.addBox(box, colorToUse);
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

    private SpatialMap spatialMap;

    public void setSpatialMap( SpatialMap spatialMap)
    {
        this.spatialMap = spatialMap;
    }

    private boolean hasSet = false;

    @Override
    public void run(float dt)
    {
        if (spatialMap == null) return;

        //if (!hasSet)
        //{
        var pr = new ArrayList<QuadRectangle>();
            for (QuadRectangle rect : spatialMap.rects)
            {
                if (rect.playerTouch) pr.add(rect);
                else add(rect);
            }
            pr.forEach(p->this.add(p));
            hasSet = true;
        //}

        render();
    }
}