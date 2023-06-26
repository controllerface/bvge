package com.controllerface.bvge.ecs.systems.renderers;

import com.controllerface.bvge.data.FBounds2D;
import com.controllerface.bvge.ecs.ECS;
import com.controllerface.bvge.ecs.systems.GameSystem;
import com.controllerface.bvge.ecs.systems.physics.SpatialMap;
import com.controllerface.bvge.gl.batches.BoxRenderBatch;
import com.controllerface.bvge.gl.Shader;
import com.controllerface.bvge.util.AssetPool;
import com.controllerface.bvge.ecs.components.QuadRectangle;
import org.joml.Vector3f;

import java.util.ArrayList;
import java.util.List;

public class SpacePartitionRenderer extends GameSystem
{
    private Shader shader;
    private List<BoxRenderBatch> batches;
    private final Vector3f color = new Vector3f(0f,0f,1f);
    private final Vector3f color2 = new Vector3f(1f,0f,0f);


    public SpacePartitionRenderer(ECS ecs)
    {
        super(ecs);
        this.batches = new ArrayList<>();
        this.shader = AssetPool.getShader("debugLine2D.glsl");
    }

    private void add(FBounds2D box)
    {
        var colorToUse = color; //box.playerTouch ? color2 : color;
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
//        var pr = new ArrayList<QuadRectangle>();
//            for (FBounds2D rect : spatialMap.rects)
//            {
//                //if (rect.playerTouch) pr.add(rect);
//                //else
//                    add(rect);
//            }
//            pr.forEach(p->this.add(p));
//            hasSet = true;
        //}

        render();
    }
}