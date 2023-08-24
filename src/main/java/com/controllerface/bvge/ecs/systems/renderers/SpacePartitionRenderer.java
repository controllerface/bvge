package com.controllerface.bvge.ecs.systems.renderers;

import com.controllerface.bvge.ecs.ECS;
import com.controllerface.bvge.ecs.systems.GameSystem;
import com.controllerface.bvge.ecs.systems.physics.UniformGrid;
import com.controllerface.bvge.gl.AbstractShader;
import com.controllerface.bvge.ecs.systems.renderers.batches.RectRenderBatch;
import com.controllerface.bvge.util.Assets;
import org.graalvm.collections.Pair;
import org.joml.Vector3f;

import java.util.ArrayList;
import java.util.List;

public class SpacePartitionRenderer extends GameSystem
{
    private AbstractShader shader;
    private List<RectRenderBatch> batches;
    private final Vector3f color = new Vector3f(0f,0f,1f);
    private final Vector3f color2 = new Vector3f(.1f,0.5f,0.1f);
    private final Vector3f color3 = new Vector3f(1f,0.1f,0.1f);

    private final UniformGrid uniformGrid;

    public SpacePartitionRenderer(ECS ecs, UniformGrid uniformGrid)
    {
        super(ecs);
        this.uniformGrid = uniformGrid;
        this.batches = new ArrayList<>();
        this.shader = Assets.shader("debugLine2D.glsl");
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

    private Vector3f getColor(float x, float y)
    {
        if (accum < 1) return color2;

        //var c = spatialPartition.countAtIndex(x, y);
        //if (c > 0)
        //{
            //return color3;
        //}
        return color2;
    }

    float accum = 0;

    @Override
    public void run(float dt)
    {
        if (uniformGrid == null) return;

        var x_max = uniformGrid.getX_origin() + uniformGrid.getWidth();
        var y_max = uniformGrid.getY_origin() + uniformGrid.getHeight();

        var deferred = new ArrayList<Pair<Float, Float>>();

        for (float i = uniformGrid.getX_origin(); i < x_max; i += uniformGrid.getX_spacing())
        {
            for (float j = uniformGrid.getY_origin(); j < y_max; j += uniformGrid.getY_spacing())
            {
                var c = getColor(i, j);
                if (c == color3)
                {
                    deferred.add(Pair.create(i, j));
                }
                else this.add(i, j, uniformGrid.getX_spacing(), uniformGrid.getY_spacing(), c);
            }
        }

        deferred.forEach(d->this.add(d.getLeft(), d.getRight(),
            uniformGrid.getX_spacing(), uniformGrid.getY_spacing(),color3));

        accum+=dt;

        this.add(uniformGrid.getX_origin(),
                uniformGrid.getY_origin(),
                uniformGrid.getWidth(),
                uniformGrid.getHeight(), color);
        render();
    }

    @Override
    public void shutdown()
    {

    }
}