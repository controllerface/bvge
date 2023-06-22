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

public class QuadTreeRendering extends GameSystem
{
    private Shader shader;
    private List<BoxRenderBatch> batches;
    private final Vector3f color = new Vector3f(1.0f,0,0);

    public QuadTreeRendering(ECS ecs)
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

    private QuadTree<RigidBody2D> quadTree;

    public void setQuadTree(QuadTree<RigidBody2D> quadTree)
    {
        this.quadTree = quadTree;
    }

    @Override
    public void run(float dt)
    {
        if (quadTree == null) return;

        var rects = new ArrayList<QuadRectangle>();
        quadTree.getAllZones(rects);
        for (QuadRectangle r : rects)
        {
            this.add(r);
        }
        render();
    }
}