package com.controllerface.bvge.ecs.systems.renderers;

import com.controllerface.bvge.ecs.ECS;
import com.controllerface.bvge.ecs.systems.GameSystem;
import com.controllerface.bvge.ecs.systems.renderers.batches.BoxRenderBatch;
import com.controllerface.bvge.gl.Shader;
import com.controllerface.bvge.util.AssetPool;

import java.util.ArrayList;
import java.util.List;

public class BoxRenderer extends GameSystem
{
    private final Shader shader;
    private final List<BoxRenderBatch> batches;

    public BoxRenderer(ECS ecs)
    {
        super(ecs);
        this.batches = new ArrayList<>();
        this.shader = AssetPool.getShader("object_outline.glsl"); // todo: need new shader program
    }

    @Override
    public void run(float dt)
    {

    }

    @Override
    public void shutdown()
    {

    }
}
