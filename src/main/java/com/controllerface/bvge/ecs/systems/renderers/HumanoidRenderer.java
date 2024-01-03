package com.controllerface.bvge.ecs.systems.renderers;

import com.controllerface.bvge.ecs.ECS;
import com.controllerface.bvge.ecs.systems.GameSystem;

public class HumanoidRenderer extends GameSystem
{
    // todo: determine the sizes required for a single render batch and calculate them


    public HumanoidRenderer(ECS ecs)
    {
        super(ecs);
        // todo: load a shader to render the meshes. The crate shader can be used a base,
        //  but the color data should be removed.
        init();
    }

    private void init()
    {
        // todo: load the humanoid model data and texture. model vertices will be tightly
        //  packed into a vbo and referenced by offset for indirect draw calls.

        // todo: setup buffers for drawing a batch. There will need to be a transform buffer,
        //  that is read from for every mesh instance, a bone buffer that is read for every
        //  vertex, texture coordinates, and of course the vertex position
        //
        // todo: maybe eliminate the bone data by doing the bone adjustment in a CL kernel
        //  before hand,
    }

    @Override
    public void tick(float dt)
    {
        // todo: get the count of all individual meshes
    }

}
