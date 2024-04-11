package com.controllerface.bvge.gl.renderers;

import com.controllerface.bvge.ecs.ECS;
import com.controllerface.bvge.ecs.systems.GameSystem;
import com.controllerface.bvge.gl.AbstractShader;
import com.controllerface.bvge.gl.Texture;
import com.controllerface.bvge.util.Assets;
import com.controllerface.bvge.util.Constants;

import static com.controllerface.bvge.util.Constants.Rendering.VECTOR_4D_LENGTH;

public class BlockRenderer extends GameSystem
{
    private static final int TRANSFORM_VERTEX_COUNT = Constants.Rendering.MAX_BATCH_SIZE * VECTOR_4D_LENGTH;
    private static final int TRANSFORM_BUFFER_SIZE = TRANSFORM_VERTEX_COUNT * Float.BYTES;
    private static final int POSITION_ATTRIBUTE = 0;
    private static final int UV_COORD_ATTRIBUTE = 1;
    private static final int TRANSFORM_ATTRIBUTE = 2;

    private final int[] texture_slots = {0};

    private final AbstractShader shader;
    private final Texture texture;

    public BlockRenderer(ECS ecs)
    {
        super(ecs);
        this.shader = Assets.load_shader("box_model.glsl");
        this.texture = new Texture();
    }

    @Override
    public void tick(float dt)
    {

    }

    @Override
    public void shutdown()
    {
        shader.destroy();
    }
}
