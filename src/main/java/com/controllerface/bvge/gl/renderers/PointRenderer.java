package com.controllerface.bvge.gl.renderers;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.ecs.ECS;
import com.controllerface.bvge.ecs.systems.GameSystem;
import com.controllerface.bvge.gl.AbstractShader;
import com.controllerface.bvge.gl.GLUtils;
import com.controllerface.bvge.util.Assets;
import com.controllerface.bvge.util.Constants;
import com.controllerface.bvge.window.Window;

import static com.controllerface.bvge.util.Constants.Rendering.VECTOR_FLOAT_2D_SIZE;
import static org.lwjgl.opengl.GL15C.GL_POINTS;
import static org.lwjgl.opengl.GL15C.glDrawArrays;
import static org.lwjgl.opengl.GL30C.glBindVertexArray;
import static org.lwjgl.opengl.GL45C.*;

/**
 * Renders physics edge constraints. All defined edges are rendered as lines.
 */
public class PointRenderer extends GameSystem
{
    private static final int BATCH_BUFFER_SIZE = Constants.Rendering.MAX_BATCH_SIZE * VECTOR_FLOAT_2D_SIZE;

    private static final int POSITION_ATTRIBUTE = 0;

    private final AbstractShader shader;
    private int vao_id;
    private int point_vbo;

    public PointRenderer(ECS ecs)
    {
        super(ecs);
        this.shader = Assets.load_shader("point_shader.glsl");
        init();
    }

    public void init()
    {
        vao_id = glCreateVertexArrays();
        point_vbo = GLUtils.new_buffer_vec2(vao_id, POSITION_ATTRIBUTE, BATCH_BUFFER_SIZE);
        GPU.share_memory(point_vbo);
    }

    @Override
    public void tick(float dt)
    {
        glBindVertexArray(vao_id);

        shader.use();
        shader.uploadMat4f("uVP", Window.get().camera().get_uVP());

        glEnableVertexArrayAttrib(vao_id, POSITION_ATTRIBUTE);

        int offset = 0;
        for (int remaining = GPU.core_memory.next_point(); remaining > 0; remaining -= Constants.Rendering.MAX_BATCH_SIZE)
        {
            int count = Math.min(Constants.Rendering.MAX_BATCH_SIZE, remaining);
            GPU.GL_points(point_vbo, offset, count);
            glDrawArrays(GL_POINTS, 0, count);
            offset += count;
        }

        glDisableVertexArrayAttrib(vao_id, POSITION_ATTRIBUTE);

        glBindVertexArray(0);

        shader.detach();
    }
}