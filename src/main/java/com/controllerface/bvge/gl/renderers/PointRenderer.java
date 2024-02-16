package com.controllerface.bvge.gl.renderers;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.ecs.ECS;
import com.controllerface.bvge.ecs.systems.GameSystem;
import com.controllerface.bvge.gl.AbstractShader;
import com.controllerface.bvge.util.Assets;
import com.controllerface.bvge.util.Constants;
import com.controllerface.bvge.window.Window;

import static org.lwjgl.opengl.GL11C.GL_FLOAT;
import static org.lwjgl.opengl.GL15C.*;
import static org.lwjgl.opengl.GL20C.*;
import static org.lwjgl.opengl.GL30C.glBindVertexArray;
import static org.lwjgl.opengl.GL30C.glGenVertexArrays;

/**
 * Renders physics edge constraints. All defined edges are rendered as lines.
 */
public class PointRenderer extends GameSystem
{
    private static final int VERTEX_SIZE = 2;
    private static final int VERTEX_SIZE_BYTES = Float.BYTES * VERTEX_SIZE;
    private static final int BATCH_VERTEX_COUNT = Constants.Rendering.MAX_BATCH_SIZE;
    private static final int BATCH_BUFFER_SIZE = BATCH_VERTEX_COUNT * VERTEX_SIZE_BYTES;

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
        vao_id = glGenVertexArrays();
        glBindVertexArray(vao_id);

        point_vbo = glGenBuffers();
        glBindBuffer(GL_ARRAY_BUFFER, point_vbo);
        glBufferData(GL_ARRAY_BUFFER, BATCH_BUFFER_SIZE, GL_DYNAMIC_DRAW);
        glVertexAttribPointer(0, VERTEX_SIZE, GL_FLOAT, false, VERTEX_SIZE_BYTES, 0);
        GPU.share_memory(point_vbo);

        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindVertexArray(0);
    }

    @Override
    public void tick(float dt)
    {
        glBindVertexArray(vao_id);

        shader.use();
        shader.uploadMat4f("uVP", Window.get().camera().get_uVP());

        glEnableVertexAttribArray(0);

        int offset = 0;
        for (int remaining = GPU.Memory.next_point(); remaining > 0; remaining -= Constants.Rendering.MAX_BATCH_SIZE)
        {
            int count = Math.min(Constants.Rendering.MAX_BATCH_SIZE, remaining);
            GPU.GL_points(point_vbo, offset, count);
            glDrawArrays(GL_POINTS, 0, count);
            offset += count;
        }

        glDisableVertexAttribArray(0);
        glBindVertexArray(0);

        shader.detach();
    }
}