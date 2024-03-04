package com.controllerface.bvge.gl.renderers;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.ecs.ECS;
import com.controllerface.bvge.ecs.systems.GameSystem;
import com.controllerface.bvge.gl.AbstractShader;
import com.controllerface.bvge.gl.GLUtils;
import com.controllerface.bvge.util.Assets;
import com.controllerface.bvge.util.Constants;
import com.controllerface.bvge.window.Window;

import static com.controllerface.bvge.util.Constants.Rendering.*;
import static org.lwjgl.opengl.GL15C.GL_LINES;
import static org.lwjgl.opengl.GL15C.glDrawArrays;
import static org.lwjgl.opengl.GL30C.glBindVertexArray;
import static org.lwjgl.opengl.GL45C.*;
import static org.lwjgl.opengl.GL45C.glDisableVertexArrayAttrib;

/**
 * Renders physics edge constraints. All defined edges are rendered as lines.
 */
public class EdgeRenderer extends GameSystem
{
    private static final int DATA_POINTS_PER_EDGE = 2;

    private static final int BATCH_VERTEX_COUNT = Constants.Rendering.MAX_BATCH_SIZE * DATA_POINTS_PER_EDGE * VECTOR_2D_LENGTH;
    private static final int BATCH_BUFFER_SIZE = BATCH_VERTEX_COUNT * Float.BYTES;

    private static final int BATCH_FLAG_COUNT = Constants.Rendering.MAX_BATCH_SIZE * DATA_POINTS_PER_EDGE * SCALAR_LENGTH;
    private static final int BATCH_FLAG_SIZE = BATCH_FLAG_COUNT * Float.BYTES;

    private static final int EDGE_ATTRIBUTE = 0;
    private static final int FLAG_ATTRIBUTE = 1;

    private final AbstractShader shader;
    private int vao_id;
    private int edge_vbo;
    private int flag_vbo;


    public EdgeRenderer(ECS ecs)
    {
        super(ecs);
        this.shader = Assets.load_shader("object_outline.glsl");
        init();
    }

    public void init()
    {
        vao_id = glCreateVertexArrays();
        edge_vbo = GLUtils.new_buffer_vec2(vao_id, EDGE_ATTRIBUTE, BATCH_BUFFER_SIZE);
        flag_vbo = GLUtils.new_buffer_float(vao_id, FLAG_ATTRIBUTE, BATCH_FLAG_SIZE);
        GPU.share_memory(edge_vbo);
        GPU.share_memory(flag_vbo);
    }

    @Override
    public void tick(float dt)
    {
        glBindVertexArray(vao_id);

        shader.use();
        shader.uploadMat4f("uVP", Window.get().camera().get_uVP());

        glEnableVertexArrayAttrib(vao_id, EDGE_ATTRIBUTE);
        glEnableVertexArrayAttrib(vao_id, FLAG_ATTRIBUTE);

        int offset = 0;
        for (int remaining = GPU.Memory.next_edge(); remaining > 0; remaining -= Constants.Rendering.MAX_BATCH_SIZE)
        {
            int count = Math.min(Constants.Rendering.MAX_BATCH_SIZE, remaining);
            GPU.GL_edges(edge_vbo, flag_vbo, offset, count);
            glDrawArrays(GL_LINES, 0, count * 2);
            offset += count;
        }

        glDisableVertexArrayAttrib(vao_id, EDGE_ATTRIBUTE);
        glDisableVertexArrayAttrib(vao_id, FLAG_ATTRIBUTE);

        glBindVertexArray(0);

        shader.detach();
    }
}