package com.controllerface.bvge.gl.renderers;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.ecs.ECS;
import com.controllerface.bvge.ecs.systems.GameSystem;
import com.controllerface.bvge.gl.AbstractShader;
import com.controllerface.bvge.util.Assets;
import com.controllerface.bvge.util.Constants;
import com.controllerface.bvge.window.Window;

import static com.controllerface.bvge.util.Constants.Rendering.VECTOR_4D_LENGTH;
import static org.lwjgl.opengl.GL11C.GL_FLOAT;
import static org.lwjgl.opengl.GL15C.*;
import static org.lwjgl.opengl.GL20C.*;
import static org.lwjgl.opengl.GL30C.glBindVertexArray;
import static org.lwjgl.opengl.GL30C.glGenVertexArrays;
import static org.lwjgl.opengl.GL45C.*;

/**
 * Renders physics edge constraints. All defined edges are rendered as lines.
 */
public class EdgeRenderer extends GameSystem
{
    private static final int VERTEX_SIZE = 2;
    private static final int VERTS_PER_EDGE = 2;
    private static final int VERTEX_SIZE_BYTES = VERTEX_SIZE * Float.BYTES;
    private static final int BATCH_VERTEX_COUNT = Constants.Rendering.MAX_BATCH_SIZE * VERTS_PER_EDGE * VERTEX_SIZE;
    private static final int BATCH_BUFFER_SIZE = BATCH_VERTEX_COUNT * Float.BYTES;

    private static final int FLAG_SIZE = 1;
    private static final int FLAGS_PER_EDGE = 2;
    private static final int FLAG_SIZE_BYTES = FLAG_SIZE * Float.BYTES;
    private static final int BATCH_FLAG_COUNT = Constants.Rendering.MAX_BATCH_SIZE * FLAGS_PER_EDGE * FLAG_SIZE;
    private static final int BATCH_FLAG_SIZE = BATCH_FLAG_COUNT * Float.BYTES;

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

        edge_vbo = glCreateBuffers();
        glNamedBufferData(edge_vbo, BATCH_BUFFER_SIZE, GL_DYNAMIC_DRAW);
        glVertexArrayVertexBuffer(vao_id, 0, edge_vbo, 0, VERTEX_SIZE_BYTES);
        glEnableVertexArrayAttrib(vao_id, 0);
        glVertexArrayAttribFormat(vao_id, 0, VERTEX_SIZE, GL_FLOAT, false, 0);
        glVertexArrayAttribBinding(vao_id, 0, 0);

        flag_vbo = glCreateBuffers();
        glNamedBufferData(flag_vbo, BATCH_FLAG_SIZE, GL_DYNAMIC_DRAW);
        glVertexArrayVertexBuffer(vao_id, 1, flag_vbo, 0, FLAG_SIZE_BYTES);
        glEnableVertexArrayAttrib(vao_id, 1);
        glVertexArrayAttribFormat(vao_id, 1, FLAG_SIZE, GL_FLOAT, false, 0);
        glVertexArrayAttribBinding(vao_id, 1, 1);

        GPU.share_memory(edge_vbo);
        GPU.share_memory(flag_vbo);
    }

    @Override
    public void tick(float dt)
    {
        glBindVertexArray(vao_id);

        shader.use();
        shader.uploadMat4f("uVP", Window.get().camera().get_uVP());

        glEnableVertexArrayAttrib(vao_id, 0);
        glEnableVertexArrayAttrib(vao_id, 1);

        int offset = 0;
        for (int remaining = GPU.Memory.next_edge(); remaining > 0; remaining -= Constants.Rendering.MAX_BATCH_SIZE)
        {
            int count = Math.min(Constants.Rendering.MAX_BATCH_SIZE, remaining);
            GPU.GL_edges(edge_vbo, flag_vbo, offset, count);
            glDrawArrays(GL_LINES, 0, count * 2);
            offset += count;
        }

        glBindVertexArray(0);

        shader.detach();
    }
}