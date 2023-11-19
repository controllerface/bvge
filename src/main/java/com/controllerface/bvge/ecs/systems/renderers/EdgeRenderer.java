package com.controllerface.bvge.ecs.systems.renderers;

import com.controllerface.bvge.Main;
import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.ecs.ECS;
import com.controllerface.bvge.ecs.systems.GameSystem;
import com.controllerface.bvge.gl.AbstractShader;
import com.controllerface.bvge.util.Assets;
import com.controllerface.bvge.util.Constants;
import com.controllerface.bvge.window.Window;

import static org.lwjgl.opengl.GL11C.GL_FLOAT;
import static org.lwjgl.opengl.GL15C.GL_ARRAY_BUFFER;
import static org.lwjgl.opengl.GL15C.GL_DYNAMIC_DRAW;
import static org.lwjgl.opengl.GL15C.GL_LINES;
import static org.lwjgl.opengl.GL15C.glBindBuffer;
import static org.lwjgl.opengl.GL15C.glBufferData;
import static org.lwjgl.opengl.GL15C.glDrawArrays;
import static org.lwjgl.opengl.GL15C.glGenBuffers;
import static org.lwjgl.opengl.GL20C.glDisableVertexAttribArray;
import static org.lwjgl.opengl.GL20C.glEnableVertexAttribArray;
import static org.lwjgl.opengl.GL20C.glVertexAttribPointer;
import static org.lwjgl.opengl.GL30C.glBindVertexArray;
import static org.lwjgl.opengl.GL30C.glGenVertexArrays;

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
        this.shader = Assets.shader("object_outline.glsl");
        init();
    }

    public void init()
    {
        vao_id = glGenVertexArrays();
        glBindVertexArray(vao_id);

        edge_vbo = glGenBuffers();
        glBindBuffer(GL_ARRAY_BUFFER, edge_vbo);
        glBufferData(GL_ARRAY_BUFFER, BATCH_BUFFER_SIZE, GL_DYNAMIC_DRAW);
        glVertexAttribPointer(0, VERTEX_SIZE, GL_FLOAT, false, VERTEX_SIZE_BYTES, 0);
        GPU.share_memory(edge_vbo);

        flag_vbo = glGenBuffers();
        glBindBuffer(GL_ARRAY_BUFFER, flag_vbo);
        glBufferData(GL_ARRAY_BUFFER, BATCH_FLAG_SIZE, GL_DYNAMIC_DRAW);
        glVertexAttribPointer(1, FLAG_SIZE, GL_FLOAT, false, FLAG_SIZE_BYTES, 0);
        GPU.share_memory(flag_vbo);

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
        glEnableVertexAttribArray(1);

        int offset = 0;
        for (int remaining = Main.Memory.edge_count(); remaining > 0; remaining -= Constants.Rendering.MAX_BATCH_SIZE)
        {
            int count = Math.min(Constants.Rendering.MAX_BATCH_SIZE, remaining);
            GPU.GL_edges(edge_vbo, flag_vbo, offset, count);
            glDrawArrays(GL_LINES, 0, count * 2);
            offset += count;
        }

        glDisableVertexAttribArray(0);
        glDisableVertexAttribArray(1);
        glBindVertexArray(0);

        shader.detach();
    }
}