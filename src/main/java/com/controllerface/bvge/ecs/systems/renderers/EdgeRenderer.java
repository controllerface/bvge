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
import static org.lwjgl.opengl.GL15C.*;
import static org.lwjgl.opengl.GL15C.GL_DYNAMIC_DRAW;
import static org.lwjgl.opengl.GL20C.glDisableVertexAttribArray;
import static org.lwjgl.opengl.GL20C.glEnableVertexAttribArray;
import static org.lwjgl.opengl.GL20C.glVertexAttribPointer;
import static org.lwjgl.opengl.GL30C.glBindVertexArray;
import static org.lwjgl.opengl.GL30C.glGenVertexArrays;

/**
 * Manages rendering of edge constraints. All edges that are defined in the currently
 * loaded physics state are rendered as lines.
 */
public class EdgeRenderer extends GameSystem
{
    private static final int VERTEX_SIZE = 2; // a vertex is 2 floats (x,y)
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
    private int vaoID;
    private int vertex_vbo;
    private int flag_vbo;

    public EdgeRenderer(ECS ecs)
    {
        super(ecs);
        this.shader = Assets.shader("object_outline.glsl");
        init();
    }

    public void init()
    {
        // Generate and bind a Vertex Array Object
        vaoID = glGenVertexArrays();
        glBindVertexArray(vaoID); // this sets this VAO as being active

        // generate and bind a Vertex Buffer Object, note that because our vao is currently active,
        // the new vbo is "attached" to it implicitly when it is bound
        vertex_vbo = glGenBuffers();
        glBindBuffer(GL_ARRAY_BUFFER, vertex_vbo);

        // allocates enough space for the vertices we can handle in this batch, but doesn't transfer any data
        // into the buffer just yet
        glBufferData(GL_ARRAY_BUFFER, BATCH_BUFFER_SIZE, GL_DYNAMIC_DRAW);

        // share the buffer with the CL context
        GPU.share_memory(vertex_vbo);

        // define the buffer attribute pointers
        glVertexAttribPointer(0, VERTEX_SIZE, GL_FLOAT, false, VERTEX_SIZE_BYTES, 0);

        // create buffer for edge flags, used to modify rendering output
        flag_vbo = glGenBuffers();
        glBindBuffer(GL_ARRAY_BUFFER, flag_vbo);
        glBufferData(GL_ARRAY_BUFFER, BATCH_FLAG_SIZE, GL_DYNAMIC_DRAW);
        GPU.share_memory(flag_vbo);
        glVertexAttribPointer(1, FLAG_SIZE, GL_FLOAT, false, FLAG_SIZE_BYTES, 0);

        // bind zero to unbind
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        // unbind the vao since we're done defining things
        glBindVertexArray(0);
    }

    private void render_batch(int offset, int edge_count)
    {
        GPU.GL_edges(vertex_vbo, flag_vbo, offset, edge_count);
        glDrawArrays(GL_LINES, 0, edge_count * 2);
    }

    @Override
    public void run(float dt)
    {
        var edge_count = Main.Memory.edge_count();

        glBindVertexArray(vaoID);

        // Use shader
        shader.use();
        shader.uploadMat4f("uProjection", Window.get().camera().getProjectionMatrix());
        shader.uploadMat4f("uView", Window.get().camera().getViewMatrix());

        glEnableVertexAttribArray(0);
        glEnableVertexAttribArray(1);

        int offset = 0;
        for (int i = edge_count; i > 0; i -= Constants.Rendering.MAX_BATCH_SIZE)
        {
            int count = Math.min(Constants.Rendering.MAX_BATCH_SIZE, i);
            render_batch(offset, count);
            offset += count;
        }

        glDisableVertexAttribArray(0);
        glDisableVertexAttribArray(1);
        glBindVertexArray(0);

        shader.detach();
    }
}