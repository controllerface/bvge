package com.controllerface.bvge.ecs.systems.renderers;

import com.controllerface.bvge.Main;
import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.ecs.ECS;
import com.controllerface.bvge.ecs.systems.GameSystem;
import com.controllerface.bvge.gl.AbstractShader;
import com.controllerface.bvge.util.Assets;
import com.controllerface.bvge.util.Constants;
import com.controllerface.bvge.window.Window;

import java.util.ArrayList;
import java.util.List;

import static org.lwjgl.opengl.GL11.GL_FLOAT;
import static org.lwjgl.opengl.GL15.*;
import static org.lwjgl.opengl.GL15.GL_DYNAMIC_DRAW;
import static org.lwjgl.opengl.GL20.glDisableVertexAttribArray;
import static org.lwjgl.opengl.GL20.glEnableVertexAttribArray;
import static org.lwjgl.opengl.GL20C.glVertexAttribPointer;
import static org.lwjgl.opengl.GL30.glBindVertexArray;
import static org.lwjgl.opengl.GL30.glGenVertexArrays;
import static org.lwjgl.opengl.GL33.glVertexAttribDivisor;

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
    private final List<EdgeRenderBatch> batches;
    private int vaoID;
    private int vboID;
    private int vboID2;

    private int last_edge_count = 0;

    public EdgeRenderer(ECS ecs)
    {
        super(ecs);
        this.batches = new ArrayList<>();
        this.shader = Assets.shader("object_outline.glsl");
        start();
    }

    public void start()
    {
        // Generate and bind a Vertex Array Object
        vaoID = glGenVertexArrays();
        glBindVertexArray(vaoID); // this sets this VAO as being active

        // generate and bind a Vertex Buffer Object, note that because our vao is currently active,
        // the new vbo is "attached" to it implicitly when it is bound
        vboID = glGenBuffers();
        glBindBuffer(GL_ARRAY_BUFFER, vboID);

        // allocates enough space for the vertices we can handle in this batch, but doesn't transfer any data
        // into the buffer just yet
        glBufferData(GL_ARRAY_BUFFER, BATCH_BUFFER_SIZE, GL_DYNAMIC_DRAW);

        // share the buffer with the CL context
        GPU.share_memory(vboID);

        // define the buffer attribute pointers
        glVertexAttribPointer(0, VERTEX_SIZE, GL_FLOAT, false, VERTEX_SIZE_BYTES, 0);

        // create buffer for edge flags, used to modify rendering output
        vboID2 = glGenBuffers();
        glBindBuffer(GL_ARRAY_BUFFER, vboID2);
        glBufferData(GL_ARRAY_BUFFER, BATCH_FLAG_SIZE, GL_DYNAMIC_DRAW);
        GPU.share_memory(vboID2);
        glVertexAttribPointer(1, FLAG_SIZE, GL_FLOAT, false, FLAG_SIZE_BYTES, 0);

        glBindBuffer(GL_ARRAY_BUFFER, 0);

        // unbind the vao since we're done defining things for now
        glBindVertexArray(0);
    }

    private void render()
    {
        for (EdgeRenderBatch batch : batches)
        {
            batch.render();
            batch.clear();
        }
    }


    @Override
    public void run(float dt)
    {
        var edge_count = Main.Memory.edge_count();

        if (edge_count != last_edge_count)
        {
            last_edge_count = edge_count;

            // calculate the 5total number of batches needed. If the number of edges does not
            // divide evenly into the number of edges per batch, one extra batch is created
            // that will render the remaining objects.
            var needed_batches = edge_count / Constants.Rendering.MAX_BATCH_SIZE;
            var r = edge_count % Constants.Rendering.MAX_BATCH_SIZE;
            if (r > 0)
            {
                needed_batches++;
            }

            // create more batches if needed
            while (needed_batches > batches.size())
            {
                var b = new EdgeRenderBatch(shader, vaoID, vboID, vboID2);
                batches.add(b);
            }

            // remove excess batches if needed
            while (batches.size() > needed_batches)
            {
                batches.remove(0);
            }
        }

        int next = 0;
        int offset = 0;
        for (int i = edge_count; i > 0; i -= Constants.Rendering.MAX_BATCH_SIZE)
        {
            int count = Math.min(Constants.Rendering.MAX_BATCH_SIZE, i);
            var b = batches.get(next++);
            b.setLineCount(count);
            b.setOffset(offset);
            offset += count;
        }

        render();
    }

    private static class EdgeRenderBatch
    {
        private int edge_count;
        private int offset;
        private final int vaoID;
        private final int vboID;
        private final int vboID2;
        private final AbstractShader shader;

        public EdgeRenderBatch(AbstractShader shader, int vaoID, int vboID, int vboID2)
        {
            this.edge_count = 0;
            this.shader = shader;
            this.vaoID = vaoID;
            this.vboID = vboID;
            this.vboID2 = vboID2;
        }

        public void clear()
        {
            edge_count = 0;
        }

        public void setLineCount(int numLines)
        {
            this.edge_count = numLines;
        }

        public void setOffset(int offset)
        {
            this.offset = offset;
        }

        public void render()
        {
            if (edge_count == 0) return;

            glBindVertexArray(vaoID);

            GPU.GL_edges(vboID, vboID2, offset, edge_count);

            // Use shader
            shader.use();
            shader.uploadMat4f("uProjection", Window.get().camera().getProjectionMatrix());
            shader.uploadMat4f("uView", Window.get().camera().getViewMatrix());

            glEnableVertexAttribArray(0);
            glEnableVertexAttribArray(1);
            glDrawArrays(GL_LINES, 0, edge_count * 2);
            glDisableVertexAttribArray(0);
            glDisableVertexAttribArray(1);
            glBindVertexArray(0);

            shader.detach();
        }
    }
}