package com.controllerface.bvge.ecs.systems.renderers;

import com.controllerface.bvge.Main;
import com.controllerface.bvge.cl.OpenCL;
import com.controllerface.bvge.ecs.ECS;
import com.controllerface.bvge.ecs.systems.GameSystem;
import com.controllerface.bvge.ecs.systems.renderers.batches.BoundingBoxRenderBatch;
import com.controllerface.bvge.gl.AbstractShader;
import com.controllerface.bvge.util.Assets;
import com.controllerface.bvge.util.Constants;

import java.util.ArrayList;
import java.util.List;

import static org.lwjgl.opengl.GL11.GL_FLOAT;
import static org.lwjgl.opengl.GL15.*;
import static org.lwjgl.opengl.GL20C.glVertexAttribPointer;
import static org.lwjgl.opengl.GL30.glBindVertexArray;
import static org.lwjgl.opengl.GL30.glGenVertexArrays;

/**
 * Manages rendering of edge constraints. All edges that are defined in the currently
 * loaded physics state are rendered as lines.
 */
public class BoundingBoxRenderer extends GameSystem
{
    private static final int VERTEX_SIZE = 2; // a vertex is 2 floats (x,y)
    private static final int VERTS_PER_BOX = 4;
    private static final int VERTEX_SIZE_BYTES = VERTEX_SIZE * Float.BYTES;
    private static final int BATCH_VERTEX_COUNT = Constants.Rendering.MAX_BATCH_SIZE * VERTS_PER_BOX * VERTEX_SIZE;
    private static final int BATCH_BUFFER_SIZE = BATCH_VERTEX_COUNT * Float.BYTES;

    private final AbstractShader shader;
    private final List<BoundingBoxRenderBatch> batches;
    private int vaoID, vboID;

    public BoundingBoxRenderer(ECS ecs)
    {
        super(ecs);
        this.batches = new ArrayList<>();
        this.shader = Assets.shader("bounding_outline.glsl");
        start();
    }

    public void start()
    {
        // Generate and bind a Vertex Array Object
        vaoID = glGenVertexArrays();
        glBindVertexArray(vaoID); // this sets this VAO as being active

        // generate and bind a Vertex Buffer Object, note that because our vao is currently active,
        // the new vbo is "attached" to it implicitly
        vboID = glGenBuffers();
        glBindBuffer(GL_ARRAY_BUFFER, vboID);
        // allocates enough space for the vertices we can handle in this batch, but doesn't transfer any data
        // into the buffer just yet
        glBufferData(GL_ARRAY_BUFFER, BATCH_BUFFER_SIZE, GL_DYNAMIC_DRAW);

        // share the buffer with the CL context
        OpenCL.share_memory(vboID);

        // define the buffer attribute pointers
        glVertexAttribPointer(0, VERTEX_SIZE, GL_FLOAT, false, VERTEX_SIZE_BYTES, 0);

        // unbind the vao since we're done defining things for now
        glBindVertexArray(0);
    }

    private void render()
    {
        for (BoundingBoxRenderBatch batch : batches)
        {
            batch.render();
            batch.clear();
        }
    }

    @Override
    public void run(float dt)
    {
        // todo: right now, this check only adds batches, never reducing them if the count goes
        //  low enough that some batches would be unneeded. This will leak memory resources
        //  so should be adjusted when deleting entities is added.

        var hull_count = Main.Memory.hull_count();
        var needed_batches = hull_count / Constants.Rendering.MAX_BATCH_SIZE;
        var r = hull_count % Constants.Rendering.MAX_BATCH_SIZE;
        if (r > 0)
        {
            needed_batches++;
        }
        while (needed_batches > batches.size())
        {
            var b = new BoundingBoxRenderBatch(shader, vaoID, vboID);
            batches.add(b);
        }

        int next = 0;
        int offset = 0;
        for (int i = hull_count; i > 0; i -= Constants.Rendering.MAX_BATCH_SIZE)
        {
            int count = Math.min(Constants.Rendering.MAX_BATCH_SIZE, i);
            var b = batches.get(next++);
            b.setLineCount(count);
            b.setOffset(offset);
            offset += count;
        }
        render();
    }

    @Override
    public void shutdown()
    {

    }
}