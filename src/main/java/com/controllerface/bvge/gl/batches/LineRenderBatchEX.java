package com.controllerface.bvge.gl.batches;

import com.controllerface.bvge.cl.OpenCL;
import com.controllerface.bvge.gl.Shader;
import com.controllerface.bvge.util.Constants;
import com.controllerface.bvge.window.Window;

import static org.lwjgl.opengl.GL15.*;
import static org.lwjgl.opengl.GL20.glDisableVertexAttribArray;
import static org.lwjgl.opengl.GL20.glEnableVertexAttribArray;
import static org.lwjgl.opengl.GL20C.glVertexAttribPointer;
import static org.lwjgl.opengl.GL30.glBindVertexArray;
import static org.lwjgl.opengl.GL30.glGenVertexArrays;

/**
 * Rendering batch specifically for sprites
 */
public class LineRenderBatchEX implements Comparable<LineRenderBatchEX>
{
    private static final int VERTEX_SIZE = 2; // a vertex is 2 floats (x,y)
    private static final int VERTS_PER_LINE = 2;
    private static final int VERTEX_SIZE_BYTES = VERTEX_SIZE * Float.BYTES;
    private static final int BATCH_VERTEX_COUNT = Constants.Rendering.MAX_BATCH_SIZE * VERTS_PER_LINE * VERTEX_SIZE;
    private static final int BATCH_BUFFER_SIZE = BATCH_VERTEX_COUNT * Float.BYTES;

    private int numLines;
    private int offset;

    // todo: try and make two vbo's and avoid needed to deal with offsets when defining attribute
    //  pointers.
    private int vaoID, vboID;

    private int zIndex;

    private final Shader currentShader;

    public LineRenderBatchEX(int zIndex, Shader currentShader)
    {
        this.zIndex = zIndex;
        this.numLines = 0;
        this.currentShader = currentShader;
    }

    public void clear()
    {
        numLines = 0;
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

        // bind the buffer to the CL context
        OpenCL.bindEdgeVBO(vboID);

        // define the buffer attribute pointers
        glVertexAttribPointer(0, VERTEX_SIZE, GL_FLOAT, false, VERTEX_SIZE_BYTES, 0);

        // unbind the vao since we're done defining things for now
        glBindVertexArray(0);
    }

    public void setLineCount(int numLines)
    {
        this.numLines = numLines;
    }

    public void setOffset(int offset)
    {
        this.offset = offset;
    }

    public void render()
    {
        glLineWidth(1.5f);
        glBindVertexArray(vaoID);
        glBindBuffer(GL_ARRAY_BUFFER, vboID);

        OpenCL.batchVbo(vboID, offset, numLines);

        // Use shader
        currentShader.use();
        currentShader.uploadMat4f("uProjection", Window.get().camera().getProjectionMatrix());
        currentShader.uploadMat4f("uView", Window.get().camera().getViewMatrix());

        glEnableVertexAttribArray(0);
        glEnableVertexAttribArray(1);
        glDrawArrays(GL_LINES, 0, numLines * 2 * 2);
        glDisableVertexAttribArray(0);
        glDisableVertexAttribArray(1);
        glBindVertexArray(0);

        currentShader.detach();
    }

    public int zIndex()
    {
        return this.zIndex;
    }

    @Override
    public int compareTo(LineRenderBatchEX o)
    {
        return Integer.compare(this.zIndex, o.zIndex());
    }
}
