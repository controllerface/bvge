package com.controllerface.bvge.ecs.systems.renderers.batches;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.gl.AbstractShader;
import com.controllerface.bvge.util.Constants;
import com.controllerface.bvge.window.Window;

import static org.lwjgl.opengl.GL15.*;
import static org.lwjgl.opengl.GL20.glDisableVertexAttribArray;
import static org.lwjgl.opengl.GL20.glEnableVertexAttribArray;
import static org.lwjgl.opengl.GL30.glBindVertexArray;

/**
 * Rendering batch specifically for edge constraints
 */
public class BoundingBoxRenderBatch
{
    private int box_count;
    private int offset;
    private final int vaoID, vboID;
    private final AbstractShader currentShader;
    private final int[] offsets = new int[Constants.Rendering.MAX_BATCH_SIZE];;
    private final int[] counts = new int[Constants.Rendering.MAX_BATCH_SIZE];

    public BoundingBoxRenderBatch(AbstractShader currentShader, int vaoID, int vboID)
    {
        this.box_count = 0;
        this.currentShader = currentShader;
        this.vaoID = vaoID;
        this.vboID = vboID;
        for (int i = 0; i < Constants.Rendering.MAX_BATCH_SIZE; i++)
        {
            offsets[i] = i * 4;
            counts[i] = 4;
        }
    }

    public void clear()
    {
        box_count = 0;
    }

    public void setLineCount(int numLines)
    {
        this.box_count = numLines;
    }

    public void setOffset(int offset)
    {
        this.offset = offset;
    }

    public void render()
    {
        glBindVertexArray(vaoID);
        glBindBuffer(GL_ARRAY_BUFFER, vboID);

        GPU.GL_bounds(vboID, offset, box_count);

        // Use shader
        currentShader.use();
        currentShader.uploadMat4f("uProjection", Window.get().camera().getProjectionMatrix());
        currentShader.uploadMat4f("uView", Window.get().camera().getViewMatrix());

        glEnableVertexAttribArray(0);

        glMultiDrawArrays(GL_LINE_LOOP, offsets, counts);

        glDisableVertexAttribArray(0);
        glBindVertexArray(0);

        currentShader.detach();
    }
}
