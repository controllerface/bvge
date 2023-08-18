package com.controllerface.bvge.ecs.systems.renderers.batches;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.gl.AbstractShader;
import com.controllerface.bvge.window.Window;

import static org.lwjgl.opengl.GL15.*;
import static org.lwjgl.opengl.GL20.glDisableVertexAttribArray;
import static org.lwjgl.opengl.GL20.glEnableVertexAttribArray;
import static org.lwjgl.opengl.GL30.glBindVertexArray;

/**
 * Rendering batch specifically for edge constraints
 */
public class BoneRenderBatch
{
    private int numBones;
    private int offset;
    private final int vaoID, vboID;
    private final AbstractShader currentShader;

    public BoneRenderBatch(AbstractShader currentShader, int vaoID, int vboID)
    {
        this.numBones = 0;
        this.currentShader = currentShader;
        this.vaoID = vaoID;
        this.vboID = vboID;
    }

    public void clear()
    {
        numBones = 0;
    }

    public void setLineCount(int numLines)
    {
        this.numBones = numLines;
    }

    public void setOffset(int offset)
    {
        this.offset = offset;
    }

    public void render()
    {
        glBindVertexArray(vaoID);
        glBindBuffer(GL_ARRAY_BUFFER, vboID);

        GPU.GL_bones(vboID, offset, numBones);

        // Use shader
        currentShader.use();
        currentShader.uploadMat4f("uProjection", Window.get().camera().getProjectionMatrix());
        currentShader.uploadMat4f("uView", Window.get().camera().getViewMatrix());

        glEnableVertexAttribArray(0);
        glDrawArrays(GL_POINTS, 0, numBones);
        glDisableVertexAttribArray(0);
        glBindVertexArray(0);

        currentShader.detach();
    }
}
