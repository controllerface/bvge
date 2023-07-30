package com.controllerface.bvge.ecs.systems.renderers.batches;

import com.controllerface.bvge.cl.OpenCL;
import com.controllerface.bvge.gl.Shader;
import com.controllerface.bvge.window.Window;

import static org.lwjgl.opengl.GL15.*;
import static org.lwjgl.opengl.GL20.*;
import static org.lwjgl.opengl.GL30.glBindVertexArray;
import static org.lwjgl.opengl.GL30.glGenVertexArrays;
import static org.lwjgl.opengl.GL31.glDrawArraysInstanced;
import static org.lwjgl.opengl.GL33.glVertexAttribDivisor;

public class BoxRenderBatch
{
    private static final int VERTEX_SIZE = 2; // a vertex is 2 floats (x,y)
    private static final int VERTEX_SIZE_BYTES = VERTEX_SIZE * Float.BYTES;

    private static final int TRANSFORM_SIZE = 2; // a transform is 3 floats (x,y,w)
    private static final int TRANSFORM_SIZE_BYTES = TRANSFORM_SIZE * Float.BYTES;

    private int numModels;
    private int vaoID, vboID;
    private final int tranform_buffer_ID, model_buffer_id;
    private final Shader currentShader;
    private int[] indices; // will be large enough to hold a full batch, but may only contain a partial one

    public BoxRenderBatch(Shader currentShader,
                          int tranform_buffer_ID,
                          int model_buffer_id)
    {
        //this.numModels = numModels;
        this.currentShader = currentShader;
        this.tranform_buffer_ID = tranform_buffer_ID;
        this.model_buffer_id = model_buffer_id;
        //this.indices = indices;
        //start();
    }

    public void start()
    {
        // Generate and bind a Vertex Array Object
        vaoID = glGenVertexArrays();
        glBindVertexArray(vaoID); // this sets this VAO as being active

        vboID = glGenBuffers();
        glBindBuffer(GL_ARRAY_BUFFER, vboID);
        glBufferData(GL_ARRAY_BUFFER, indices, GL_STATIC_DRAW);

        // load model data and configure the vertex attributes
        glBindBuffer(GL_ARRAY_BUFFER, model_buffer_id);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, VERTEX_SIZE, GL_FLOAT, false, VERTEX_SIZE_BYTES, 0);

        // create buffer for transforms and configure the instance divisor
        glBindBuffer(GL_ARRAY_BUFFER, tranform_buffer_ID); // this attribute comes from a different vertex buffer
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, TRANSFORM_SIZE, GL_FLOAT, false, TRANSFORM_SIZE_BYTES, 0);
        glVertexAttribDivisor(1, 1);

        // share the buffer with the CL context
        OpenCL.share_memory(vboID);

        // unbind
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        // unbind the vao since we're done defining things for now
        glBindVertexArray(0);
    }

    public void clear()
    {
        numModels = 0;
    }

    public void setIndices(int[] indices)
    {
        this.indices = indices;
    }

    public void setModelCount(int numModels)
    {
        this.numModels = numModels;
    }


    public void render()
    {
        currentShader.use();
        currentShader.uploadMat4f("uProjection", Window.get().camera().getProjectionMatrix());
        currentShader.uploadMat4f("uView", Window.get().camera().getViewMatrix());

        glBindVertexArray(vaoID);

        OpenCL.batch_transforms_GL(vboID, tranform_buffer_ID, numModels);

        glEnableVertexAttribArray(0);
        glEnableVertexAttribArray(1);
        glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, numModels);
        glDisableVertexAttribArray(0);
        glDisableVertexAttribArray(1);
        glBindVertexArray(0);
        currentShader.detach();
    }
}
