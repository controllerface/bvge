package com.controllerface.bvge.ecs.systems.renderers.batches;

import com.controllerface.bvge.cl.OpenCL;
import com.controllerface.bvge.gl.AbstractShader;
import com.controllerface.bvge.gl.Shader;
import com.controllerface.bvge.gl.Texture;
import com.controllerface.bvge.window.Window;

import static com.controllerface.bvge.util.Constants.Rendering.*;
import static org.lwjgl.opengl.GL15.*;
import static org.lwjgl.opengl.GL20.*;
import static org.lwjgl.opengl.GL30.glBindVertexArray;
import static org.lwjgl.opengl.GL30.glGenVertexArrays;
import static org.lwjgl.opengl.GL31.glDrawArraysInstanced;
import static org.lwjgl.opengl.GL33.glVertexAttribDivisor;

public class CircleRenderBatch
{
    private final AbstractShader shader;
    private int numModels;
    private int vaoID, vboID;
    private final int transform_buffer_ID;
    private int[] indices; // will be large enough to hold a full batch, but may only contain a partial one

    public CircleRenderBatch(AbstractShader shader, int transform_buffer_ID)
    {
        this.shader = shader;
        this.transform_buffer_ID = transform_buffer_ID;
    }

    public void start()
    {
        // Generate and bind a Vertex Array Object
        vaoID = glGenVertexArrays();
        glBindVertexArray(vaoID); // this sets this VAO as being active

        vboID = glGenBuffers();
        glBindBuffer(GL_ARRAY_BUFFER, vboID);
        glBufferData(GL_ARRAY_BUFFER, indices, GL_STATIC_DRAW);

        glBindBuffer(GL_ARRAY_BUFFER, transform_buffer_ID);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, VECTOR_4D_LENGTH, GL_FLOAT, false, VECTOR_FLOAT_4D_SIZE, 0);
        glVertexAttribDivisor(0, 1);

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
        glPointSize(10f);
        shader.use();
        shader.uploadMat4f("uProjection", Window.get().camera().getProjectionMatrix());
        shader.uploadMat4f("uView", Window.get().camera().getViewMatrix());

        OpenCL.batch_transforms_GL(vboID, transform_buffer_ID, numModels);

        glBindVertexArray(vaoID);

        glEnableVertexAttribArray(0);
        glDrawArrays(GL_POINTS, 0, numModels);
        glDisableVertexAttribArray(0);
        glBindVertexArray(0);
        shader.detach();
    }
}
