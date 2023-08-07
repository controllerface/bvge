package com.controllerface.bvge.ecs.systems.renderers.batches;

import com.controllerface.bvge.cl.OpenCL;
import com.controllerface.bvge.gl.AbstractShader;
import com.controllerface.bvge.gl.Texture;
import com.controllerface.bvge.window.Window;

import static com.controllerface.bvge.util.Constants.Rendering.*;
import static org.lwjgl.opengl.GL15.*;
import static org.lwjgl.opengl.GL20.*;
import static org.lwjgl.opengl.GL30.glBindVertexArray;
import static org.lwjgl.opengl.GL30.glGenVertexArrays;
import static org.lwjgl.opengl.GL31.glDrawArraysInstanced;
import static org.lwjgl.opengl.GL33.glVertexAttribDivisor;

public class CrateRenderBatch
{
    private final AbstractShader shader;
    private final Texture texture;
    private int vao;
    private int mesh_count;
    private int index_buffer_id;
    private final int transform_buffer_ID;
    private final int model_buffer_id;
    private final int texture_uv_buffer_id;
    private final int color_buffer_id;
    private final int[] texture_slots = { 0 };
    private int[] indices; // will be large enough to hold a full batch, but may only contain a partial one

    public CrateRenderBatch(AbstractShader shader,
                            Texture texture,
                            int transform_buffer_ID,
                            int model_buffer_id,
                            int texture_uv_buffer_id,
                            int color_buffer_id)
    {
        this.shader = shader;
        this.texture = texture;
        this.transform_buffer_ID = transform_buffer_ID;
        this.model_buffer_id = model_buffer_id;
        this.texture_uv_buffer_id = texture_uv_buffer_id;
        this.color_buffer_id = color_buffer_id;
    }

    public void start()
    {
        // Generate and bind a Vertex Array Object
        vao = glGenVertexArrays();
        glBindVertexArray(vao); // this sets this VAO as being active

        index_buffer_id = glGenBuffers();
        glBindBuffer(GL_ARRAY_BUFFER, index_buffer_id);
        glBufferData(GL_ARRAY_BUFFER, indices, GL_STATIC_DRAW);

        glBindBuffer(GL_ARRAY_BUFFER, model_buffer_id);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, VECTOR_2D_LENGTH, GL_FLOAT, false, VECTOR_FLOAT_2D_SIZE, 0);

        glBindBuffer(GL_ARRAY_BUFFER, transform_buffer_ID);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, VECTOR_4D_LENGTH, GL_FLOAT, false, VECTOR_FLOAT_4D_SIZE, 0);
        glVertexAttribDivisor(1, 1);

        glBindBuffer(GL_ARRAY_BUFFER, texture_uv_buffer_id);
        glEnableVertexAttribArray(2);
        glVertexAttribPointer(2, VECTOR_2D_LENGTH, GL_FLOAT, false, VECTOR_FLOAT_2D_SIZE, 0);

        glBindBuffer(GL_ARRAY_BUFFER, color_buffer_id);
        glEnableVertexAttribArray(3);
        glVertexAttribPointer(3, VECTOR_4D_LENGTH, GL_FLOAT, false, VECTOR_FLOAT_4D_SIZE, 0);

        // share the buffer with the CL context
        OpenCL.share_memory(index_buffer_id);

        // unbind
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        // unbind the vao since we're done defining things for now
        glBindVertexArray(0);
    }

    public void clear()
    {
        mesh_count = 0;
    }

    public void setIndices(int[] indices)
    {
        this.indices = indices;
    }

    public void setModelCount(int numModels)
    {
        this.mesh_count = numModels;
    }


    public void render()
    {
        shader.use();
        texture.bind();

        shader.uploadMat4f("uProjection", Window.get().camera().getProjectionMatrix());
        shader.uploadMat4f("uView", Window.get().camera().getViewMatrix());
        shader.uploadIntArray("uTextures", texture_slots);

        OpenCL.GL_transforms(index_buffer_id, transform_buffer_ID, mesh_count);

        glBindVertexArray(vao);
        glActiveTexture(GL_TEXTURE0);
        glEnableVertexAttribArray(0);
        glEnableVertexAttribArray(1);
        glEnableVertexAttribArray(2);
        glEnableVertexAttribArray(3);

        glDrawArraysInstanced(GL_TRIANGLES, 0, 6, mesh_count);

        glDisableVertexAttribArray(0);
        glDisableVertexAttribArray(1);
        glDisableVertexAttribArray(2);
        glDisableVertexAttribArray(3);
        glBindVertexArray(0);

        shader.detach();
        texture.unbind();
    }
}
