package com.controllerface.bvge.gpu.gl.renderers;

import com.controllerface.bvge.ecs.ECS;
import com.controllerface.bvge.ecs.GameSystem;
import com.controllerface.bvge.gpu.GPU;
import com.controllerface.bvge.gpu.gl.GLUtils;
import com.controllerface.bvge.gpu.gl.shaders.GL_Shader;
import com.controllerface.bvge.gpu.gl.shaders.GL_ShaderType;
import com.controllerface.bvge.gpu.gl.textures.GL_Texture2D;

import static org.lwjgl.opengl.GL15C.glDrawArrays;
import static org.lwjgl.opengl.GL30C.glBindVertexArray;
import static org.lwjgl.opengl.GL45C.*;

public class BackgroundRenderer extends GameSystem
{
    private static final int POSITION_ATTRIBUTE = 0;
    private static final int UV_ATTRIBUTE = 1;

    private int vao;
    private int position_vbo;
    private int uv_vbo;

    private GL_Texture2D texture;
    private GL_Shader shader;

    public BackgroundRenderer(ECS ecs)
    {
        super(ecs);
        init_GL();
    }

    private void init_GL()
    {
        float[] vertices = new float[]
            {
                -1.0f, -1.0f,
                 1.0f, -1.0f,
                 1.0f,  1.0f,
                -1.0f, -1.0f,
                 1.0f,  1.0f,
                -1.0f,  1.0f,
            };

        float[] uvs = new float[]
            {
                0.0f, 0.0f,
                1.0f, 0.0f,
                1.0f, 1.0f,
                0.0f, 0.0f,
                1.0f, 1.0f,
                0.0f, 1.0f,
            };

        texture = new GL_Texture2D();
        texture.init("/img/cave_bg.png");
        vao = glCreateVertexArrays();

        // these "old style" calls prevent the shader from being re-compiled on first use. Not really
        // needed tbh, but remove a perf warning message from debug output, so leaving in for now to
        // reduce noise while debugging.
        glEnableVertexAttribArray(POSITION_ATTRIBUTE);
        glEnableVertexAttribArray(UV_ATTRIBUTE);

        glEnableVertexArrayAttrib(vao, POSITION_ATTRIBUTE);
        glEnableVertexArrayAttrib(vao, UV_ATTRIBUTE);
        shader = GPU.GL.new_shader("bg_shader.glsl", GL_ShaderType.TWO_STAGE);
        shader.uploadInt("uTexture", 0);
        position_vbo = GLUtils.fill_buffer_vec2(vao, POSITION_ATTRIBUTE, vertices);
        uv_vbo = GLUtils.fill_buffer_vec2(vao, UV_ATTRIBUTE, uvs);
    }


    @Override
    public void tick(float dt)
    {
        glBindVertexArray(vao);
        shader.use();
        texture.bind(0);
        glDrawArrays(GL_TRIANGLES, 0, 6);
        glBindVertexArray(0);
        shader.detach();
    }

    @Override
    public void shutdown()
    {
        glDeleteVertexArrays(vao);
        glDeleteBuffers(position_vbo);
        glDeleteBuffers(uv_vbo);
        shader.release();
    }
}