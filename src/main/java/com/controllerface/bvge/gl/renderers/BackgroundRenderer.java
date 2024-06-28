package com.controllerface.bvge.gl.renderers;

import com.controllerface.bvge.ecs.ECS;
import com.controllerface.bvge.ecs.systems.GameSystem;
import com.controllerface.bvge.gl.GLUtils;
import com.controllerface.bvge.gl.Shader;
import com.controllerface.bvge.gl.Texture;
import com.controllerface.bvge.util.Assets;
import com.controllerface.bvge.window.Window;

import static org.lwjgl.opengl.GL15C.glDrawArrays;
import static org.lwjgl.opengl.GL30C.glBindVertexArray;
import static org.lwjgl.opengl.GL45C.*;

public class BackgroundRenderer extends GameSystem
{
    private static final int POSITION_ATTRIBUTE = 0;
    private static final int UV_ATTRIBUTE = 1;

    private final int[] texture_slots = { 0 };

    private int vao;
    private int position_vbo;
    private int uv_vbo;

    private Texture texture;
    private Shader shader;

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

        texture = new Texture();
        texture.init("/img/cave_bg.png");
        shader = Assets.load_shader("bg_shader.glsl");
        shader.uploadIntArray("uTextures", texture_slots);
        vao = glCreateVertexArrays();
        position_vbo = GLUtils.fill_buffer_vec2(vao, POSITION_ATTRIBUTE, vertices);
        uv_vbo = GLUtils.fill_buffer_vec2(vao, UV_ATTRIBUTE, uvs);
        glEnableVertexArrayAttrib(vao, POSITION_ATTRIBUTE);
        glEnableVertexArrayAttrib(vao, UV_ATTRIBUTE);
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
        shader.destroy();
    }
}