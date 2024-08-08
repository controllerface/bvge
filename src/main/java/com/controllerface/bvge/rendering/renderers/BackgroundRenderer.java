package com.controllerface.bvge.rendering.renderers;

import com.controllerface.bvge.ecs.ECS;
import com.controllerface.bvge.ecs.GameSystem;
import com.controllerface.bvge.gpu.GPU;
import com.controllerface.bvge.gpu.gl.buffers.GL_VertexArray;
import com.controllerface.bvge.gpu.gl.buffers.GL_VertexBuffer;
import com.controllerface.bvge.gpu.gl.shaders.GL_Shader;
import com.controllerface.bvge.gpu.gl.shaders.GL_ShaderType;
import com.controllerface.bvge.gpu.gl.textures.GL_Texture2D;

import static org.lwjgl.opengl.GL15C.glDrawArrays;
import static org.lwjgl.opengl.GL45C.GL_TRIANGLE_STRIP;

public class BackgroundRenderer extends GameSystem
{
    private static final int XY_ATTRIBUTE = 0;
    private static final int UV_ATTRIBUTE = 1;

    private final GL_VertexArray vao;
    private final GL_VertexBuffer xy_vbo;
    private final GL_VertexBuffer uv_vbo;
    private final GL_Texture2D texture;
    private final GL_Shader shader;

    public BackgroundRenderer(ECS ecs)
    {
        super(ecs);
        vao     = GPU.GL.new_vao();
        texture = GPU.GL.new_texture("/img/cave_bg.png");
        shader  = GPU.GL.new_shader("bg_shader.glsl", GL_ShaderType.TWO_STAGE);
        xy_vbo  = GPU.GL.new_vec2_buffer_static(vao, XY_ATTRIBUTE, GPU.GL.screen_quad_vertices);
        uv_vbo  = GPU.GL.new_vec2_buffer_static(vao, UV_ATTRIBUTE, GPU.GL.screen_quad_uvs);
        vao.enable_attribute(XY_ATTRIBUTE);
        vao.enable_attribute(UV_ATTRIBUTE);
    }

    @Override
    public void tick(float dt)
    {
        vao.bind();
        shader.use();
        texture.bind(0);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
        shader.detach();
        vao.unbind();
    }

    @Override
    public void shutdown()
    {
        vao.release();
        xy_vbo.release();
        uv_vbo.release();
        shader.release();
        texture.release();
    }
}