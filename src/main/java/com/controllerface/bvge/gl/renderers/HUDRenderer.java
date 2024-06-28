package com.controllerface.bvge.gl.renderers;

import com.controllerface.bvge.ecs.ECS;
import com.controllerface.bvge.ecs.systems.GameSystem;
import com.controllerface.bvge.gl.GLUtils;
import com.controllerface.bvge.gl.Shader;
import com.controllerface.bvge.gl.Texture;
import com.controllerface.bvge.util.Assets;
import com.controllerface.bvge.window.Window;

import java.util.HashMap;
import java.util.Map;

import static com.controllerface.bvge.util.Constants.Rendering.*;
import static org.lwjgl.opengl.GL15C.glDrawArrays;
import static org.lwjgl.opengl.GL30C.glBindVertexArray;
import static org.lwjgl.opengl.GL45C.*;

public class HUDRenderer extends GameSystem
{
    private static final int TEXTURE_SIZE = 64;
    private static final int POSITION_ATTRIBUTE = 0;
    private static final int UV_ATTRIBUTE = 1;
    private static final int ID_ATTRIBUTE = 2;

    private int vao;
    private int position_vbo;
    private int uv_vbo;
    private int id_vbo;

    private Texture texture;
    private Shader shader;

    private final Map<Character, GLUtils.RenderableGlyph> character_map_ex = new HashMap<>();

    private final String font_file = "C:\\Users\\Stephen\\IdeaProjects\\bvge\\src\\main\\resources\\font\\Inconsolata-Light.ttf";

    public HUDRenderer(ECS ecs)
    {
        super(ecs);
        init_GL();
    }

    private void init_GL()
    {
        shader = Assets.load_shader("text_shader.glsl");
        shader.uploadInt("uTexture", 0);
        vao = glCreateVertexArrays();
        position_vbo = GLUtils.new_buffer_vec2(vao, POSITION_ATTRIBUTE, VECTOR_FLOAT_2D_SIZE * 4);
        uv_vbo = GLUtils.new_buffer_vec2(vao, UV_ATTRIBUTE, VECTOR_FLOAT_2D_SIZE * 4);
        id_vbo = GLUtils.new_buffer_float(vao, ID_ATTRIBUTE, SCALAR_FLOAT_SIZE * 4);

        glEnableVertexArrayAttrib(vao, POSITION_ATTRIBUTE);
        glEnableVertexArrayAttrib(vao, UV_ATTRIBUTE);
        glEnableVertexArrayAttrib(vao, ID_ATTRIBUTE);

        texture = GLUtils.build_character_map_ex(TEXTURE_SIZE, font_file, character_map_ex);
    }

    private void render_text(String text, float x, float y, float scale)
    {
        texture.bind(0);
        for (var character : text.toCharArray())
        {
            var glyph = character_map_ex.get(character);

            float w = glyph.size()[0] * scale;
            float h = glyph.size()[1] * scale;
            float x1 = x + glyph.bearing()[0] * scale;
            float y1 = y - (glyph.size()[1] - glyph.bearing()[1]) * scale;
            float x2 = x1 + w;
            float y2 = y1 + h;
            float u1 = 0.0f;
            float v1 = 0.0f;
            float u2 = (float) glyph.size()[0] / TEXTURE_SIZE;
            float v2 = (float) glyph.size()[1] / TEXTURE_SIZE;

            float[] vertices = new float[]
                {
                    x2, y1,
                    x2, y2,
                    x1, y1,
                    x1, y2,
                };

            float[] uvs = new float[]
                {
                    u2, v1,
                    u2, v2,
                    u1, v1,
                    u1, v2,
                };

            float[] id = new float[]{ glyph.texture_id(), glyph.texture_id(), glyph.texture_id(), glyph.texture_id() };

            glNamedBufferSubData(position_vbo, 0, vertices);
            glNamedBufferSubData(uv_vbo, 0, uvs);
            glNamedBufferSubData(id_vbo, 0, id);

            glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
            x += (glyph.advance() >> 6) * scale;
        }
    }

    @Override
    public void tick(float dt)
    {
        glBindVertexArray(vao);
        shader.use();
        shader.uploadMat4f("projection", Window.get().camera().get_screen_matrix());

        render_text("BVGE Prototype 2024.6.0", 100, 100, 1f);

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