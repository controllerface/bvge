package com.controllerface.bvge.gl.renderers;

import com.controllerface.bvge.ecs.ECS;
import com.controllerface.bvge.ecs.systems.GameSystem;
import com.controllerface.bvge.gl.GLUtils;
import com.controllerface.bvge.gl.Shader;
import com.controllerface.bvge.util.Assets;
import com.controllerface.bvge.window.Window;

import java.util.HashMap;
import java.util.Map;

import static com.controllerface.bvge.util.Constants.Rendering.VECTOR_FLOAT_2D_SIZE;
import static org.lwjgl.opengl.GL15C.glDrawArrays;
import static org.lwjgl.opengl.GL30C.glBindVertexArray;
import static org.lwjgl.opengl.GL45C.*;

public class HUDRenderer extends GameSystem
{
    private static final int POSITION_ATTRIBUTE = 0;
    private static final int UV_ATTRIBUTE = 1;

    private int vao;
    private int position_vbo;
    private int uv_vbo;

    private Shader shader;

    private final Map<Character, GLUtils.RenderableGlyph> character_map = new HashMap<>();

    private final String font_file = "C:\\Users\\Stephen\\IdeaProjects\\bvge\\src\\main\\resources\\font\\Inconsolata-Light.ttf";

    private static final String[] character_set =
        {
            "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
            "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
            "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "!", "@", "#", "$", "%", "^", "&", "*", "(", ")", "-", "_", "=", "+", "[", "]",
            "{", "}", "|", ";", ":", "'", ",", "<", ".", ">", "/", "?", "~", "`", " ", "\\", "\""
        };

    public HUDRenderer(ECS ecs)
    {
        super(ecs);
        init_GL();
    }

    private void init_GL()
    {
        float[] uvs = new float[]
            {
                1.0f, 0.0f,
                1.0f, 1.0f,
                0.0f, 0.0f,
                0.0f, 1.0f,
            };

        shader = Assets.load_shader("text_shader.glsl");
        shader.uploadInt("uTexture", 0);
        vao = glCreateVertexArrays();
        position_vbo = GLUtils.new_buffer_vec2(vao, POSITION_ATTRIBUTE, VECTOR_FLOAT_2D_SIZE * 4);
        uv_vbo = GLUtils.fill_buffer_vec2(vao, UV_ATTRIBUTE, uvs);
        glEnableVertexArrayAttrib(vao, POSITION_ATTRIBUTE);
        glEnableVertexArrayAttrib(vao, UV_ATTRIBUTE);

        GLUtils.build_character_map(font_file, character_set, character_map);
    }

    private void render_text(String text, float x, float y, float scale)
    {
        for (var character : text.toCharArray())
        {
            var glyph = character_map.get(character);

            float w = glyph.size()[0] * scale;
            float h = glyph.size()[1] * scale;
            float x1_pos = x + glyph.bearing()[0] * scale;
            float y1_pos = y - (glyph.size()[1] - glyph.bearing()[1]) * scale;
            float x2_pos = x1_pos + w;
            float y2_pos = y1_pos + h;

            float[] vertices = new float[]
                {
                    x2_pos, y1_pos,
                    x2_pos, y2_pos,
                    x1_pos, y1_pos,
                    x1_pos, y2_pos,
                };

            glyph.texture().bind(0);
            glNamedBufferSubData(position_vbo, 0, vertices);
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