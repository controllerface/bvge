package com.controllerface.bvge.gl.renderers;

import com.controllerface.bvge.ecs.ECS;
import com.controllerface.bvge.ecs.systems.GameSystem;
import com.controllerface.bvge.gl.GLUtils;
import com.controllerface.bvge.gl.TextGlyph;
import com.controllerface.bvge.gl.Shader;
import com.controllerface.bvge.gl.Texture;
import com.controllerface.bvge.util.Assets;
import com.controllerface.bvge.util.Constants;
import com.controllerface.bvge.window.Window;

import java.util.HashMap;
import java.util.Map;

import static com.controllerface.bvge.util.Constants.Rendering.*;
import static org.lwjgl.opengl.GL30C.glBindVertexArray;
import static org.lwjgl.opengl.GL45C.*;

public class HUDRenderer extends GameSystem
{
    private static final int TEXTURE_SIZE = 64;
    private static final int VERTICES_PER_LETTER = 4;

    private static final int COMMAND_BUFFER_SIZE = MAX_BATCH_SIZE * Integer.BYTES * 4;

    private static final int POSITION_ATTRIBUTE = 0;
    private static final int UV_ATTRIBUTE = 1;
    private static final int ID_ATTRIBUTE = 2;

    private int vao;
    private int position_vbo;
    private int uv_vbo;
    private int id_vbo;
    private int cbo;

    private Texture texture;
    private Shader shader;

    private final Map<Character, TextGlyph> character_map_ex = new HashMap<>();
    private final int[] raw_cmd = new int[VERTICES_PER_LETTER * MAX_BATCH_SIZE];

    private enum DockEdge
    {
        TOP,
        BOTTOM,
        LEFT,
        RIGHT
    }

    private record TextContainer(DockEdge edge, String message, float x, float y, float size){ }

    private Map<String, TextContainer> text_boxes = new HashMap<>();

    public HUDRenderer(ECS ecs)
    {
        super(ecs);
        int cmd_offset = 0;
        for (int i = 0; i < Constants.Rendering.MAX_BATCH_SIZE; i++)
        {
            int index = i * VERTICES_PER_LETTER;
            raw_cmd[cmd_offset++] = VERTICES_PER_LETTER;
            raw_cmd[cmd_offset++] = 1;
            raw_cmd[cmd_offset++] = index;
            raw_cmd[cmd_offset++] = i;
        }
        init_GL();
    }

    private void init_GL()
    {
        //text_boxes.put("debug", new TextContainer("BVGE Prototype 2024.6.0", 100, 100, .75f));
        //text_boxes.put("debug2", new TextContainer("Test", 100, 100, .75f));

        shader = Assets.load_shader("text_shader.glsl");
        shader.uploadInt("uTexture", 0);
        vao = glCreateVertexArrays();
        position_vbo = GLUtils.new_buffer_vec2(vao, POSITION_ATTRIBUTE, VECTOR_FLOAT_2D_SIZE * VERTICES_PER_LETTER * MAX_BATCH_SIZE);
        uv_vbo = GLUtils.new_buffer_vec2(vao, UV_ATTRIBUTE, VECTOR_FLOAT_2D_SIZE * VERTICES_PER_LETTER * MAX_BATCH_SIZE);
        id_vbo = GLUtils.new_buffer_float(vao, ID_ATTRIBUTE, SCALAR_FLOAT_SIZE * VERTICES_PER_LETTER * MAX_BATCH_SIZE);

        glEnableVertexArrayAttrib(vao, POSITION_ATTRIBUTE);
        glEnableVertexArrayAttrib(vao, UV_ATTRIBUTE);
        glEnableVertexArrayAttrib(vao, ID_ATTRIBUTE);

        cbo = GLUtils.dynamic_command_buffer(vao, COMMAND_BUFFER_SIZE);
        glNamedBufferSubData(cbo, 0, raw_cmd);
        texture = GLUtils.build_character_map(TEXTURE_SIZE, "/font/Inconsolata-Light.ttf", character_map_ex);
    }

    private boolean dirty = true;

    private void transfer_hud_data(String text, float x, float y, float scale)
    {
        int pos_offset = 0;
        int uv_offset = 0;
        int id_offset = 0;
        char[] chars = text.toCharArray();

        var p_buf = glMapNamedBuffer(position_vbo, GL_WRITE_ONLY).asFloatBuffer();
        var u_buf = glMapNamedBuffer(uv_vbo, GL_WRITE_ONLY).asFloatBuffer();
        var i_buf = glMapNamedBuffer(id_vbo, GL_WRITE_ONLY).asFloatBuffer();

        for (var character : chars)
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

            p_buf.put(pos_offset++, x2);
            p_buf.put(pos_offset++, y1);
            p_buf.put(pos_offset++, x2);
            p_buf.put(pos_offset++, y2);
            p_buf.put(pos_offset++, x1);
            p_buf.put(pos_offset++, y1);
            p_buf.put(pos_offset++, x1);
            p_buf.put(pos_offset++, y2);

            u_buf.put(uv_offset++, u2);
            u_buf.put(uv_offset++, v1);
            u_buf.put(uv_offset++, u2);
            u_buf.put(uv_offset++, v2);
            u_buf.put(uv_offset++, u1);
            u_buf.put(uv_offset++, v1);
            u_buf.put(uv_offset++, u1);
            u_buf.put(uv_offset++, v2);

            i_buf.put(id_offset++, glyph.texture_id());
            i_buf.put(id_offset++, glyph.texture_id());
            i_buf.put(id_offset++, glyph.texture_id());
            i_buf.put(id_offset++, glyph.texture_id());

            x += (glyph.advance() >> 6) * scale;
        }

        glUnmapNamedBuffer(position_vbo);
        glUnmapNamedBuffer(uv_vbo);
        glUnmapNamedBuffer(id_vbo);

        dirty = false;
    }

    private void render_text(String text, float x, float y, float scale)
    {
        if (dirty) transfer_hud_data(text, x, y, scale);
        glMultiDrawArraysIndirect(GL_TRIANGLE_STRIP, 0, text.length(), 0);
    }

    @Override
    public void tick(float dt)
    {
        glBindVertexArray(vao);
        shader.use();
        shader.uploadMat4f("projection", Window.get().camera().get_screen_matrix());
        texture.bind(0);
        glBindBuffer(GL_DRAW_INDIRECT_BUFFER, cbo);

        render_text("BVGE Prototype 2024.6.0", 100, 100, .75f);

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