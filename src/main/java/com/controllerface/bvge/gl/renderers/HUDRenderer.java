package com.controllerface.bvge.gl.renderers;

import com.controllerface.bvge.ecs.ECS;
import com.controllerface.bvge.ecs.systems.GameSystem;
import com.controllerface.bvge.gl.GLUtils;
import com.controllerface.bvge.gl.TextGlyph;
import com.controllerface.bvge.gl.Shader;
import com.controllerface.bvge.gl.Texture;
import com.controllerface.bvge.util.Assets;
import com.controllerface.bvge.util.Constants;
import com.controllerface.bvge.window.EventType;
import com.controllerface.bvge.window.Window;

import java.util.*;
import java.util.concurrent.ConcurrentLinkedQueue;

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

    private final Map<Character, TextGlyph> character_map = new HashMap<>();
    private final int[] raw_cmd = new int[VERTICES_PER_LETTER * MAX_BATCH_SIZE];

    private final Map<String, TextContainer> text_boxes = new HashMap<>();

    private final Queue<EventType> event_queue = new ConcurrentLinkedQueue<>();

    private enum SnapPosition
    {
        NONE,
        TOP,
        BOTTOM,
        LEFT,
        RIGHT,
        TOP_LEFT,
        TOP_RIGHT,
        BOTTOM_LEFT,
        BOTTOM_RIGHT,
    }

    private record TextContainer(SnapPosition snap, String message, float x, float y, float scale) { }

    public HUDRenderer(ECS ecs)
    {
        super(ecs);
        Window.get().event_bus().register(event_queue, EventType.WINDOW_RESIZE);
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
        text_boxes.put("debug", new TextContainer(SnapPosition.BOTTOM_LEFT,
            "BVGE Prototype 2024.6.0", 100, 100, .75f));

        text_boxes.put("debug2", new TextContainer(SnapPosition.TOP_RIGHT,
            "BVGE Prototype", 100, 100, .75f));

        text_boxes.put("debug3", new TextContainer(SnapPosition.BOTTOM_RIGHT,
            "BVGE Prototype", 100, 100, .75f));

        text_boxes.put("debug5", new TextContainer(SnapPosition.TOP_LEFT,
            "BVGE Prototype", 100, 100, .75f));

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
        texture = GLUtils.build_character_map(TEXTURE_SIZE, "/font/Inconsolata-Light.ttf", character_map);
    }

    private boolean dirty = true;

    private int current_glyph_count = 0;

    private float measure_text_width(String text, float scale)
    {
        float width = 0.0f;
        for (var character : text.toCharArray())
        {
            var glyph = character_map.get(character);
            width += (glyph.advance() >> 6) * scale;
        }
        return width;
    }

    private void rebuild_hud()
    {
        System.out.println("rebuilding HUD");
        int pos_offset = 0;
        int uv_offset = 0;
        int id_offset = 0;

        var pos_buf = Objects.requireNonNull(glMapNamedBuffer(position_vbo, GL_WRITE_ONLY)).asFloatBuffer();
        var uv_buf = Objects.requireNonNull(glMapNamedBuffer(uv_vbo, GL_WRITE_ONLY)).asFloatBuffer();
        var id_buf = Objects.requireNonNull(glMapNamedBuffer(id_vbo, GL_WRITE_ONLY)).asFloatBuffer();

        for (var text_box : text_boxes.values())
        {
            char[] chars = text_box.message().toCharArray();
            float x      = text_box.x();
            float y      = text_box.y();;
            float scale  = text_box.scale();

            float window_width  = Window.get().width();
            float window_height = Window.get().height();

            float width = measure_text_width(text_box.message(), scale);
            float height = character_map.get('H').size()[1] * scale;

            switch (text_box.snap())
            {
                case NONE, BOTTOM, LEFT, BOTTOM_LEFT -> { }
                case TOP, TOP_LEFT -> y = window_height - height - y;
                case RIGHT, BOTTOM_RIGHT -> x = window_width - width - x;
                case TOP_RIGHT ->
                {
                    y = window_height - height - y;
                    x = window_width - width - x;
                }
            }

            for (var character : chars)
            {
                var glyph = character_map.get(character);
                current_glyph_count++;

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

                pos_buf.put(pos_offset++, x2);
                pos_buf.put(pos_offset++, y1);
                pos_buf.put(pos_offset++, x2);
                pos_buf.put(pos_offset++, y2);
                pos_buf.put(pos_offset++, x1);
                pos_buf.put(pos_offset++, y1);
                pos_buf.put(pos_offset++, x1);
                pos_buf.put(pos_offset++, y2);

                uv_buf.put(uv_offset++, u2);
                uv_buf.put(uv_offset++, v1);
                uv_buf.put(uv_offset++, u2);
                uv_buf.put(uv_offset++, v2);
                uv_buf.put(uv_offset++, u1);
                uv_buf.put(uv_offset++, v1);
                uv_buf.put(uv_offset++, u1);
                uv_buf.put(uv_offset++, v2);

                id_buf.put(id_offset++, glyph.texture_id());
                id_buf.put(id_offset++, glyph.texture_id());
                id_buf.put(id_offset++, glyph.texture_id());
                id_buf.put(id_offset++, glyph.texture_id());

                x += (glyph.advance() >> 6) * scale;
            }
        }

        glUnmapNamedBuffer(position_vbo);
        glUnmapNamedBuffer(uv_vbo);
        glUnmapNamedBuffer(id_vbo);

        dirty = false;
    }

    private void render_hud()
    {
        if (dirty) rebuild_hud();
        glMultiDrawArraysIndirect(GL_TRIANGLE_STRIP, 0, current_glyph_count, 0);
    }

    @Override
    public void tick(float dt)
    {
        EventType next_event;
        while ((next_event = event_queue.poll())!= null)
        {
            if (next_event == EventType.WINDOW_RESIZE) dirty = true;
        }

        glBindVertexArray(vao);
        shader.use();
        shader.uploadMat4f("projection", Window.get().camera().get_screen_matrix());
        texture.bind(0);
        glBindBuffer(GL_DRAW_INDIRECT_BUFFER, cbo);
        render_hud();
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