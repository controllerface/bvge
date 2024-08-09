package com.controllerface.bvge.rendering.renderers;

import com.controllerface.bvge.core.Window;
import com.controllerface.bvge.events.Event;
import com.controllerface.bvge.game.Constants;
import com.controllerface.bvge.game.state.PlayerInventory;
import com.controllerface.bvge.gpu.GPU;
import com.controllerface.bvge.gpu.gl.buffers.GL_CommandBuffer;
import com.controllerface.bvge.gpu.gl.buffers.GL_VertexArray;
import com.controllerface.bvge.gpu.gl.buffers.GL_VertexBuffer;
import com.controllerface.bvge.gpu.gl.shaders.GL_Shader;
import com.controllerface.bvge.gpu.gl.shaders.GL_ShaderType;
import com.controllerface.bvge.gpu.gl.textures.GL_Texture2D;
import com.controllerface.bvge.rendering.Renderer;
import com.controllerface.bvge.rendering.TextGlyph;
import com.controllerface.bvge.substances.Solid;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.Queue;
import java.util.concurrent.ConcurrentLinkedQueue;

import static com.controllerface.bvge.game.Constants.Rendering.*;
import static org.lwjgl.opengl.GL45C.GL_TRIANGLE_STRIP;
import static org.lwjgl.opengl.GL45C.glMultiDrawArraysIndirect;

public class HUDRenderer implements Renderer
{
    private static final float INVENTORY_TEXT_SCALE = .5f;
    private static final int COMMAND_BUFFER_SIZE = MAX_BATCH_SIZE * Integer.BYTES * 4;
    private static final int SOLID_LABEL_Y_OFFSET = 150;
    private static final int TEXTURE_SIZE = 64;
    private static final int VERTICES_PER_LETTER = 4;
    private static final int XY_ATTRIBUTE = 0;
    private static final int UV_ATTRIBUTE = 1;
    private static final int ID_ATTRIBUTE = 2;

    private final int[] raw_cmd = new int[VERTICES_PER_LETTER * MAX_BATCH_SIZE];
    private final PlayerInventory player_inventory;
    private final Map<Character, TextGlyph> character_map = new HashMap<>();
    private final Map<String, TextContainer> text_boxes = new HashMap<>();
    private final Map<Solid, TextContainer> solid_labels = new HashMap<>();
    private final Queue<Event> event_queue = new ConcurrentLinkedQueue<>();

    private float max_char_height = 0;
    private int max_label_chars = 0;

    private GL_VertexArray vao;
    private GL_VertexBuffer xy_vbo;
    private GL_VertexBuffer uv_vbo;
    private GL_VertexBuffer id_vbo;
    private GL_CommandBuffer cbo;
    private GL_Texture2D texture;
    private GL_Shader shader;

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

    public HUDRenderer(PlayerInventory player_inventory)
    {
        this.player_inventory = player_inventory;
        Window.get().event_bus().register(event_queue, Event.Type.WINDOW_RESIZE, Event.Type.ITEM_CHANGE, Event.Type.ITEM_PLACING);
        build_cmd();
        init_GL();
        gather_text_metrics();
    }

    private void build_cmd()
    {
        int cmd_offset = 0;
        for (int i = 0; i < Constants.Rendering.MAX_BATCH_SIZE; i++)
        {
            int index = i * VERTICES_PER_LETTER;
            raw_cmd[cmd_offset++] = VERTICES_PER_LETTER;
            raw_cmd[cmd_offset++] = 1;
            raw_cmd[cmd_offset++] = index;
            raw_cmd[cmd_offset++] = i;
        }
    }

    private void gather_text_metrics()
    {
        for (var character : character_map.values())
        {
            max_char_height = Math.max(max_char_height, character.size()[1]);
        }
        for (var solid : Solid.values())
        {
            max_label_chars = Math.max(max_label_chars, solid.name().length());
        }
    }

    private void init_GL()
    {
        text_boxes.put("debug", new TextContainer(SnapPosition.BOTTOM_LEFT,
            "BVGE Prototype 2024.6.0", 100, 100, .75f));

        text_boxes.put("debug2", new TextContainer(SnapPosition.TOP_RIGHT,
            "Placing", 100, 100, .75f));

        text_boxes.put("placing", new TextContainer(SnapPosition.TOP_RIGHT,
            "-", 100, 150, .75f));

        text_boxes.put("inventory", new TextContainer(SnapPosition.TOP_LEFT,
            "Inventory", 100, 100, .75f));

        shader = GPU.GL.new_shader("text_shader.glsl", GL_ShaderType.TWO_STAGE);

        vao = GPU.GL.new_vao();
        xy_vbo = GPU.GL.new_buffer_vec2(vao, XY_ATTRIBUTE, VECTOR_FLOAT_2D_SIZE * VERTICES_PER_LETTER * MAX_BATCH_SIZE);
        uv_vbo = GPU.GL.new_buffer_vec2(vao, UV_ATTRIBUTE, VECTOR_FLOAT_2D_SIZE * VERTICES_PER_LETTER * MAX_BATCH_SIZE);
        id_vbo = GPU.GL.new_buffer_float(vao, ID_ATTRIBUTE, SCALAR_FLOAT_SIZE * MAX_BATCH_SIZE);

        vao.enable_attribute(XY_ATTRIBUTE);
        vao.enable_attribute(UV_ATTRIBUTE);
        vao.enable_attribute(ID_ATTRIBUTE);
        vao.instance_attribute(ID_ATTRIBUTE, 1);

        cbo = GPU.GL.dynamic_command_buffer(vao, COMMAND_BUFFER_SIZE);
        cbo.load_int_sub_data(raw_cmd, 0);
        texture = GPU.GL.build_character_map(TEXTURE_SIZE, "/font/Inconsolata-Light.ttf", character_map);
    }

    private boolean dirty = true;

    private int current_glyph_count = 0;

    private float calculate_text_width(String text, float scale)
    {
        float width = 0.0f;
        for (var character : text.toCharArray())
        {
            var glyph = character_map.get(character);
            width += (glyph.advance() >> 6) * scale;
        }
        return width;
    }

    private String pad_label(String label)
    {
        if (label.length() == max_label_chars) return label;
        int padding = max_label_chars - label.length();
        var buffer = new StringBuilder(label);
        for (int i = 0; i < padding; i++)
        {
            buffer.append(" ");
        }
        return buffer.toString();
    }

    private void rebuild_inventory()
    {
        solid_labels.clear();
        int solid_count = 0;
        for (var count : player_inventory.solid_counts().entrySet())
        {
            if (count.getValue() > 0)
            {
                var s = count.getKey();
                var msg = pad_label(s.name()) + " : " + count.getValue();
                var offset = SOLID_LABEL_Y_OFFSET + (solid_count++ * (max_char_height * INVENTORY_TEXT_SCALE));
                solid_labels.put(count.getKey(), new TextContainer(SnapPosition.TOP_LEFT, msg, 100, offset, INVENTORY_TEXT_SCALE));
            }
        }
    }

    private void rebuild_hud()
    {
        // todo: in the future, the count of glyphs and possibly other objects will need to
        //  be calculated up front, and then checked against the current buffer size. If
        //  the buffer is not large enough, it will need to be expanded. This should work
        //  similarly to the CL buffers that have size ensuring methods.
        rebuild_inventory();
        current_glyph_count = 0;
        int pos_offset = 0;
        int uv_offset = 0;
        int id_offset = 0;

        var pos_buf = xy_vbo.map_as_float_buffer();
        var uv_buf  = uv_vbo.map_as_float_buffer();
        var id_buf  = id_vbo.map_as_float_buffer();

        var text_containers = new ArrayList<TextContainer>();
        text_containers.addAll(text_boxes.values());
        text_containers.addAll(solid_labels.values());

        for (var text_box : text_containers)
        {
            float x      = text_box.x();
            float y      = text_box.y();;
            float scale  = text_box.scale();

            float window_width  = Window.get().width();
            float window_height = Window.get().height();

            float width = calculate_text_width(text_box.message(), scale);
            float height = max_char_height * scale;

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

            for (var character : text_box.message().toCharArray())
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

                x += (glyph.advance() >> 6) * scale;
            }
        }

        xy_vbo.unmap_buffer();
        uv_vbo.unmap_buffer();
        id_vbo.unmap_buffer();

        dirty = false;
    }

    private void render_hud()
    {
        if (dirty) rebuild_hud();
        glMultiDrawArraysIndirect(GL_TRIANGLE_STRIP, 0, current_glyph_count, 0);
    }

    @Override
    public void render()
    {
        Event next_event;
        while ((next_event = event_queue.poll()) != null)
        {
            // todo: eventually, the HUD can be split into static and dynamic sections, and there could be a different dirty
            //  flag for each section, or the entire HUD. This will help make it more responsive as only the sections that
            //  change can be updated. It will require designating certain positions in the buffer
            if (next_event.type() == Event.Type.WINDOW_RESIZE
                || next_event.type() == Event.Type.ITEM_CHANGE)
            {
                dirty = true;
            }
            if (next_event instanceof Event.Message(var _, var message))
            {
                var current = text_boxes.get("placing");
                var next = new TextContainer(current.snap, message, current.x, current.y, current.scale);
                text_boxes.put("placing", next);
                dirty = true;
            }
        }

        vao.bind();
        shader.use();
        shader.uploadInt("uTexture", 0);
        shader.uploadMat4f("projection", Window.get().camera().get_screen_matrix());
        texture.bind(0);
        cbo.bind();
        render_hud();
        vao.unbind();
        shader.detach();
    }

    @Override
    public void destroy()
    {
        vao.release();
        cbo.release();
        xy_vbo.release();
        uv_vbo.release();
        id_vbo.release();
        shader.release();
        texture.release();
    }
}