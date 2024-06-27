package com.controllerface.bvge.gl;

import org.lwjgl.PointerBuffer;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.util.freetype.FT_Face;
import org.lwjgl.util.freetype.FT_GlyphSlot;

import java.util.Map;
import java.util.Objects;

import static com.controllerface.bvge.util.Constants.Rendering.*;
import static org.lwjgl.opengl.GL45C.*;
import static org.lwjgl.util.freetype.FreeType.*;
import static org.lwjgl.util.freetype.FreeType.FT_Done_FreeType;
import static org.lwjgl.util.harfbuzz.HarfBuzz.*;

public class GLUtils
{
    private static final int DEFAULT_OFFSET = 0;
    private static final int DEFAULT_STRIDE = 0;

    public record RenderableGlyph(Texture texture, int[] size, int[] bearing, long advance) { }

    public static int new_buffer_float(int vao,
                                       int bind_index,
                                       int buffer_size)
    {
        return dynamic_float_buffer(vao, bind_index, buffer_size, SCALAR_LENGTH, SCALAR_FLOAT_SIZE);
    }

    public static int new_buffer_int(int vao,
                                     int bind_index,
                                     int buffer_size)
    {
        return dynamic_int_buffer(vao, bind_index, buffer_size, SCALAR_LENGTH, SCALAR_INT_SIZE);
    }

    public static int new_buffer_vec2(int vao,
                                      int bind_index,
                                      int buffer_size)
    {
        return dynamic_float_buffer(vao, bind_index, buffer_size, VECTOR_2D_LENGTH, VECTOR_FLOAT_2D_SIZE);
    }

    public static int fill_buffer_vec2(int vao,
                                       int bind_index,
                                       float[] buffer_data)
    {
        return static_float_buffer(vao, bind_index, VECTOR_2D_LENGTH, VECTOR_FLOAT_2D_SIZE, buffer_data);
    }

    public static int new_buffer_vec4(int vao,
                                      int bind_index,
                                      int buffer_size)
    {
        return dynamic_float_buffer(vao, bind_index, buffer_size, VECTOR_4D_LENGTH, VECTOR_FLOAT_4D_SIZE);
    }

    public static int static_element_buffer(int vao, int[] faces)
    {
        int ebo = glCreateBuffers();
        glNamedBufferData(ebo, faces, GL_STATIC_DRAW);
        glVertexArrayElementBuffer(vao, ebo);
        return ebo;
    }

    public static int dynamic_element_buffer(int vao, int buffer_size)
    {
        int ebo = glCreateBuffers();
        glNamedBufferData(ebo, buffer_size, GL_DYNAMIC_DRAW);
        glVertexArrayElementBuffer(vao, ebo);
        return ebo;
    }

    public static int dynamic_command_buffer(int vao, int buffer_size)
    {
        int cbo = glCreateBuffers();
        glNamedBufferData(cbo, buffer_size, GL_DYNAMIC_DRAW);
        glBindVertexArray(vao);
        glBindBuffer(GL_DRAW_INDIRECT_BUFFER, cbo);
        glBindVertexArray(0);
        return cbo;
    }

    private static int dynamic_float_buffer(int vao,
                                            int bind_index,
                                            int buffer_size,
                                            int data_count,
                                            int data_size)
    {
        return create_vertex_buffer(vao,
            DEFAULT_OFFSET,
            bind_index,
            bind_index,
            buffer_size,
            GL_FLOAT,
            data_count,
            data_size,
            DEFAULT_STRIDE,
            GL_DYNAMIC_DRAW);
    }

    private static int dynamic_int_buffer(int vao,
                                            int bind_index,
                                            int buffer_size,
                                            int data_count,
                                            int data_size)
    {
        return create_vertex_buffer(vao,
            DEFAULT_OFFSET,
            bind_index,
            bind_index,
            buffer_size,
            GL_INT,
            data_count,
            data_size,
            DEFAULT_STRIDE,
            GL_DYNAMIC_DRAW);
    }

    private static int static_float_buffer(int vao,
                                           int bind_index,
                                           int data_count,
                                           int data_size,
                                           float[] buffer_data)
    {
        return create_vertex_buffer(vao,
            DEFAULT_OFFSET,
            bind_index,
            bind_index,
            GL_FLOAT,
            data_count,
            data_size,
            DEFAULT_STRIDE,
            GL_STATIC_DRAW,
            buffer_data);
    }

    private static int create_vertex_buffer(int vao,
                                            int buffer_offset,
                                            int bind_index,
                                            int attribute_index,
                                            int buffer_size,
                                            int data_type,
                                            int data_count,
                                            int data_size,
                                            int data_stride,
                                            int flags)
    {
        int buffer = glCreateBuffers();
        glNamedBufferData(buffer, buffer_size, flags);
        glVertexArrayVertexBuffer(vao, bind_index, buffer, buffer_offset, data_size);
        glVertexArrayAttribFormat(vao, bind_index, data_count, data_type, false, data_stride);
        glVertexArrayAttribBinding(vao, attribute_index, bind_index);
        return buffer;
    }

    private static int create_vertex_buffer(int vao,
                                            int buffer_offset,
                                            int bind_index,
                                            int attribute_index,
                                            int data_type,
                                            int data_count,
                                            int data_size,
                                            int data_stride,
                                            int flags,
                                            float[] buffer_data)
    {
        int buffer = glCreateBuffers();
        glNamedBufferData(buffer, buffer_data, flags);
        glVertexArrayVertexBuffer(vao, bind_index, buffer, buffer_offset, data_size);
        glVertexArrayAttribFormat(vao, bind_index, data_count, data_type, false, data_stride);
        glVertexArrayAttribBinding(vao, attribute_index, bind_index);
        return buffer;
    }

    private static FT_Face loadFontFace(long ftLibrary, String fontPath)
    {
        try (MemoryStack stack = MemoryStack.stackPush())
        {
            PointerBuffer pp = stack.mallocPointer(1);
            int error = FT_New_Face(ftLibrary, fontPath, 0, pp);
            if (error != 0)
            {
                System.err.println("FT_New_Face error: " + error);
                return null;
            }
            return FT_Face.create(pp.get(0));
        }
    }

    private static long initFreeType()
    {
        try (MemoryStack stack = MemoryStack.stackPush())
        {
            PointerBuffer pp = stack.mallocPointer(1);
            if (FT_Init_FreeType(pp) != 0)
            {
                throw new RuntimeException("Could not initialize FreeType library");
            }
            return pp.get(0);
        }
    }

    private static FT_GlyphSlot render_glyph(FT_Face ftFace, int glyphID)
    {
        if (FT_Load_Glyph(ftFace, glyphID, FT_LOAD_DEFAULT) != 0)
        {
            throw new RuntimeException("Could not load glyph");
        }

        FT_GlyphSlot glyph = ftFace.glyph();
        if (FT_Render_Glyph(glyph, FT_RENDER_MODE_NORMAL) != 0)
        {
            throw new RuntimeException("Could not render glyph");
        }

        return glyph;
    }

    public static void build_character_map(String font_file, String[] character_set, Map<Character, RenderableGlyph> character_map)
    {
        long ft_library = initFreeType();
        FT_Face ft_face = loadFontFace(ft_library, font_file);
        Objects.requireNonNull(ft_face);
        FT_Set_Pixel_Sizes(ft_face, 0, 36);
        var font = hb_ft_font_create_referenced(ft_face.address());
        var face = hb_ft_face_create_referenced(ft_face.address());

        for (var character_string : character_set)
        {
            var buffer = hb_buffer_create();
            hb_buffer_add_utf8(buffer, character_string, 0, character_string.length());
            hb_buffer_guess_segment_properties(buffer);
            hb_shape(font, buffer, null);
            var glyph_info = hb_buffer_get_glyph_infos(buffer);
            int glyphCount = hb_buffer_get_length(buffer);
            if (glyphCount != 1 || glyph_info == null)
            {
                throw new RuntimeException("Glyph count incorrect: " + glyphCount);
            }
            var glyph_id = glyph_info.get(0).codepoint();
            var glyph = render_glyph(ft_face, glyph_id);
            var bm = glyph.bitmap();
            var glyph_texture = new Texture(bm);
            var size = new int[]{bm.width(), bm.rows()};
            var bearing = new int[]{glyph.bitmap_left(), glyph.bitmap_top()};
            var advance = glyph.advance();
            var character = new RenderableGlyph(glyph_texture, size, bearing, advance.x());
            character_map.put(character_string.charAt(0), character);
            hb_buffer_destroy(buffer);
        }

        hb_font_destroy(font);
        hb_face_destroy(face);
        FT_Done_FreeType(ft_library);
    }
}
