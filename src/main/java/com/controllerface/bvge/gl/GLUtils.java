package com.controllerface.bvge.gl;

import org.lwjgl.PointerBuffer;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.system.MemoryUtil;
import org.lwjgl.util.freetype.FT_Bitmap;
import org.lwjgl.util.freetype.FT_Face;
import org.lwjgl.util.freetype.FT_GlyphSlot;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Map;
import java.util.Objects;

import static com.controllerface.bvge.util.Constants.Rendering.*;
import static org.lwjgl.opengl.GL11.GL_NEAREST;
import static org.lwjgl.opengl.GL11.GL_TEXTURE_MAG_FILTER;
import static org.lwjgl.opengl.GL11.GL_TEXTURE_MIN_FILTER;
import static org.lwjgl.opengl.GL11.GL_TEXTURE_WRAP_S;
import static org.lwjgl.opengl.GL11.GL_TEXTURE_WRAP_T;
import static org.lwjgl.opengl.GL11.GL_UNSIGNED_BYTE;
import static org.lwjgl.opengl.GL45C.*;
import static org.lwjgl.util.freetype.FreeType.*;
import static org.lwjgl.util.freetype.FreeType.FT_Done_FreeType;
import static org.lwjgl.util.harfbuzz.HarfBuzz.*;

public class GLUtils
{
    private static final int DEFAULT_OFFSET = 0;
    private static final int DEFAULT_STRIDE = 0;

    public static final String[] character_set =
        {
            "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
            "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
            "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "!", "@", "#", "$", "%", "^", "&", "*", "(", ")", "-", "_", "=", "+", "[", "]",
            "{", "}", "|", ";", ":", "'", ",", "<", ".", ">", "/", "?", "~", "`", " ", "\\", "\""
        };


    public record RenderableGlyph(int texture_id, int[] size, int[] bearing, long advance) { }

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

    public static int fill_buffer_int(int vao,
                                      int bind_index,
                                      int[] buffer_data)
    {
        return static_int_buffer(vao, bind_index, VECTOR_2D_LENGTH, VECTOR_FLOAT_2D_SIZE, buffer_data);
    }

    public static int new_buffer_vec4(int vao,
                                      int bind_index,
                                      int buffer_size)
    {
        return dynamic_float_buffer(vao, bind_index, buffer_size, VECTOR_4D_LENGTH, VECTOR_FLOAT_4D_SIZE);
    }


    public static int new_buffer_mat4(int vao,
                                      int bind_index,
                                      int buffer_size)
    {
        return dynamic_float_buffer(vao, bind_index, buffer_size, MATRIX_4_LENGTH, MATRIX_FLOAT_4_SIZE);
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
        return create_vertex_buffer_float(vao,
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


    private static int static_int_buffer(int vao,
                                         int bind_index,
                                         int data_count,
                                         int data_size,
                                         int[] buffer_data)
    {
        return create_vertex_buffer_int(vao,
            DEFAULT_OFFSET,
            bind_index,
            bind_index,
            GL_INT,
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

    private static int create_vertex_buffer_float(int vao,
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

    private static int create_vertex_buffer_int(int vao,
                                                int buffer_offset,
                                                int bind_index,
                                                int attribute_index,
                                                int data_type,
                                                int data_count,
                                                int data_size,
                                                int data_stride,
                                                int flags,
                                                int[] buffer_data)
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

    private static void generate_texture_layer(int tex_id, int tex_slot, FT_Bitmap bitmap)
    {
        int width = bitmap.width();
        int height = bitmap.rows();

        ByteBuffer buffer = bitmap.buffer(width * height);
        var image = MemoryUtil.memAlloc(width * height).order(ByteOrder.nativeOrder());
        for (int x = 0; x < width; x++)
        {
            for (int y = 0; y < height; y++)
            {
                int flipped_row = height - y - 1; // flip Y here, so it isn't required inside rendering calls
                var pixel = buffer.get((y * bitmap.pitch()) + x);
                image.put((flipped_row * width) + x, pixel);
            }
        }

        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glTextureSubImage3D(tex_id, 0, 0, 0, tex_slot, width, height, 1, GL_RED, GL_UNSIGNED_BYTE, image);
        MemoryUtil.memFree(image);
    }

    public static Texture build_character_map_ex(int texture_size, String font_file, Map<Character, RenderableGlyph> character_map)
    {
        return build_character_map_ex(texture_size, font_file, character_set, character_map);
    }

    public static Texture build_character_map_ex(int texture_size, String font_file, String[] character_set, Map<Character, RenderableGlyph> character_map)
    {
        long ft_library = initFreeType();
        FT_Face ft_face = loadFontFace(ft_library, font_file);
        Objects.requireNonNull(ft_face);
        FT_Set_Pixel_Sizes(ft_face, texture_size, texture_size);
        var font = hb_ft_font_create_referenced(ft_face.address());
        var face = hb_ft_face_create_referenced(ft_face.address());
        var texture_3d = new Texture();
        texture_3d.init_array();

        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glTextureStorage3D(texture_3d.getTex_id(), 1, GL_R8, texture_size, texture_size, character_set.length);

        for (int text_slot = 0; text_slot < character_set.length; text_slot++)
        {
            var character_string = character_set[text_slot];
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

            generate_texture_layer(texture_3d.getTex_id(), text_slot, bm);

            var size = new int[]{bm.width(), bm.rows()};
            var bearing = new int[]{glyph.bitmap_left(), glyph.bitmap_top()};
            var advance = glyph.advance();
            var character = new RenderableGlyph(text_slot, size, bearing, advance.x());
            character_map.put(character_string.charAt(0), character);
            hb_buffer_destroy(buffer);
        }

        glTextureParameteri(texture_3d.getTex_id(), GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTextureParameteri(texture_3d.getTex_id(), GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTextureParameteri(texture_3d.getTex_id(), GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTextureParameteri(texture_3d.getTex_id(), GL_TEXTURE_MAG_FILTER, GL_NEAREST);

        hb_font_destroy(font);
        hb_face_destroy(face);
        FT_Done_FreeType(ft_library);

        return texture_3d;
    }

}
