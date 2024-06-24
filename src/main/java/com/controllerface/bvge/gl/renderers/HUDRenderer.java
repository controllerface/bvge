package com.controllerface.bvge.gl.renderers;

import com.controllerface.bvge.ecs.ECS;
import com.controllerface.bvge.ecs.systems.GameSystem;
import com.controllerface.bvge.gl.Shader;
import com.controllerface.bvge.gl.Texture;
import org.lwjgl.PointerBuffer;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.util.freetype.FT_Face;
import org.lwjgl.util.freetype.FT_GlyphSlot;

import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

import static org.lwjgl.opengl.GL15C.glDeleteBuffers;
import static org.lwjgl.opengl.GL30C.glDeleteVertexArrays;
import static org.lwjgl.opengl.GL45C.glCreateVertexArrays;
import static org.lwjgl.util.freetype.FreeType.*;
import static org.lwjgl.util.harfbuzz.HarfBuzz.*;

public class HUDRenderer extends GameSystem
{
    private static final int POSITION_ATTRIBUTE = 0;
    private static final int UV_ATTRIBUTE = 1;

    private final int[] texture_slots = {0};

    private int vao;
    private int position_vbo;
    private int uv_vbo;

    private Shader shader;

    private final String font_file = "C:\\Users\\Stephen\\IdeaProjects\\bvge\\src\\main\\resources\\font\\Inconsolata-Light.ttf";

    private record Character(Texture texture, int[] size, int[] bearing, long advance) {}

    private static final String[] character_set =
        {
            "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
            "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
            "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "!", "@", "#", "$", "%", "^", "&", "*", "(", ")", "-", "_", "=", "+", "[", "]",
            "{", "}", "|", ";", ":", "'", ",", "<", ".", ">", "/", "?", "~", "`", " ", "\\", "\""
        };

    private final Map<String, Character> character_map = new HashMap<>();

    public HUDRenderer(ECS ecs)
    {
        super(ecs);
        init_GL();
    }

    private void init_GL()
    {
        long ft_library = initFreeType();
        for (var character : character_set)
        {
            render_character(ft_library, character);
        }
    }

    @Override
    public void tick(float dt)
    {

    }

    @Override
    public void shutdown()
    {

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

    private FT_GlyphSlot render_glyph(FT_Face ftFace, int glyphID)
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

    private void render_character(long ft_library, String character_string)
    {
        FT_Face ft_face = loadFontFace(ft_library, font_file);
        Objects.requireNonNull(ft_face);
        FT_Set_Pixel_Sizes(ft_face, 0, 48);

        var buffer = hb_buffer_create();
        hb_buffer_add_utf8(buffer, character_string, 0, character_string.length());
        hb_buffer_set_direction(buffer, HB_DIRECTION_LTR);
        hb_buffer_set_script(buffer, HB_SCRIPT_LATIN);
        hb_buffer_set_language(buffer, hb_language_from_string("en"));

        var font = hb_ft_font_create_referenced(ft_face.address());
        var face = hb_ft_face_create_referenced(ft_face.address());

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
        var character = new Character(glyph_texture, size, bearing, advance.x());
        character_map.put(character_string, character);

        hb_buffer_destroy(buffer);
        hb_font_destroy(font);
        hb_face_destroy(face);
    }
}
