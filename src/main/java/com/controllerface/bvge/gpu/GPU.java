package com.controllerface.bvge.gpu;

import com.controllerface.bvge.core.Window;
import com.controllerface.bvge.events.Event;
import com.controllerface.bvge.events.EventBus;
import com.controllerface.bvge.game.InputSystem;
import com.controllerface.bvge.gpu.gl.GL_GraphicsController;
import com.controllerface.bvge.gpu.gl.buffers.GL_VertexArray;
import com.controllerface.bvge.gpu.gl.shaders.GL_Shader;
import com.controllerface.bvge.gpu.gl.shaders.ThreeStageShader;
import com.controllerface.bvge.gpu.gl.shaders.TwoStageShader;
import com.controllerface.bvge.gpu.gl.shaders.GL_ShaderType;
import com.controllerface.bvge.gpu.gl.textures.GL_Texture2D;
import com.controllerface.bvge.rendering.TextGlyph;
import org.lwjgl.PointerBuffer;
import org.lwjgl.assimp.AITexture;
import org.lwjgl.glfw.GLFWErrorCallback;
import org.lwjgl.glfw.GLFWImage;
import org.lwjgl.glfw.GLFWWindowSizeCallbackI;
import org.lwjgl.opengl.GL43C;
import org.lwjgl.opengl.GLDebugMessageCallback;
import org.lwjgl.system.APIUtil;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.system.MemoryUtil;
import org.lwjgl.util.freetype.FT_Bitmap;
import org.lwjgl.util.freetype.FT_Face;
import org.lwjgl.util.freetype.FT_GlyphSlot;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Map;
import java.util.Objects;
import java.util.logging.Level;
import java.util.logging.Logger;

import static org.lwjgl.glfw.GLFW.*;
import static org.lwjgl.opengl.GL11.GL_NEAREST;
import static org.lwjgl.opengl.GL11.GL_TEXTURE_MAG_FILTER;
import static org.lwjgl.opengl.GL11.GL_TEXTURE_MIN_FILTER;
import static org.lwjgl.opengl.GL11.GL_TEXTURE_WRAP_S;
import static org.lwjgl.opengl.GL11.GL_TEXTURE_WRAP_T;
import static org.lwjgl.opengl.GL11.GL_UNSIGNED_BYTE;
import static org.lwjgl.opengl.GL11C.*;
import static org.lwjgl.opengl.GL20C.*;
import static org.lwjgl.opengl.GL30C.*;
import static org.lwjgl.opengl.GL45C.*;
import static org.lwjgl.system.MemoryUtil.NULL;
import static org.lwjgl.util.freetype.FreeType.*;
import static org.lwjgl.util.harfbuzz.HarfBuzz.*;

public class GPU
{
    private static final Logger LOGGER = Logger.getLogger(GPU.class.getName());

    public static class GL
    {
        public static final String[] character_set =
            {
                "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
                "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
                "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "!", "@", "#", "$", "%", "^", "&", "*", "(", ")", "-", "_", "=", "+", "[", "]",
                "{", "}", "|", ";", ":", "'", ",", "<", ".", ">", "/", "?", "~", "`", " ", "\\", "\""
            };

        public static GL_Shader new_shader(String shader_file, GL_ShaderType shader_type)
        {
            GL_Shader shader = switch (shader_type)
            {
                case TWO_STAGE -> new TwoStageShader(shader_file);
                case THREE_STAGE -> new ThreeStageShader(shader_file);
            };
            shader.compile();
            return shader;
        }

        public static GL_Texture2D new_texture(AITexture textureData)
        {
            GL_Texture2D texture = new GL_Texture2D();
            texture.init(textureData);
            return texture;
        }

        public static GL_Texture2D new_texture(String resourceName)
        {
            GL_Texture2D texture = new GL_Texture2D();
            texture.init(resourceName);
            return texture;
        }

        public static GL_Texture2D build_character_map(int texture_size, String font_file, Map<Character, TextGlyph> character_map)
        {
            return build_character_map(texture_size, font_file, character_set, character_map);
        }

        public static GL_Texture2D build_character_map(int texture_size, String font_file, String[] character_set, Map<Character, TextGlyph> character_map)
        {
            long ft_library = initFreeType();
            FT_Face ft_face = loadFontFace(ft_library, font_file);
            Objects.requireNonNull(ft_face);
            FT_Set_Pixel_Sizes(ft_face, texture_size, texture_size);
            var font = hb_ft_font_create_referenced(ft_face.address());
            var face = hb_ft_face_create_referenced(ft_face.address());
            var texture_3d = new GL_Texture2D();
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
                var character = new TextGlyph(text_slot, size, bearing, advance.x());
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

        public static GL_VertexArray new_vao()
        {
            int vao = glCreateVertexArrays();
            return new GL_VertexArray(vao);
        }

        public static GL_GraphicsController init_gl(String title, EventBus event_bus, InputSystem inputSystem)
        {
            GLFWErrorCallback.createPrint(System.err).set();

            if (!glfwInit())
            {
                throw new IllegalStateException("Error: glfwInit()");
            }

            glfwDefaultWindowHints();
            glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
            glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
            glfwWindowHint(GLFW_VISIBLE, GLFW_TRUE);
            glfwWindowHint(GLFW_MAXIMIZED, GLFW_FALSE);
            glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);
            glfwWindowHint(GLFW_DECORATED, GLFW_FALSE);
            glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
            glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GLFW_TRUE);

            var primary_monitor = glfwGetPrimaryMonitor();
            var video_mode = glfwGetVideoMode(primary_monitor);
            if (video_mode != null)
            {
                Window.get().update_width(video_mode.width());
                Window.get().update_height(video_mode.height());
            }

            long glfw_window = glfwCreateWindow(Window.get().width(), Window.get().height(), title, NULL, NULL);
            if (glfw_window == NULL)
            {
                throw new IllegalStateException("Error: glfwCreateWindow()");
            }

            GLFWWindowSizeCallbackI size_callback = (_, newWidth, newHeight) ->
            {
                Window.get().update_width(newWidth);
                Window.get().update_height(newHeight);
                glViewport(0, 0, newWidth, newHeight);
                event_bus.emit_event(Event.window(Event.Type.WINDOW_RESIZE));
            };

            try (var window_cb = glfwSetWindowSizeCallback(glfw_window, size_callback))
            {
                assert window_cb != null;
            }

            glfwMakeContextCurrent(glfw_window);
            //glfwSwapInterval(1); // v-sync

            org.lwjgl.opengl.GL.createCapabilities();

            glEnable(GL_BLEND);
            glEnable(GL_DEPTH_TEST);
            glDepthMask(true);
            glDepthFunc(GL_LESS);
            glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);

            glBindFramebuffer(GL_FRAMEBUFFER, 0);

            glViewport(0, 0, Window.get().width(), Window.get().height());

            load_mouse_cursor(glfw_window);

            var debugProc = GLDebugMessageCallback.create((source, type, id, severity, length, message, _) -> {
                if (severity != 33387)
                {
                    String sb = "OpenGL debug message\n" +
                        "\tID 0x" + Integer.toHexString(id).toUpperCase() + "\n" +
                        "\tSource: " + GPU.GL.get_debug_source(source) + "\n" +
                        "\tType: " + GPU.GL.get_debug_type(type) + "\n" +
                        "\tSeverity: " + GPU.GL.get_debug_severity(severity) + "\n" +
                        "\tMessage: " + GLDebugMessageCallback.getMessage(length, message) + "\n";
                    LOGGER.log(Level.WARNING, sb);
                }
            });
            GL43C.glDebugMessageCallback(debugProc, 0L);

            LOGGER.log(Level.INFO, "-------- OPEN GL DEVICE -----------");
            LOGGER.log(Level.INFO, glGetString(GL_VENDOR));
            LOGGER.log(Level.INFO, glGetString(GL_RENDERER));
            LOGGER.log(Level.INFO, glGetString(GL_VERSION));
            LOGGER.log(Level.INFO, "-----------------------------------");

            LOGGER.log(Level.FINE, "------ OPEN GL Attributes ---------");

            var int_buffer = MemoryUtil.memAllocInt(1);
            glGetIntegerv(GL_MAX_TEXTURE_SIZE, int_buffer);
            int r = int_buffer.get(0);
            LOGGER.log(Level.FINE, "max texture size: " + r);

            int_buffer.clear();
            glGetIntegerv(GL_MAX_ARRAY_TEXTURE_LAYERS, int_buffer);
            r = int_buffer.get(0);
            LOGGER.log(Level.FINE, "max texture layers: " + r);

            int_buffer.clear();
            glGetIntegerv(GL_MAX_VERTEX_ATTRIBS, int_buffer);
            r = int_buffer.get(0);
            LOGGER.log(Level.FINE, "max vertex attributes: " + r);

            int_buffer.clear();
            glGetIntegerv(GL_MAX_VERTEX_TEXTURE_IMAGE_UNITS, int_buffer);
            r = int_buffer.get(0);
            LOGGER.log(Level.FINE, "max vertex shader texture units: " + r);

            int_buffer.clear();
            glGetIntegerv(GL_MAX_TEXTURE_IMAGE_UNITS, int_buffer);
            r = int_buffer.get(0);
            LOGGER.log(Level.FINE, "max fragment shader texture units: " + r);

            LOGGER.log(Level.FINE, "-----------------------------------");

            MemoryUtil.memFree(int_buffer);

            try (var cursor_cb = glfwSetCursorPosCallback(glfw_window, inputSystem::mousePosCallback);
                 var button_cb = glfwSetMouseButtonCallback(glfw_window, inputSystem::mouseButtonCallback);
                 var scroll_cb = glfwSetScrollCallback(glfw_window, inputSystem::mouseScrollCallback);
                 var key_cb = glfwSetKeyCallback(glfw_window, inputSystem::keyCallback))
            {
                assert cursor_cb != null;
                assert button_cb != null;
                assert scroll_cb != null;
                assert key_cb != null;
            }

            return new GL_GraphicsController(glfw_window);
        }

        private static void load_mouse_cursor(long glfw_window)
        {
            var cursor_stream = Window.class.getResourceAsStream("/img/reticule_circle_blue.png");
            BufferedImage image;
            try
            {
                Objects.requireNonNull(cursor_stream);
                image = ImageIO.read(cursor_stream);
            }
            catch (IOException e)
            {
                throw new RuntimeException("Error: mouse cursor load failed", e);
            }

            int width = image.getWidth();
            int height = image.getHeight();

            int[] pixels = new int[width * height];
            image.getRGB(0, 0, width, height, pixels, 0, width);

            // convert image to RGBA format
            var cursor_buffer = MemoryUtil.memAlloc(width * height * 4);

            for (int cur_y = 0; cur_y < height; cur_y++)
            {
                for (int cur_x = 0; cur_x < width; cur_x++)
                {
                    int pixel = pixels[cur_y * width + cur_x];
                    cursor_buffer.put((byte) ((pixel >> 16) & 0xFF));  // red
                    cursor_buffer.put((byte) ((pixel >> 8) & 0xFF));   // green
                    cursor_buffer.put((byte) (pixel & 0xFF));          // blue
                    cursor_buffer.put((byte) ((pixel >> 24) & 0xFF));  // alpha
                }
            }
            cursor_buffer.flip();

            // create a GLFWImage
            var cursor_image = GLFWImage.create();
            cursor_image.width(width);     // set up image width
            cursor_image.height(height);   // set up image height
            cursor_image.pixels(cursor_buffer);   // pass image data

            // the hotspot indicates the displacement of the sprite to the
            // position where mouse clicks are registered (see image below)
            int hotspot_x = width / 2;
            int hotspot_y = height / 2;

            // create custom cursor and store its ID
            long cursor_id = org.lwjgl.glfw.GLFW.glfwCreateCursor(cursor_image, hotspot_x, hotspot_y);

            MemoryUtil.memFree(cursor_buffer);

            // set current cursor
            glfwSetCursor(glfw_window, cursor_id);
            //glfwSetInputMode(glfwWindow, GLFW_CURSOR, GLFW_CURSOR_HIDDEN);
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

        private static FT_Face loadFontFace(long ftLibrary, String fontPath)
        {
            ByteBuffer buf = null;
            try (MemoryStack stack = MemoryStack.stackPush())
            {
                PointerBuffer pp = stack.mallocPointer(1);

                var stream = GPU.class.getResourceAsStream(fontPath);
                try
                {
                    var bytes = stream.readAllBytes();
                    buf = MemoryUtil.memAlloc(bytes.length);
                    buf.put(bytes);
                    buf.flip();
                }
                catch (IOException | NullPointerException e)
                {
                    throw new RuntimeException("Could not find font: " + fontPath,e);
                }
                int error = FT_New_Memory_Face(ftLibrary, buf, 0, pp);

                if (error != 0)
                {
                    throw new RuntimeException("Error loading font: " + fontPath);
                }
                return FT_Face.create(pp.get(0));
            }
            finally
            {
                if (buf != null)
                {
                    MemoryUtil.memFree(buf);
                }
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

        public static String get_debug_source(int source)
        {
            return switch (source)
            {
                case 33350 -> "API";
                case 33351 -> "WINDOW SYSTEM";
                case 33352 -> "SHADER COMPILER";
                case 33353 -> "THIRD PARTY";
                case 33354 -> "APPLICATION";
                case 33355 -> "OTHER";
                default -> APIUtil.apiUnknownToken(source);
            };
        }

        public static String get_debug_type(int type)
        {
            return switch (type)
            {
                case 33356 -> "ERROR";
                case 33357 -> "DEPRECATED BEHAVIOR";
                case 33358 -> "UNDEFINED BEHAVIOR";
                case 33359 -> "PORTABILITY";
                case 33360 -> "PERFORMANCE";
                case 33361 -> "OTHER";
                case 33384 -> "MARKER";
                default -> APIUtil.apiUnknownToken(type);
            };
        }

        public static String get_debug_severity(int severity)
        {
            return switch (severity)
            {
                case 33387 -> "NOTIFICATION";
                case 37190 -> "HIGH";
                case 37191 -> "MEDIUM";
                case 37192 -> "LOW";
                default -> APIUtil.apiUnknownToken(severity);
            };
        }
    }
}
