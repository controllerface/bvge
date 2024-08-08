package com.controllerface.bvge.gpu;

import com.controllerface.bvge.core.Window;
import com.controllerface.bvge.events.Event;
import com.controllerface.bvge.events.EventBus;
import com.controllerface.bvge.game.InputSystem;
import com.controllerface.bvge.gpu.cl.CL_ComputeController;
import com.controllerface.bvge.gpu.cl.buffers.CL_DataTypes;
import com.controllerface.bvge.gpu.cl.contexts.CL_CommandQueue;
import com.controllerface.bvge.gpu.cl.contexts.CL_Context;
import com.controllerface.bvge.gpu.cl.devices.CL_Device;
import com.controllerface.bvge.gpu.cl.kernels.KernelArg;
import com.controllerface.bvge.gpu.cl.kernels.KernelType;
import com.controllerface.bvge.gpu.gl.GL_GraphicsController;
import com.controllerface.bvge.gpu.gl.buffers.GL_CommandBuffer;
import com.controllerface.bvge.gpu.gl.buffers.GL_ElementBuffer;
import com.controllerface.bvge.gpu.gl.buffers.GL_VertexArray;
import com.controllerface.bvge.gpu.gl.buffers.GL_VertexBuffer;
import com.controllerface.bvge.gpu.gl.shaders.GL_Shader;
import com.controllerface.bvge.gpu.gl.shaders.ThreeStageShader;
import com.controllerface.bvge.gpu.gl.shaders.TwoStageShader;
import com.controllerface.bvge.gpu.gl.shaders.GL_ShaderType;
import com.controllerface.bvge.gpu.gl.textures.GL_Texture2D;
import com.controllerface.bvge.rendering.TextGlyph;
import org.joml.Matrix4f;
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
import java.nio.IntBuffer;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.Map;
import java.util.Objects;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.stream.Collectors;

import static com.controllerface.bvge.game.Constants.Rendering.*;
import static org.lwjgl.glfw.GLFW.*;
import static org.lwjgl.opencl.AMDDeviceAttributeQuery.CL_DEVICE_WAVEFRONT_WIDTH_AMD;
import static org.lwjgl.opencl.CL10.*;
import static org.lwjgl.opencl.CL11.CL_DEVICE_HOST_UNIFIED_MEMORY;
import static org.lwjgl.opencl.KHRGLSharing.CL_GL_CONTEXT_KHR;
import static org.lwjgl.opencl.KHRGLSharing.CL_WGL_HDC_KHR;
import static org.lwjgl.opencl.NVDeviceAttributeQuery.CL_DEVICE_WARP_SIZE_NV;
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
import static org.lwjgl.opengl.WGL.wglGetCurrentContext;
import static org.lwjgl.opengl.WGL.wglGetCurrentDC;
import static org.lwjgl.system.MemoryUtil.NULL;
import static org.lwjgl.util.freetype.FreeType.*;
import static org.lwjgl.util.harfbuzz.HarfBuzz.*;

public class GPU
{
    private static final Logger LOGGER = Logger.getLogger(GPU.class.getName());

    public static class GL
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

        public static GL_VertexBuffer new_buffer_float(GL_VertexArray vao,
                                                       int bind_index,
                                                       int buffer_size)
        {
            return dynamic_float_buffer(vao, bind_index, buffer_size, SCALAR_LENGTH, SCALAR_FLOAT_SIZE);
        }

        public static GL_VertexBuffer new_buffer_vec2(GL_VertexArray vao,
                                                      int bind_index,
                                                      int buffer_size)
        {
            return dynamic_float_buffer(vao, bind_index, buffer_size, VECTOR_2D_LENGTH, VECTOR_FLOAT_2D_SIZE);
        }

        public static GL_VertexBuffer new_buffer_vec4(GL_VertexArray vao,
                                                      int bind_index,
                                                      int buffer_size)
        {
            return dynamic_float_buffer(vao, bind_index, buffer_size, VECTOR_4D_LENGTH, VECTOR_FLOAT_4D_SIZE);
        }

        public static GL_VertexBuffer new_vec2_buffer_static(GL_VertexArray vao,
                                                             int bind_index,
                                                             float[] buffer_data)
        {
            return static_float_buffer(vao, bind_index, VECTOR_2D_LENGTH, VECTOR_FLOAT_2D_SIZE, buffer_data);
        }


        private static GL_VertexBuffer static_float_buffer(GL_VertexArray vao,
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

        private static GL_VertexBuffer dynamic_float_buffer(GL_VertexArray vao,
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

        private static GL_VertexBuffer create_vertex_buffer_float(GL_VertexArray vao,
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
            int vbo = glCreateBuffers();
            glNamedBufferData(vbo, buffer_data, flags);
            glVertexArrayVertexBuffer(vao.gl_id(), bind_index, vbo, buffer_offset, data_size);
            glVertexArrayAttribFormat(vao.gl_id(), bind_index, data_count, data_type, false, data_stride);
            glVertexArrayAttribBinding(vao.gl_id(), attribute_index, bind_index);
            return new GL_VertexBuffer(vbo);
        }

        private static GL_VertexBuffer create_vertex_buffer(GL_VertexArray vao,
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
            int vbo = glCreateBuffers();
            glNamedBufferData(vbo, buffer_size, flags);
            glVertexArrayVertexBuffer(vao.gl_id(), bind_index, vbo, buffer_offset, data_size);
            glVertexArrayAttribFormat(vao.gl_id(), bind_index, data_count, data_type, false, data_stride);
            glVertexArrayAttribBinding(vao.gl_id(), attribute_index, bind_index);
            return new GL_VertexBuffer(vbo);
        }

        public static GL_ElementBuffer dynamic_element_buffer(GL_VertexArray vao, int buffer_size)
        {
            int ebo = glCreateBuffers();
            glNamedBufferData(ebo, buffer_size, GL_DYNAMIC_DRAW);
            glVertexArrayElementBuffer(vao.gl_id(), ebo);
            return new GL_ElementBuffer(ebo);
        }

        public static GL_CommandBuffer dynamic_command_buffer(GL_VertexArray vao, int buffer_size)
        {
            int cbo = glCreateBuffers();
            glNamedBufferData(cbo, buffer_size, GL_DYNAMIC_DRAW);
            glBindVertexArray(vao.gl_id());
            glBindBuffer(GL_DRAW_INDIRECT_BUFFER, cbo);
            glBindVertexArray(0);
            return new GL_CommandBuffer(cbo);
        }

        private static GL_VertexBuffer create_vertex_buffer(int vao,
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
            int vbo = glCreateBuffers();
            glNamedBufferData(vbo, buffer_size, flags);
            glVertexArrayVertexBuffer(vao, bind_index, vbo, buffer_offset, data_size);
            glVertexArrayAttribFormat(vao, bind_index, data_count, data_type, false, data_stride);
            glVertexArrayAttribBinding(vao, attribute_index, bind_index);
            return new GL_VertexBuffer(vbo);
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

    public static class CL
    {
        public static final String BUFFER_PREFIX = "__global";
        public static final String BUFFER_SUFFIX = "*";

        public static short[] arg_short2(short x, short y)
        {
            return new short[]{x, y};
        }

        public static int[] arg_int2(int x, int y)
        {
            return new int[]{x, y};
        }

        public static int[] arg_int4(int x, int y, int z, int w)
        {
            return new int[]{x, y, z, w};
        }

        public static long[] arg_long(long arg)
        {
            return new long[]{arg};
        }

        public static float[] arg_float2(float x, float y)
        {
            return new float[]{x, y};
        }

        public static float[] arg_float4(float x, float y, float z, float w)
        {
            return new float[]{x, y, z, w};
        }

        public static float[] arg_float16(float s0, float s1, float s2, float s3,
                                          float s4, float s5, float s6, float s7,
                                          float s8, float s9, float sA, float sB,
                                          float sC, float sD, float sE, float sF)
        {
            return new float[]
                {
                    s0, s1, s2, s3,
                    s4, s5, s6, s7,
                    s8, s9, sA, sB,
                    sC, sD, sE, sF
                };
        }

        public static float[] arg_float16_matrix(Matrix4f matrix)
        {
            return arg_float16(
                matrix.m00(), matrix.m01(), matrix.m02(), matrix.m03(),
                matrix.m10(), matrix.m11(), matrix.m12(), matrix.m13(),
                matrix.m20(), matrix.m21(), matrix.m22(), matrix.m23(),
                matrix.m30(), matrix.m31(), matrix.m32(), matrix.m33());
        }

        public static CL_Device init_device()
        {
            int result;

            int[] numPlatformsArray = new int[1];
            result = clGetPlatformIDs(null, numPlatformsArray);
            if (result != CL_SUCCESS)
            {
                throw new RuntimeException("Error: clGetPlatformIDs(): " + result);
            }

            int numPlatforms = numPlatformsArray[0];
            var platform_buffer = MemoryUtil.memAllocPointer(numPlatforms);
            result = clGetPlatformIDs(platform_buffer, (IntBuffer) null);
            if (result != CL_SUCCESS)
            {
                throw new RuntimeException("Error: clGetPlatformIDs(): " + result);
            }

            var platform = platform_buffer.get();
            MemoryUtil.memFree(platform_buffer);
            int[] numDevicesArray = new int[1];
            result = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, null, numDevicesArray);
            if (result != CL_SUCCESS)
            {
                throw new RuntimeException("Error: clGetDeviceIDs(): " + result);
            }

            int numDevices = numDevicesArray[0];
            var device_buffer = MemoryUtil.memAllocPointer(numDevices);
            result = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, device_buffer, (IntBuffer) null);
            if (result != CL_SUCCESS)
            {
                throw new RuntimeException("Error: clGetDeviceIDs(): " + result);
            }

            long device = device_buffer.get();
            MemoryUtil.memFree(device_buffer);
            return new CL_Device(device, platform);
        }

        public static CL_Context new_context(CL_Device device)
        {
            try (var stack = MemoryStack.stackPush())
            {
                var dc = wglGetCurrentDC();
                var ctx = wglGetCurrentContext();
                var ctx_props_buffer = stack.mallocPointer(7);
                var return_code = stack.mallocInt(1);
                ctx_props_buffer.put(CL_CONTEXT_PLATFORM)
                    .put(device.platform())
                    .put(CL_GL_CONTEXT_KHR)
                    .put(ctx)
                    .put(CL_WGL_HDC_KHR)
                    .put(dc)
                    .put(0L)
                    .flip();

                // todo: the above code is windows specific add linux code path,
                //  should look something like this:
                // var ctx = glXGetCurrentContext();
                // var dc = glXGetCurrentDrawable(); OR glfwGetX11Display();
                // contextProperties.addProperty(CL_GLX_DISPLAY_KHR, dc);

                long ptr_context = clCreateContext(ctx_props_buffer, device.ptr(), null, 0L, return_code);
                int result = return_code.get(0);
                if (result != CL_SUCCESS)
                {
                    throw new RuntimeException("Error: clCreateContext(): " + result);
                }
                return new CL_Context(ptr_context);
            }
        }

        public static CL_CommandQueue new_command_queue(CL_Context context, CL_Device device)
        {
            try (var stack = MemoryStack.stackPush())
            {
                var return_code = stack.mallocInt(1);
                var queue_ptr = clCreateCommandQueue(context.ptr(), device.ptr(), 0, return_code);
                int result = return_code.get(0);
                if (result != CL_SUCCESS)
                {
                    throw new RuntimeException("Error: clCreateCommandQueue(): " + result);
                }
                return new CL_CommandQueue(queue_ptr);
            }
        }

        public static CL_ComputeController init_cl()
        {
            var device = init_device();
            var context = new_context(device);
            var compute_queue = new_command_queue(context, device);
            var render_queue = new_command_queue(context, device);
            var sector_queue = new_command_queue(context, device);


            LOGGER.log(Level.INFO, "-------- OPEN CL DEVICE -----------");
            LOGGER.log(Level.INFO, get_device_string(device.ptr(), CL_DEVICE_VENDOR));
            LOGGER.log(Level.INFO, get_device_string(device.ptr(), CL_DEVICE_NAME));
            LOGGER.log(Level.INFO, get_device_string(device.ptr(), CL_DRIVER_VERSION));
            LOGGER.log(Level.INFO, "-----------------------------------");

            long max_local_buffer_size = get_device_long(device.ptr(), CL_DEVICE_LOCAL_MEM_SIZE);
            long current_max_group_size = get_device_long(device.ptr(), CL_DEVICE_MAX_WORK_GROUP_SIZE);
            long compute_unit_count = get_device_long(device.ptr(), CL_DEVICE_MAX_COMPUTE_UNITS);
            long wavefront_width = get_device_long(device.ptr(), CL_DEVICE_WAVEFRONT_WIDTH_AMD);
            long warp_width = get_device_long(device.ptr(), CL_DEVICE_WARP_SIZE_NV);

            long[] preferred_work_size = arg_long(wavefront_width != -1
                ? wavefront_width
                : warp_width != -1
                    ? warp_width
                    : 32);

            int preferred_work_size_int = (int) preferred_work_size[0];

            long current_max_block_size = current_max_group_size * 2;

            long max_mem = get_device_long(device.ptr(), CL_DEVICE_MAX_MEM_ALLOC_SIZE);
            long sz_char = get_device_long(device.ptr(), CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR);
            long sz_flt = get_device_long(device.ptr(), CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT);
            boolean non_uniform = get_device_boolean(device.ptr(), CL_DEVICE_HOST_UNIFIED_MEMORY);


            LOGGER.log(Level.FINE, "------ OPEN CL Attributes ---------");

            LOGGER.log(Level.FINE, "CL_DEVICE_MAX_COMPUTE_UNITS: " + compute_unit_count);
            LOGGER.log(Level.FINE, "CL_DEVICE_WAVEFRONT_WIDTH_AMD: " + wavefront_width);
            LOGGER.log(Level.FINE, "CL_DEVICE_WARP_SIZE_NV: " + warp_width);

            LOGGER.log(Level.FINE, "CL_DEVICE_LOCAL_MEM_SIZE: " + max_local_buffer_size);
            LOGGER.log(Level.FINE, "CL_DEVICE_MAX_WORK_GROUP_SIZE: " + current_max_group_size);
            LOGGER.log(Level.FINE, "CL_DEVICE_NON_UNIFORM_WORK_GROUP_SUPPORT: " + non_uniform);

            LOGGER.log(Level.FINE, "CL_DEVICE_MAX_MEM_ALLOC_SIZE: " + max_mem);
            LOGGER.log(Level.FINE, "preferred float: " + sz_flt);
            LOGGER.log(Level.FINE, "preferred char: " + sz_char);

            LOGGER.log(Level.FINE, "-----------------------------------");

            long int2_max = CL_DataTypes.cl_int2.size() * current_max_block_size;
            long int4_max = CL_DataTypes.cl_int4.size() * current_max_block_size;
            long size_cap = int2_max + int4_max;

            while (size_cap >= max_local_buffer_size)
            {
                current_max_group_size /= 2;
                current_max_block_size = current_max_group_size * 2;
                int2_max = CL_DataTypes.cl_int2.size() * current_max_block_size;
                int4_max = CL_DataTypes.cl_int4.size() * current_max_block_size;
                size_cap = int2_max + int4_max;
            }

            long max_work_group_size = current_max_group_size;
            long max_scan_block_size = current_max_block_size;
            long[] local_work_default = arg_long(max_work_group_size);

            return new CL_ComputeController(max_work_group_size,
                max_scan_block_size,
                preferred_work_size,
                preferred_work_size_int,
                local_work_default,
                device,
                context,
                compute_queue,
                render_queue,
                sector_queue);
        }

        public static String read_src(String file)
        {
            try (var stream = GPU.class.getResourceAsStream("/cl/" + file))
            {
                byte[] bytes = Objects.requireNonNull(stream).readAllBytes();
                return new String(bytes, StandardCharsets.UTF_8);
            }
            catch (NullPointerException | IOException e)
            {
                throw new RuntimeException("Could not load kernel source: " + file, e);
            }
        }

        /**
         * Generates source code for an Open CL kernel that can be called to create or update an object
         * using the specified arguments as input. Run any of the tests in the following package to see
         * example kernel output: {@link com.controllerface.bvge.gpu.cl.kernels.crud}
         *
         * @param kernel    {@linkplain KernelType} enum class identifying the name of the kernel to generate
         * @param args_enum {@linkplain KernelArg} enum class defining the ordered set of argument to the kernel
         * @param <E>       Type argument restricting the args_enum implementation
         * @return a String containing the generated kernel source
         */
        public static <E extends Enum<E> & KernelArg> String crud_create_k_src(KernelType kernel, Class<E> args_enum)
        {
            E[] arguments = args_enum.getEnumConstants();

        /*
        By convention, all Create* kernels define an argument named `target` which is used to determine the
        target position within the GPU buffer where the object being created will be stored. if missing,
        it is considered a critical failure.
         */
            int target_index = Arrays.stream(args_enum.getEnumConstants())
                .filter(arg -> arg.name().equals("target"))
                .map(Enum::ordinal)
                .findAny()
                .orElseThrow();

            var src = new StringBuilder("__kernel void ").append(kernel.name());

            var parameters = Arrays.stream(args_enum.getEnumConstants())
                .map(arg -> String.join(" ", arg.cl_type(), arg.name()))
                .collect(Collectors.joining(",\n\t", "(", ")\n"));

            src.append(parameters);
            src.append("{\n");
            for (int name_index = 0; name_index < target_index; name_index++)
            {
                int value_index = name_index + target_index + 1;
                var name = arguments[name_index].name();
                var value = arguments[value_index];

                src.append("\t")
                    .append(name)
                    .append("[target] = ")
                    .append(value)
                    .append(";\n");
            }
            src.append("}\n\n");
            return src.toString();
        }

        /**
         * Generates source code for an Open CL kernel that can be called to compact the buffers used
         * for logical objects. Run any of the tests in the following package to see example kernel
         * output: {@link com.controllerface.bvge.gpu.cl.kernels.compact}
         *
         * @param kernel    {@linkplain KernelType} enum class identifying the name of the kernel to generate
         * @param args_enum {@linkplain KernelArg} enum class defining the ordered set of arguments to the kernel
         * @param <E>       Type argument restricting the args_enum implementation
         * @return a String containing the generated kernel source
         */
        public static <E extends Enum<E> & KernelArg> String compact_k_src(KernelType kernel, Class<E> args_enum)
        {
            E[] arguments = args_enum.getEnumConstants();

            var src = new StringBuilder("__kernel void ").append(kernel.name());

            var parameters = Arrays.stream(args_enum.getEnumConstants())
                .map(arg -> String.join(" ", arg.cl_type(), arg.name()))
                .collect(Collectors.joining(",\n\t", "(", ")\n"));

            src.append(parameters);
            src.append("{\n");
            src.append("\tint current = get_global_id(0);\n");
            src.append("\tint shift = ").append(arguments[0].name()).append("[current];\n");

            for (int arg_index = 1; arg_index < arguments.length; arg_index++)
            {
                var arg = arguments[arg_index];
                var type = arg.cl_type()
                    .replace(BUFFER_PREFIX, "")
                    .replace(BUFFER_SUFFIX, "")
                    .trim();
                var _name = "_" + arg.name();
                src.append("\t")
                    .append(type).append(" ")
                    .append(_name).append(" = ")
                    .append(arg.name()).append("[current]")
                    .append(";\n");
            }
            src.append("\tbarrier(CLK_GLOBAL_MEM_FENCE);\n");
            src.append("\tif (shift > 0)\n");
            src.append("\t{\n");
            src.append("\t\tint new_index = current - shift;\n");

            for (int arg_index = 1; arg_index < arguments.length; arg_index++)
            {
                var arg = arguments[arg_index];
                var _name = "_" + arg.name();
                src.append("\t\t")
                    .append(arg.name()).append("[new_index]")
                    .append(" = ")
                    .append(_name)
                    .append(";\n");
            }

            src.append("\t}\n");
            src.append("}\n\n");
            return src.toString();
        }

        private static boolean get_device_boolean(long device_ptr, int param_code)
        {
            var size_buffer = MemoryUtil.memAllocPointer(1);
            clGetDeviceInfo(device_ptr, param_code, (long[]) null, size_buffer);
            long size = size_buffer.get();
            var value_buffer = MemoryUtil.memAlloc((int) size);
            clGetDeviceInfo(device_ptr, param_code, value_buffer, null);
            var result = value_buffer.get();

            MemoryUtil.memFree(size_buffer);
            MemoryUtil.memFree(value_buffer);
            return result == 1;
        }

        private static long get_device_long(long device_ptr, int param_code)
        {
            try (var stack = MemoryStack.stackPush())
            {
                int result;

                var size_buffer = stack.mallocPointer(1);
                result = clGetDeviceInfo(device_ptr, param_code, (long[]) null, size_buffer);
                if (result != CL_SUCCESS)
                {
                    return -1;
                }

                long size = size_buffer.get();
                var value_buffer = MemoryUtil.memAlloc((int) size);
                result = clGetDeviceInfo(device_ptr, param_code, value_buffer, null);
                if (result != CL_SUCCESS)
                {
                    return -1;
                }

                long value = size == 4
                    ? value_buffer.getInt(0)
                    : value_buffer.getLong(0);

                MemoryUtil.memFree(value_buffer);
                return value;
            }
        }

        private static String get_device_string(long device_ptr, int param_code)
        {
            try (var stack = MemoryStack.stackPush())
            {
                int result;

                var size_buffer = stack.mallocPointer(1);
                result = clGetDeviceInfo(device_ptr, param_code, (long[]) null, size_buffer);
                if (result != CL_SUCCESS)
                {
                    throw new RuntimeException("Error: clGetDeviceInfo(): " + result);
                }

                long size = size_buffer.get();
                var value_buffer = MemoryUtil.memAlloc((int) size);
                byte[] bytes = new byte[(int) size];
                result = clGetDeviceInfo(device_ptr, param_code, value_buffer, null);
                if (result != CL_SUCCESS)
                {
                    throw new RuntimeException("Error: clGetDeviceInfo(): " + result);
                }

                value_buffer.get(bytes);
                MemoryUtil.memFree(value_buffer);
                return new String(bytes, 0, bytes.length - 1);
            }
        }
    }
}
