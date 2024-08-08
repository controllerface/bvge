package com.controllerface.bvge.gpu;

import com.controllerface.bvge.core.Window;
import com.controllerface.bvge.events.Event;
import com.controllerface.bvge.events.EventBus;
import com.controllerface.bvge.game.InputSystem;
import com.controllerface.bvge.gpu.gl.GL_GraphicsController;
import com.controllerface.bvge.gpu.gl.shaders.GL_Shader;
import com.controllerface.bvge.gpu.gl.shaders.ThreeStageShader;
import com.controllerface.bvge.gpu.gl.shaders.TwoStageShader;
import com.controllerface.bvge.gpu.gl.shaders.GL_ShaderType;
import org.lwjgl.glfw.GLFWErrorCallback;
import org.lwjgl.glfw.GLFWImage;
import org.lwjgl.glfw.GLFWWindowSizeCallbackI;
import org.lwjgl.opengl.GL43C;
import org.lwjgl.opengl.GLDebugMessageCallback;
import org.lwjgl.system.APIUtil;
import org.lwjgl.system.MemoryUtil;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.util.Objects;
import java.util.logging.Level;
import java.util.logging.Logger;

import static org.lwjgl.glfw.GLFW.*;
import static org.lwjgl.opengl.GL11C.*;
import static org.lwjgl.opengl.GL20C.*;
import static org.lwjgl.opengl.GL30C.*;
import static org.lwjgl.system.MemoryUtil.NULL;

public class GPU
{
    private static final Logger LOGGER = Logger.getLogger(GPU.class.getName());

    public static class GL
    {
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
