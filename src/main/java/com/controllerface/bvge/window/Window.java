package com.controllerface.bvge.window;

import com.controllerface.bvge.ecs.ECS;
import com.controllerface.bvge.ecs.systems.GameSystem;
import com.controllerface.bvge.ecs.systems.KBMInput;
import com.controllerface.bvge.editor.Editor;
import com.controllerface.bvge.game.GameMode;
import com.controllerface.bvge.game.TestGame;
import org.joml.Vector2f;
import org.lwjgl.Version;
import org.lwjgl.glfw.GLFWErrorCallback;
import org.lwjgl.glfw.GLFWImage;
import org.lwjgl.glfw.GLFWWindowContentScaleCallbackI;
import org.lwjgl.glfw.GLFWWindowSizeCallbackI;
import org.lwjgl.opengl.GL;
import org.lwjgl.system.MemoryUtil;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.util.Objects;

import static org.lwjgl.glfw.Callbacks.glfwFreeCallbacks;
import static org.lwjgl.glfw.GLFW.*;
import static org.lwjgl.opengl.GL11C.*;
import static org.lwjgl.opengl.GL20C.GL_MAX_VERTEX_TEXTURE_IMAGE_UNITS;
import static org.lwjgl.opengl.GL20C.GL_MAX_TEXTURE_IMAGE_UNITS;
import static org.lwjgl.opengl.GL20C.GL_MAX_VERTEX_ATTRIBS;
import static org.lwjgl.opengl.GL30C.*;
import static org.lwjgl.system.MemoryUtil.NULL;

/**
 * The Window is intended to be globally accessible
 */
public class Window
{
    private int width;
    private int height;
    private final String title;

    private static Window INSTANCE = null;

    private long glfwWindow;

    public float r, g, b, a;

    private static GameMode currentGameMode;
    private final ECS ecs = new ECS();
    private final Camera camera;

    private boolean closing = false;

    private final EventBus event_bus;

    private Window()
    {
        this.event_bus = new EventBus();

        this.width = 1920;
        this.height = 1080;
        this.title = "BVGE Test";

        camera = new Camera(new Vector2f(0, 0), height, width);

        this.r = 0.1f;
        this.g = 0.1f;
        this.b = 0.1f;

//        this.r = 0.0f;
//        this.g = 0.0f;
//        this.b = 0.0f;

        this.a = 1;
    }

    public static Window get()
    {
        if (Window.INSTANCE == null)
        {
            Window.INSTANCE = new Window();
        }
        return Window.INSTANCE;
    }

    public EventBus event_bus()
    {
        return event_bus;
    }

    public boolean is_closing()
    {
        return closing;
    }

    public void run()
    {
        glfwShowWindow(glfwWindow);
        loop();

        closing = true;

        ecs.shutdown();

        glfwFreeCallbacks(glfwWindow);
        glfwDestroyWindow(glfwWindow);

        glfwTerminate();
        try (var error_cb = glfwSetErrorCallback(null))
        {
            assert error_cb != null;
        }
    }

    private void window_upkeep()
    {
        glClearColor(r, g, b, a);
        camera.adjust_projection(this.height, this.width);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    }

    public void init()
    {
        GLFWErrorCallback.createPrint(System.err).set();

        if (!glfwInit())
        {
            throw new IllegalStateException("could not init GLFW");
        }

        glfwDefaultWindowHints();
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
        glfwWindowHint(GLFW_VISIBLE, GLFW_TRUE);
        glfwWindowHint(GLFW_MAXIMIZED, GLFW_FALSE);
        glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);
        glfwWindowHint(GLFW_DECORATED, GLFW_FALSE);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

        var primary_monitor = glfwGetPrimaryMonitor();

        // example for how to get area with start menu accounted for
        // todo: use this for windowed mode
//        int[] x = new int[1];
//        int[] y = new int[1];
//        int[] w = new int[1];
//        int[] h = new int[1];
//        glfwGetMonitorWorkarea(primary_monitor, x,y,w,h);
//        this.width = w[0];
//        this.height = h[0];




        var video_mode = glfwGetVideoMode(primary_monitor);
        if (video_mode != null)
        {
            this.width = video_mode.width();
            this.height = video_mode.height();
        }

        glfwWindow = glfwCreateWindow(this.width, this.height, this.title, NULL, NULL);
        if (glfwWindow == NULL)
        {
            throw new IllegalStateException("could not create window");
        }

        GLFWWindowSizeCallbackI size_callback = (win, newWidth, newHeight) ->
        {
            get().width = newWidth;
            get().height = newHeight;
            System.out.println("size: x="+newWidth+" y="+newHeight);
            event_bus.report_event(EventType.WINDOW_RESIZE);
        };

        try (var window_cb = glfwSetWindowSizeCallback(glfwWindow, size_callback))
        {
            assert window_cb != null;
        }

        GLFWWindowContentScaleCallbackI scale_callback = (win, newScaleX, newScaleY) ->
        {
            System.out.println("scale: x="+newScaleX+" y="+newScaleY);
        };

        glfwSetWindowContentScaleCallback(glfwWindow, scale_callback);

        glfwMakeContextCurrent(glfwWindow);
        //glfwSwapInterval(1); // v-sync

        // note: this must be called or nothing will work
        GL.createCapabilities();

        glEnable(GL_BLEND);
        glEnable(GL_DEPTH_TEST);
        glDepthMask(true);
        glDepthFunc(GL_LESS);
        glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
        //glDepthRange(0.0, 1.0);

        // Create and bind the default framebuffer (typically 0)
        glBindFramebuffer(GL_FRAMEBUFFER, 0);

        glViewport(0, 0, this.width, this.height);

        // mouse cursor
        var cursor_stream = Window.class.getResourceAsStream("/img/reticule_circle_blue.png");
        BufferedImage image;
        try
        {
            Objects.requireNonNull(cursor_stream);
            image = ImageIO.read(cursor_stream);
        }
        catch (IOException e)
        {
            assert false : "Couldn't load mouse icon";
            throw new RuntimeException(e);
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
        glfwSetCursor(glfwWindow, cursor_id);
        //glfwSetInputMode(glfwWindow, GLFW_CURSOR, GLFW_CURSOR_HIDDEN);


        var int_buffer = MemoryUtil.memAllocInt(1);
        glGetIntegerv(GL_MAX_TEXTURE_SIZE, int_buffer);
        int r = int_buffer.get(0);
        System.out.println("max texture size: " + r);

        int_buffer.clear();
        glGetIntegerv(GL_MAX_ARRAY_TEXTURE_LAYERS, int_buffer);
        r = int_buffer.get(0);
        System.out.println("max texture layers: " + r);

        int_buffer.clear();
        glGetIntegerv(GL_MAX_VERTEX_ATTRIBS, int_buffer);
        r = int_buffer.get(0);
        System.out.println("max vertex attributes: " + r);

        int_buffer.clear();
        glGetIntegerv(GL_MAX_VERTEX_TEXTURE_IMAGE_UNITS, int_buffer);
        r = int_buffer.get(0);
        System.out.println("max vertex shader texture units: " + r);

        int_buffer.clear();
        glGetIntegerv(GL_MAX_TEXTURE_IMAGE_UNITS, int_buffer);
        r = int_buffer.get(0);
        System.out.println("max fragment shader texture units: " +r);

        var int_buffer2 = MemoryUtil.memAllocInt(2);
        glGetIntegerv(GL_POINT_SIZE_RANGE, int_buffer2);
        int a = int_buffer2.get(0);
        int b = int_buffer2.get(1);
        System.out.println("point size range: " +a+" - "+b);

        MemoryUtil.memFree(int_buffer);
        MemoryUtil.memFree(int_buffer2);

        System.out.println("LWJGL version: " + Version.getVersion());

        System.out.println("\n-------- OPEN GL DEVICE -----------");
        System.out.println(glGetString(GL_VENDOR));
        System.out.println(glGetString(GL_RENDERER));
        System.out.println(glGetString(GL_VERSION));
        System.out.println("-----------------------------------\n");
    }

    private void init_input(KBMInput inputSystem)
    {
        try (var cursor_cb = glfwSetCursorPosCallback(glfwWindow, inputSystem::mousePosCallback);
             var button_cb = glfwSetMouseButtonCallback(glfwWindow, inputSystem::mouseButtonCallback);
             var scroll_cb = glfwSetScrollCallback(glfwWindow, inputSystem::mouseScrollCallback);
             var key_cb = glfwSetKeyCallback(glfwWindow, inputSystem::keyCallback))
        {
            assert cursor_cb != null;
            assert button_cb != null;
            assert scroll_cb != null;
            assert key_cb != null;
        }
    }

    /**
     * A simple utility system that just blanks the screen, getting it ready to render. This is used
     * instead of just calling it at the top of the loop, so that the screen clear can happen s late
     * as possible, just before rendering. todo: it may be better to use a framebuffer of some kind
     */
    private class BlankSystem extends GameSystem
    {
        public BlankSystem(ECS ecs)
        {
            super(ecs);
        }

        @Override
        public void tick(float dt)
        {
            window_upkeep();
        }
    }

    public void init_game_mode()
    {
        var blanking_system = new BlankSystem(null);
        currentGameMode = new TestGame(ecs, blanking_system);
        currentGameMode.load();
        currentGameMode.start();

        camera.projection_size().x = this.width;
        camera.projection_size().y = this.height;

        // order of system registry is important, systems run in the order they are added
        var inputSystem = new KBMInput(ecs);
        ecs.register_system(inputSystem);

        init_input(inputSystem);
    }

    public Camera camera()
    {
        return camera;
    }

    private static final float MAX_DT = 100.0f;

    // this is the main game loop
    public void loop()
    {
        float lastTime = (float) glfwGetTime();
        float currentTime;
        float dt = -1.0f;
        float fpsTimer = 0.0f;
        int fps;
        int frameCount = 0;

        while (!glfwWindowShouldClose(glfwWindow) && dt < MAX_DT)
        {
            glfwPollEvents();

            if (dt >= 0)
            {
                currentGameMode.update(dt);
                ecs.tick(dt);
                glfwSwapBuffers(glfwWindow);
            }

            currentTime = (float) glfwGetTime();
            dt = currentTime - lastTime;
            lastTime = currentTime;

            if (Editor.ACTIVE)
            {
                Editor.queue_event("dt", String.valueOf(dt));
            }

            if (Editor.ACTIVE)
            {
                frameCount++;
                fpsTimer += dt;
                if (fpsTimer >= 1.0)
                {
                    fps = frameCount;
                    frameCount = 0;
                    fpsTimer -= 1.0;
                    Editor.queue_event("fps", String.valueOf(fps));
                }
            }
        }
        if (dt >= MAX_DT)
        {
            System.err.println("excessive frame time: " + dt);
        }
    }

    public int width()
    {
        return width;
    }

    public int height()
    {
        return height;
    }
}
