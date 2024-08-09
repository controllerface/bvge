package com.controllerface.bvge.core;

import com.controllerface.bvge.ecs.ECS;
import com.controllerface.bvge.ecs.GameSystem;
import com.controllerface.bvge.editor.Editor;
import com.controllerface.bvge.events.EventBus;
import com.controllerface.bvge.game.GameMode;
import com.controllerface.bvge.game.InputSystem;
import com.controllerface.bvge.game.TestGame;
import com.controllerface.bvge.gpu.GPU;
import com.controllerface.bvge.gpu.gl.GL_GraphicsController;
import com.controllerface.bvge.rendering.Camera;
import org.joml.Vector2f;

import java.util.logging.Level;
import java.util.logging.Logger;

import static org.lwjgl.glfw.GLFW.*;
import static org.lwjgl.opengl.GL11C.*;

/**
 * The Window is intended to be globally accessible
 */
public class Window
{
    private static final Logger LOGGER = Logger.getLogger(Window.class.getName());

    private int width;
    private int height;
    private final String title;

    private static Window INSTANCE = null;

    private GL_GraphicsController gl_window;

    public float r, g, b, a;

    private static GameMode game_mode;
    private ECS ecs;
    private final Camera camera;

    private boolean closing = false;

    private final EventBus event_bus;
    private InputSystem input_system;

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
        this.a = 1;
    }

    public void update_width(int new_width)
    {
        this.width = new_width;
    }

    public void update_height(int new_height)
    {
        this.height = new_height;
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
        gl_window.show_window();
        loop();

        closing = true;

        gl_window.release();

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

    public void init(ECS ecs)
    {
        this.ecs = ecs;
        this.input_system = new InputSystem(ecs);
        gl_window = GPU.GL.init_gl(this.title, event_bus, input_system);
    }

    public void init_game_mode()
    {
        game_mode = new TestGame(ecs, this::window_upkeep);
        game_mode.init();
        camera.projection_size().set(this.width, this.height);
        ecs.register_system(input_system);
    }

    public Camera camera()
    {
        return camera;
    }

    private static final float MAX_DT = 100.0f;

    // this is the main game loop
    public void loop()
    {
        LOGGER.log(Level.FINE, "Starting Game loop");

        float lastTime = (float) glfwGetTime();
        float currentTime;
        float dt = -1.0f;
        float fpsTimer = 0.0f;
        int fps;
        int frameCount = 0;

        while (!gl_window.should_close() && dt < MAX_DT)
        {
            glfwPollEvents();

            if (dt >= 0)
            {
                ecs.tick(dt);
                gl_window.swap_buffers();
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

        game_mode.destroy();

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
