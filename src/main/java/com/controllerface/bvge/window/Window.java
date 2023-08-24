package com.controllerface.bvge.window;

import com.controllerface.bvge.ecs.ECS;
import com.controllerface.bvge.ecs.systems.GameSystem;
import com.controllerface.bvge.ecs.systems.KBMInput;
import com.controllerface.bvge.game.GameMode;
import com.controllerface.bvge.game.TestGame;
import org.joml.Vector2f;
import org.lwjgl.Version;
import org.lwjgl.glfw.GLFWErrorCallback;
import org.lwjgl.glfw.GLFWVidMode;
import org.lwjgl.opengl.GL;

import static org.lwjgl.glfw.Callbacks.glfwFreeCallbacks;
import static org.lwjgl.glfw.GLFW.*;
import static org.lwjgl.opengl.GL11.*;
import static org.lwjgl.system.MemoryUtil.NULL;

/**
 * The Window is intended to be globally accessible
 */
public class Window
{
    int width, height;
    String title;

    private static Window INSTANCE = null;

    private long glfwWindow;

    //private PickingTexture pickingTexture;

    public float r, g, b, a;

    private static GameMode currentGameMode;
    private final ECS ecs = new ECS();
    private final Camera camera = new Camera(new Vector2f(0, 0));

    private Window()
    {
        this.width = 1920;
        this.height = 1080;
        this.title = "BVGE Test";
        this.r = 0.0f;
        this.g = 0.0f;
        this.b = 0.0f;
        this.a = 1;
    }

    public static Window get()
    {
        if (Window.INSTANCE == null)
        {
            Window.INSTANCE = new Window();
            //Window.INSTANCE.init();
        }
        return Window.INSTANCE;
    }

    public void run()
    {
        System.out.println("LWJGL version: " + Version.getVersion());

        glfwShowWindow(glfwWindow);
        loop();

        ecs.shutdown();

        glfwFreeCallbacks(glfwWindow);
        glfwDestroyWindow(glfwWindow);

        glfwTerminate();
        glfwSetErrorCallback(null).free();
    }

    private void windowUpkeep()
    {
        glClearColor(r, g, b, a);
        glClear(GL_COLOR_BUFFER_BIT);
        camera.adjustProjection();
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
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
        glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
        glfwWindowHint(GLFW_MAXIMIZED, GLFW_TRUE);
        glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);

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

        GLFWVidMode mode = glfwGetVideoMode(primary_monitor);
        if (mode != null)
        {
            this.width = mode.width();
            this.height = mode.height();
        }

        glfwWindow = glfwCreateWindow(this.width, this.height, this.title, NULL, NULL);
        if (glfwWindow == NULL)
        {
            throw new IllegalStateException("could not create window");
        }

        glfwSetWindowSizeCallback(glfwWindow, (win, newWidth, newHeight)->
        {
            get().width = newWidth;
            get().height = newHeight;
            currentGameMode.resizeSpatialMap(newWidth, newHeight);
        });

        glfwMakeContextCurrent(glfwWindow);
        glfwSwapInterval(1); // v-sync

        // note: this must be called or nothing will work
        GL.createCapabilities();

        glEnable(GL_BLEND);
        glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);

        glViewport(0,0,this.width, this.height);

        System.out.println("\n-------- OPEN GL DEVICE -----------");
        System.out.println(glGetString(GL_VENDOR));
        System.out.println(glGetString(GL_RENDERER));
        System.out.println(glGetString(GL_VERSION));
        System.out.println("-----------------------------------\n");
    }

    private void initInput(KBMInput inputSystem)
    {
        glfwSetCursorPosCallback(glfwWindow, inputSystem::mousePosCallback);
        glfwSetMouseButtonCallback(glfwWindow, inputSystem::mouseButtonCallback);
        glfwSetScrollCallback(glfwWindow, inputSystem::mouseScrollCallback);
        glfwSetKeyCallback(glfwWindow, inputSystem::keyCallback);
    }

    /**
     * A simple utility system that just blanks the screen, getting it ready to render. This is used
     * instead of just calling it at the top of the loop, so that the screen clear can happen s late
     * as possible, just before rendering. todo: it may be better to use a framebuffer of some kind
     */
    private class BlankSystem extends GameSystem
    {

        public BlankSystem(ECS ecs) {
            super(ecs);
        }

        @Override
        public void run(float dt) {
            windowUpkeep();
        }

        @Override
        public void shutdown() {

        }
    }

    public void initGameMode()
    {
        var blanking_system = new BlankSystem(null);
        currentGameMode = new TestGame(ecs, blanking_system);
        currentGameMode.load();
        currentGameMode.start();

        camera.projectionSize.x = this.width;
        camera.projectionSize.y = this.height;

        currentGameMode.resizeSpatialMap(this.width, this.height);

        // order of system registry is important, systems run in the order they are added
        var inputSystem = new KBMInput(ecs);
        ecs.registerSystem(inputSystem);

        initInput(inputSystem);
    }

    public Camera camera()
    {
        return camera;
    }

    // this is the main game loop
    public void loop()
    {
        float lastTime = (float) glfwGetTime();
        float currentTime;
        float dt = -1.0f;

        while (!glfwWindowShouldClose(glfwWindow))
        {
            glfwPollEvents();

            if (dt >= 0)
            {
                ecs.run(dt);
                currentGameMode.update(dt);
            }

            glfwSwapBuffers(glfwWindow);

            currentTime = (float) glfwGetTime();
            dt = currentTime - lastTime;
            lastTime = currentTime;
        }
    }

    public int getWidth() {
        return width;
    }

    public int getHeight() {
        return height;
    }
}
