package com.controllerface.bvge.window;

import com.controllerface.bvge.cl.OpenCL;
import com.controllerface.bvge.ecs.ECS;
import com.controllerface.bvge.ecs.systems.KBMInput;
import com.controllerface.bvge.game.GameMode;
import com.controllerface.bvge.game.TestGame;
import org.joml.Vector2f;
import org.lwjgl.Version;
import org.lwjgl.glfw.GLFWErrorCallback;
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
        this.r = 0.5f;
        this.g = 0.5f;
        this.b = 0.5f;
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
        //OpenCL.init();
        init();
        OpenCL.init();
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
        glfwPollEvents();
        glClearColor(r, g, b, a);
        glClear(GL_COLOR_BUFFER_BIT);
        camera.adjustProjection();
    }

    private void initWindow()
    {
        GLFWErrorCallback.createPrint(System.err).set();

        if (!glfwInit())
        {
            throw new IllegalStateException("could not init GLFW");
        }

        glfwDefaultWindowHints();
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
        glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
        glfwWindowHint(GLFW_MAXIMIZED, GLFW_TRUE);
        glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);

        var prim = glfwGetPrimaryMonitor();

        int[] x = new int[1];
        int[] y = new int[1];
        int[] w = new int[1];
        int[] h = new int[1];
        glfwGetMonitorWorkarea(prim, x,y,w,h);
        this.width = w[0];
        this.height = h[0];

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
        glfwSwapInterval(0); // v-sync

        // note: this must be called or nothing will work
        GL.createCapabilities();

        glEnable(GL_BLEND);
        glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);

        glViewport(0,0,this.width, this.height);
    }

    private void initInput(KBMInput inputSystem)
    {
        glfwSetCursorPosCallback(glfwWindow, inputSystem::mousePosCallback);
        glfwSetMouseButtonCallback(glfwWindow, inputSystem::mouseButtonCallback);
        glfwSetScrollCallback(glfwWindow, inputSystem::mouseScrollCallback);
        glfwSetKeyCallback(glfwWindow, inputSystem::keyCallback);
    }

    public void init()
    {
        initWindow();

        currentGameMode = new TestGame(ecs);
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
            windowUpkeep();

            if (dt >= 0)
            {
                ecs.run(dt);
                currentGameMode.update(dt);
            }
            //System.out.println("FPS:" + (1000 / dt) / 1000);

            glfwSwapBuffers(glfwWindow);
            MouseListener.endFrame();

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
