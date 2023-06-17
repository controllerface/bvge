package com.controllerface.bvge.window;

import com.controllerface.bvge.ecs.ECS;
import com.controllerface.bvge.ecs.systems.KBMInput;
import com.controllerface.bvge.ecs.systems.LineRendering;
import com.controllerface.bvge.ecs.systems.VerletPhysics;
import com.controllerface.bvge.ecs.systems.SpriteRendering;
import com.controllerface.bvge.scene.GameRunning;
import com.controllerface.bvge.scene.GameMode;
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

    private ECS ecs = new ECS();

    private Window()
    {
        this.width = 1920;
        this.height = 1080;
        this.title = "BVGE Test";
        this.r = 1;
        this.g = 1;
        this.b = 1;
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

    public static GameMode getScene()
    {
        return get().currentGameMode;
    }

    public void run()
    {
        System.out.println("LWJGL version: " + Version.getVersion());
        init();
        loop();

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
    }

    private void initWindow()
    {
        GLFWErrorCallback.createPrint(System.err).set();

        if (!glfwInit())
        {
            throw new IllegalStateException("could not init GLFW");
        }

        glfwDefaultWindowHints();
        glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
        glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
        glfwWindowHint(GLFW_MAXIMIZED, GLFW_TRUE);

        glfwWindow = glfwCreateWindow(this.width, this.height, this.title, NULL, NULL);
        if (glfwWindow == NULL)
        {
            throw new IllegalStateException("could not create window");
        }

        glfwSetWindowSizeCallback(glfwWindow, (win, newWidth, newHeight)->
        {
            Window.setWidth(newWidth);
            Window.setHeight(newHeight);
        });

        glfwMakeContextCurrent(glfwWindow);
        glfwSwapInterval(1); // v-sync
        glfwShowWindow(glfwWindow);

        // note: this must be called or nothing will work
        GL.createCapabilities();

        glEnable(GL_BLEND);
        glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
        glViewport(0,0,1920, 1080);
    }

    private void initInput(KBMInput inputSystem)
    {
        glfwSetCursorPosCallback(glfwWindow, MouseListener::mousePosCallback);
        glfwSetMouseButtonCallback(glfwWindow, MouseListener::mouseButtonCallback);
        glfwSetScrollCallback(glfwWindow, MouseListener::mouseScrollCallback);
        glfwSetKeyCallback(glfwWindow, inputSystem::keyCallback);
    }

    public void init()
    {
        initWindow();

        // order of system registry is important, systems run in the order they are added
        var inputSystem = new KBMInput(ecs);
        ecs.registerSystem(inputSystem);
        ecs.registerSystem(new VerletPhysics(ecs));
        ecs.registerSystem(new SpriteRendering(ecs));
        ecs.registerSystem(new LineRendering(ecs));

        initInput(inputSystem);

        currentGameMode = new GameRunning(ecs);
        currentGameMode.load();
        currentGameMode.start();
    }

    public static int getWidth() {
        return get().width;
    }

    public static int getHeight() {
        return get().height;
    }

    public static void setWidth(int newWidth) {
        get().width = newWidth;
    }

    public static void setHeight(int newHeight) {
        get().height = newHeight;
    }

    public static float getTargetAspectRatio()
    {
        return 16.0f / 9.0f;
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

            glfwSwapBuffers(glfwWindow);
            MouseListener.endFrame();

            currentTime = (float) glfwGetTime();
            dt = currentTime - lastTime;
            lastTime = currentTime;
        }
    }
}
