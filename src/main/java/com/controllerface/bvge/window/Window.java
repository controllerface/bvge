package com.controllerface.bvge.window;

import com.controllerface.bvge.ecs.ECS;
import com.controllerface.bvge.rendering.*;
import com.controllerface.bvge.scene.GenericScene;
import com.controllerface.bvge.scene.Scene;
import com.controllerface.bvge.util.AssetPool;
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

    //private ImGuiLayer imGuiLayer;

//    private FrameBuffer frameBuffer;

    private PickingTexture pickingTexture;

    public float r, g, b, a;

    private static Scene currentScene;

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

    public static Scene getScene()
    {
        return get().currentScene;
    }

    public void run()
    {
        System.out.println("LWJGL started: " + Version.getVersion());
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

    public void init()
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


//        this.frameBuffer = new FrameBuffer(1920, 1080);
        //this.pickingTexture = new PickingTexture(1920, 1080);
        glViewport(0,0,1920, 1080);


//        this.imGuiLayer = new ImGuiLayer(glfwWindow, pickingTexture);
//        this.imGuiLayer.initImGui();


        glfwSetCursorPosCallback(glfwWindow, MouseListener::mousePosCallback);
        glfwSetMouseButtonCallback(glfwWindow, MouseListener::mouseButtonCallback);
        glfwSetScrollCallback(glfwWindow, MouseListener::mouseScrollCallback);

        var is = new InputSystem(ecs);

        glfwSetKeyCallback(glfwWindow, is::keyCallback);
        //glfwSetKeyCallback(glfwWindow, KeyListener::keyCallback);

        ecs.registerSystem(is);
        ecs.registerSystem(new RenderSystem(ecs));

        currentScene = new GenericScene(ecs);
        currentScene.load();
        currentScene.init();
        currentScene.start();
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

//    public static FrameBuffer getFrameBuffer()
//    {
//        return get().frameBuffer;
//    }

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
                currentScene.update(dt);
            }

            glfwSwapBuffers(glfwWindow);
            MouseListener.endFrame();

            currentTime = (float) glfwGetTime();
            dt = currentTime - lastTime;
            lastTime = currentTime;
        }
    }
}
