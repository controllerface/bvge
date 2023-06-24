package com.controllerface.bvge.window;

import com.controllerface.bvge.scene.Camera;
import org.joml.Matrix4f;
import org.joml.Vector2f;
import org.joml.Vector4f;

import static org.lwjgl.glfw.GLFW.GLFW_PRESS;
import static org.lwjgl.glfw.GLFW.GLFW_RELEASE;

public class MouseListener
{
    private static MouseListener INSTANCE;
    private double scrollX, scrollY;
    private double xPos;
    private double yPos;
    private double lastX;
    private double lastY;
    private double worldX;
    private double worldY;
    private double lastWorldX;
    private double lastWorldY;
    private boolean[] mouseButtonsPressed =  new boolean[9];
    private boolean isDragging;

    private int mouseButtonsDown = 0;

    private Vector2f gameViewPortPos = new Vector2f();
    private Vector2f gameViewPortSize = new Vector2f();

    private MouseListener()
    {
        this.scrollX = 0.0f;
        this.scrollY = 0.0f;
        this.xPos = 0.0f;
        this.yPos = 0.0f;
        this.lastX = 0.0f;
        this.lastY = 0.0f;
    }

    public static MouseListener get()
    {
        if (MouseListener.INSTANCE == null)
        {
            MouseListener.INSTANCE = new MouseListener();
        }
        return MouseListener.INSTANCE;
    }

    public static void mousePosCallback(long window, double xpos, double ypos)
    {
        if (get().mouseButtonsDown > 0)
        {
            get().isDragging = true;
        }

        get().lastX = get().xPos;
        get().lastY = get().yPos;
        get().lastWorldX = get().worldX;
        get().lastWorldY = get().worldY;
        get().xPos = xpos;
        get().yPos = ypos;
        calcOrthoX();
        calcOrthoY();
    }

    public static void mouseButtonCallback(long window, int button, int action, int mods)
    {
        if (action == GLFW_PRESS)
        {
            get().mouseButtonsDown++;

            if (button < get().mouseButtonsPressed.length)
            {
                get().mouseButtonsPressed[button] = true;
            }
        }
        else if (action == GLFW_RELEASE)
        {
            get().mouseButtonsDown--;

            if (button < get().mouseButtonsPressed.length)
            {
                get().mouseButtonsPressed[button] = false;
                get().isDragging = false;
            }
        }
    }

    public static void mouseScrollCallback(long window, double xOffset, double yOffset)
    {
        get().scrollX = xOffset;
        get().scrollY = yOffset;
    }

    public static void endFrame()
    {
        get().scrollX = 0;
        get().scrollY = 0;
        get().lastX = 0;
        get().lastY = 0;
        get().lastWorldX = get().worldX;
        get().lastWorldY = get().worldY;
    }

    public static float getX()
    {
        return (float)get().xPos;
    }

    public static float getY()
    {
        return (float)get().yPos;
    }

    public static float getDx()
    {
        return (float)(get().lastX - get().xPos);
    }

    public static float getWorldDx()
    {
        return (float)(get().lastWorldX - get().worldX);
    }

    public static float getDy()
    {
        return (float)(get().lastY - get().yPos);
    }

    public static float getWorldDy()
    {
        return (float)(get().lastWorldY - get().worldY);
    }

    public static float getScrollX()
    {
        return (float)get().scrollX;
    }

    public static float getScrollY()
    {
        return (float)get().scrollY;
    }

    public static boolean isDragging()
    {
        return get().isDragging;
    }

    public static boolean mouseButton(int button)
    {
        if (button < get().mouseButtonsPressed.length)
        {
            return get().mouseButtonsPressed[button];
        }
        return false;
    }

    public static float getScreenX()
    {
        float currentX = getX() - get().gameViewPortPos.x;
        currentX = (currentX / get().gameViewPortSize.x) * 1920.0f;
        return currentX;
    }

    public static float getScreenY()
    {
        float currentY = getY() - get().gameViewPortPos.y; // we have p2 account for the texture flip here
        // flipping the Y here is needed because ImGui has Y co-ordinates flipped relative p2 OpenGL
        currentY = 1080.0f - ((currentY / get().gameViewPortSize.y) * 1080.0f);
        return currentY;
    }

    private static void calcOrthoX()
    {
        float currentX = getX() - get().gameViewPortPos.x;
        currentX = (currentX / get().gameViewPortSize.x) * 2.0f - 1.0f;
        Vector4f tmp = new Vector4f(currentX, 0, 0, 1);

        Camera camera = Window.getScene().camera();
        Matrix4f viewProjection = new Matrix4f();
        camera.getInverseView().mul(camera.getInverseProjection(), viewProjection);
        tmp.mul(viewProjection);

        get().worldX = tmp.x;
    }

    private static void calcOrthoY()
    {
        float currentY = getY() - get().gameViewPortPos.y;
        currentY = -((currentY / get().gameViewPortSize.y) * 2.0f - 1.0f);
        Vector4f tmp = new Vector4f(0, currentY, 0, 1);

        Camera camera = Window.getScene().camera();
        Matrix4f viewProjection = new Matrix4f();
        camera.getInverseView().mul(camera.getInverseProjection(), viewProjection);
        tmp.mul(viewProjection);

        get().worldY = tmp.y;
    }

    public static float getOrthoX()
    {
        return (float) get().worldX;
    }


    public static float getOrthoY()
    {
        return (float) get().worldY;
    }

    public static void setGameViewPortPos(Vector2f gameViewPortPos)
    {
        get().gameViewPortPos.set(gameViewPortPos);
    }

    public static void setGameViewPortSize(Vector2f gameViewPortSize)
    {
        get().gameViewPortSize.set(gameViewPortSize);
    }
}

