package com.controllerface.bvge.window;

import static org.lwjgl.glfw.GLFW.GLFW_PRESS;
import static org.lwjgl.glfw.GLFW.GLFW_RELEASE;

public class KeyListener
{
    private static KeyListener INSTANCE;
    private boolean keypressed[] = new boolean[350];

    private KeyListener()
    {

    }

    public static KeyListener get()
    {
        if (KeyListener.INSTANCE == null)
        {
            KeyListener.INSTANCE = new KeyListener();
        }
        return KeyListener.INSTANCE;
    }

    public static void keyCallback(long window, int key, int scancode, int action, int mods)
    {
        if (action == GLFW_PRESS)
        {
            get().keypressed[key] = true;
        }
        else if (action == GLFW_RELEASE)
        {
            get().keypressed[key] = false;
        }
    }

    public static boolean isKeyPressed(int keycode)
    {
        return get().keypressed[keycode];
    }
}
