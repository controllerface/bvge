package com.controllerface.bvge.ecs.systems;

import com.controllerface.bvge.ecs.ECS;
import com.controllerface.bvge.ecs.components.Component;
import com.controllerface.bvge.ecs.components.ControlPoints;
import com.controllerface.bvge.ecs.components.GameComponent;
import com.controllerface.bvge.window.Window;

import java.util.Arrays;
import java.util.Map;

import static org.lwjgl.glfw.GLFW.*;

public class KBMInput extends GameSystem
{
    private final boolean[] key_down = new boolean[350];

    double scrollX, scrollY;
    public KBMInput(ECS ecs)
    {
        super(ecs);
    }

    @Override
    public void tick(float dt)
    {
        var controllables = ecs.getComponents(Component.ControlPoints);
        for (Map.Entry<String, GameComponent> entry : controllables.entrySet())
        {
            GameComponent component = entry.getValue();
            ControlPoints controlPoints = Component.ControlPoints.coerce(component);
            assert controlPoints != null : "Component was null";
            if (controlPoints.is_disabled()) continue;

            controlPoints.set_moving_up(key_down[GLFW_KEY_W]);
            controlPoints.set_moving_Left(key_down[GLFW_KEY_A]);
            controlPoints.set_moving_down(key_down[GLFW_KEY_S]);
            controlPoints.set_moving_right(key_down[GLFW_KEY_D]);
            controlPoints.set_rot_right(key_down[GLFW_KEY_E]);
            controlPoints.set_rotating_Left(key_down[GLFW_KEY_Q]);
            controlPoints.set_space_bar(key_down[GLFW_KEY_SPACE]);
        }

        // test camera moving code
        if (key_down[GLFW_KEY_LEFT])
        {
            Window.get().camera().position.x -= 5;
        }

        if (key_down[GLFW_KEY_RIGHT])
        {
            Window.get().camera().position.x += 5;
        }

        if (key_down[GLFW_KEY_UP])
        {
            Window.get().camera().position.y += 5;
        }

        if (key_down[GLFW_KEY_DOWN])
        {
            Window.get().camera().position.y -= 5;
        }

        if (key_down[GLFW_KEY_COMMA])
        {
            Window.get().camera().add_zoom(.5f);
        }

        if (key_down[GLFW_KEY_PERIOD])
        {
            Window.get().camera().add_zoom(-.5f);
        }
    }

    public void keyCallback(long window, int key, int scancode, int action, int mods)
    {
        if (key == GLFW_KEY_UNKNOWN)
        {
            System.out.println("Ignoring invalid keycode: " + key + " for action: " + action);
            return;
        }

        if (action == GLFW_PRESS)
        {
            key_down[key] = true;
        }
        else if(action == GLFW_RELEASE)
        {
            key_down[key] = false;
        }
    }

    private int mouseButtonsDown = 0;
    private boolean[] mouseButtonsPressed = new boolean[9];
    private boolean isDragging;
    private double xPos;
    private double yPos;
    private double lastX;
    private double lastY;


    public void mouseScrollCallback(long window, double xOffset, double yOffset)
    {
        scrollX = xOffset;
        scrollY = yOffset;
        float x = yOffset < 0 ? 0.5f : -0.5f;
        Window.get().camera().add_zoom(x);
    }

    public void mouseButtonCallback(long window, int button, int action, int mods)
    {
        //System.out.println(button);
        if (action == GLFW_PRESS)
        {
            mouseButtonsDown++;

            if (button < mouseButtonsPressed.length)
            {
                mouseButtonsPressed[button] = true;
            }
        }
        else if (action == GLFW_RELEASE)
        {
            mouseButtonsDown--;

            if (button < mouseButtonsPressed.length)
            {
                mouseButtonsPressed[button] = false;
                isDragging = false;
            }
        }
    }

    public void mousePosCallback(long window, double xpos, double ypos)
    {
        if (mouseButtonsDown > 0)
        {
            isDragging = true;
        }
        lastX = xPos;
        lastY = yPos;
        xPos = xpos;
        yPos = ypos;
    }
}
