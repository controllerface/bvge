package com.controllerface.bvge.ecs.systems;

import com.controllerface.bvge.ecs.ECS;
import com.controllerface.bvge.ecs.components.Component;
import com.controllerface.bvge.ecs.components.ControlPoints;
import com.controllerface.bvge.ecs.components.GameComponent;
import com.controllerface.bvge.window.Window;

import java.util.Map;

import static org.lwjgl.glfw.GLFW.*;

public class KBMInput extends GameSystem
{
    private final boolean[] key_down = new boolean[350];
    private final boolean[] mouse_down = new boolean[9];
    private int mouse_count = 0;
    private boolean isDragging;
    private double xPos;
    private double yPos;
    private double lastX;
    private double lastY;

    double scrollX, scrollY;

    public KBMInput(ECS ecs)
    {
        super(ecs);
    }

    @Override
    public void tick(float dt)
    {
        var controllables = ecs.get_components(Component.ControlPoints);
        for (Map.Entry<String, GameComponent> entry : controllables.entrySet())
        {
            GameComponent component = entry.getValue();
            ControlPoints controlPoints = Component.ControlPoints.coerce(component);
            assert controlPoints != null : "Component was null";
            if (controlPoints.is_disabled()) continue;
            controlPoints.update_input_states(key_down, mouse_down);
            controlPoints.get_screen_target().set(xPos, yPos);
        }

        // test camera moving code
        if (key_down[GLFW_KEY_LEFT])
        {
            Window.get().camera().position().x -= 5;
        }

        if (key_down[GLFW_KEY_RIGHT])
        {
            Window.get().camera().position().x += 5;
        }

        if (key_down[GLFW_KEY_UP])
        {
            Window.get().camera().position().y += 5;
        }

        if (key_down[GLFW_KEY_DOWN])
        {
            Window.get().camera().position().y -= 5;
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


    public void mouseScrollCallback(long window, double xOffset, double yOffset)
    {
        scrollX = xOffset;
        scrollY = yOffset;
        float x = yOffset < 0 ? 0.5f : -0.5f;
        Window.get().camera().add_zoom(x);
    }

    public void mouseButtonCallback(long window, int button, int action, int mods)
    {
        if (action == GLFW_PRESS)
        {
            mouse_count++;

            if (button < mouse_down.length)
            {
                mouse_down[button] = true;
            }
        }
        else if (action == GLFW_RELEASE)
        {
            mouse_count--;

            if (button < mouse_down.length)
            {
                mouse_down[button] = false;
                isDragging = false;
            }
        }
    }

    public void mousePosCallback(long window, double xpos, double ypos)
    {
        if (mouse_count > 0)
        {
            isDragging = true;
        }
        lastX = xPos;
        lastY = yPos;
        xPos = xpos;
        yPos = ypos;
    }
}
