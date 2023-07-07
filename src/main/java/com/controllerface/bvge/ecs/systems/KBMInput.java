package com.controllerface.bvge.ecs.systems;

import com.controllerface.bvge.ecs.components.Component;
import com.controllerface.bvge.ecs.components.ControlPoints;
import com.controllerface.bvge.ecs.ECS;
import com.controllerface.bvge.window.Window;

import static org.lwjgl.glfw.GLFW.*;
import static org.lwjgl.glfw.GLFW.GLFW_RELEASE;

public class KBMInput extends GameSystem
{
    private final boolean[] keyDown = new boolean[350];
    double scrollX, scrollY;
    public KBMInput(ECS ecs)
    {
        super(ecs);
    }

    @Override
    public void run(float dt)
    {
        var controllables = ecs.getComponents(Component.ControlPoints);
        controllables.forEach((entity, component) ->
        {
            ControlPoints controlPoints = Component.ControlPoints.coerce(component);
            assert controlPoints != null : "Component was null";
            if (controlPoints.isDisabled()) return;

            controlPoints.setUp(keyDown[GLFW_KEY_W]);
            controlPoints.setLeft(keyDown[GLFW_KEY_A]);
            controlPoints.setDown(keyDown[GLFW_KEY_S]);
            controlPoints.setRight(keyDown[GLFW_KEY_D]);
        });

        // test camera moving code
        if (keyDown[GLFW_KEY_LEFT])
        {
            Window.get().camera().position.x -= 5;
        }

        if (keyDown[GLFW_KEY_RIGHT])
        {
            Window.get().camera().position.x += 5;
        }

        if (keyDown[GLFW_KEY_UP])
        {
            Window.get().camera().position.y += 5;
        }

        if (keyDown[GLFW_KEY_DOWN])
        {
            Window.get().camera().position.y -= 5;
        }

        if (keyDown[GLFW_KEY_COMMA])
        {
            Window.get().camera().addZoom(.5f);
        }

        if (keyDown[GLFW_KEY_PERIOD])
        {
            Window.get().camera().addZoom(-.5f);
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
            keyDown[key] = true;
        }
        else if (action == GLFW_RELEASE)
        {
            keyDown[key] = false;
        }
    }

    private int mouseButtonsDown = 0;
    private boolean[] mouseButtonsPressed =  new boolean[9];
    private boolean isDragging;
    private double xPos;
    private double yPos;
    private double lastX;
    private double lastY;


    public void mouseScrollCallback(long window, double xOffset, double yOffset)
    {
        scrollX = xOffset;
        scrollY = yOffset;
        Window.get().camera().addZoom(-(float) yOffset / 2);
    }

    public void mouseButtonCallback(long window, int button, int action, int mods)
    {
        System.out.println(button);
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
