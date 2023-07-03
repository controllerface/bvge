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
    }

    public void keyCallback(long window, int key, int scancode, int action, int mods)
    {
        if (key < 0)
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
}
