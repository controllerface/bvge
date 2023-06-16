package com.controllerface.bvge.window;

import com.controllerface.bvge.ecs.Component;
import com.controllerface.bvge.ecs.ControlPoints;
import com.controllerface.bvge.ecs.ECS;
import com.controllerface.bvge.ecs.SystemEX;

import static org.lwjgl.glfw.GLFW.*;
import static org.lwjgl.glfw.GLFW.GLFW_RELEASE;

public class KBMInput extends SystemEX
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
    }

    public void keyCallback(long window, int key, int scancode, int action, int mods)
    {
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
