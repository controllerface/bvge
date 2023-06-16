package com.controllerface.bvge.window;

import com.controllerface.bvge.TransformEX;
import com.controllerface.bvge.ecs.Component;
import com.controllerface.bvge.ecs.ECS;
import com.controllerface.bvge.ecs.PlayerControlled;
import com.controllerface.bvge.ecs.SystemEX;

import static org.lwjgl.glfw.GLFW.*;
import static org.lwjgl.glfw.GLFW.GLFW_RELEASE;

public class InputSystem extends SystemEX
{
    private boolean keypressed[] = new boolean[350];

    public InputSystem(ECS ecs)
    {
        super(ecs);
    }

    @Override
    public void run(float dt)
    {
        // get all controllable components
        var contr = ecs.getComponents(Component.PlayerControlled);
        contr.forEach((entity, component) ->
        {
            var t = ecs.getComponentFor(entity, Component.Transform);
            TransformEX te = Component.Transform.coerce(t);
            if (keypressed[GLFW_KEY_LEFT])
            {
                te.position.x-= 2;
            }
            if (keypressed[GLFW_KEY_RIGHT])
            {
                te.position.x+= 2;
            }
            if (keypressed[GLFW_KEY_UP])
            {
                te.position.y+= 2;
            }
            if (keypressed[GLFW_KEY_DOWN])
            {
                te.position.y-= 2;
            }
        });
        //
    }


    public void keyCallback(long window, int key, int scancode, int action, int mods)
    {
        if (action == GLFW_PRESS)
        {
            keypressed[key] = true;
        }
        else if (action == GLFW_RELEASE)
        {
            keypressed[key] = false;
        }
    }
}
