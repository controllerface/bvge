package com.controllerface.bvge.game;

import com.controllerface.bvge.core.Window;
import com.controllerface.bvge.ecs.ECS;
import com.controllerface.bvge.ecs.GameSystem;
import com.controllerface.bvge.ecs.components.ComponentType;
import com.controllerface.bvge.events.Event;

import static org.lwjgl.glfw.GLFW.*;

public class InputSystem extends GameSystem
{
    private final boolean[] key_down = new boolean[350];
    private final boolean[] mouse_down = new boolean[9];
    private int mouse_count = 0;
    private boolean isDragging;
    private double xPos;
    private double yPos;
    private double lastX;
    private double lastY;

    private boolean shift;
    private boolean control;

    double scrollX, scrollY;

    public InputSystem(ECS ecs)
    {
        super(ecs);
    }

    @Override
    public void tick(float dt)
    {
        PlayerInput playerInput = ComponentType.InputState.forEntity(ecs, Constants.PLAYER_ID);
        assert playerInput != null : "Component was null";
        playerInput.update_inputs(key_down, mouse_down);
        playerInput.get_screen_target().set(xPos, yPos);
        shift = key_down[GLFW_KEY_LEFT_SHIFT];
        control = key_down[GLFW_KEY_LEFT_CONTROL];
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
        boolean down = yOffset < 0;
        float amount = down
            ? 0.1f
            : -0.1f;

        if (shift)
        {
            var type = down
                ? Event.Type.NEXT_ITEM
                : Event.Type.PREV_ITEM;

            Window.get().event_bus().emit_event(Event.input(type));
        }
        else if (control)
        {
            Window.get().camera().add_zoom(amount);
        }
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
