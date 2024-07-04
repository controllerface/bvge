package com.controllerface.bvge.ecs.systems;

import com.controllerface.bvge.ecs.ECS;
import com.controllerface.bvge.ecs.components.Component;
import com.controllerface.bvge.ecs.components.ControlPoints;
import com.controllerface.bvge.ecs.components.GameComponent;
import com.controllerface.bvge.window.events.Event;
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

    private boolean shift;
    private boolean control;

    double scrollX, scrollY;

    public KBMInput(ECS ecs)
    {
        super(ecs);
    }

    @Override
    public void tick(float dt)
    {
        var controllables = ecs.get_components(Component.ControlPoints);
        // todo: remove loop, get player data directly
        for (Map.Entry<String, GameComponent> entry : controllables.entrySet())
        {
            GameComponent component = entry.getValue();
            ControlPoints controlPoints = Component.ControlPoints.coerce(component);
            assert controlPoints != null : "Component was null";
            if (controlPoints.is_disabled()) continue;
            controlPoints.update_input_states(key_down, mouse_down);
            controlPoints.get_screen_target().set(xPos, yPos);
        }

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
            ? 0.5f
            : -0.5f;

        if (shift && control)
        {

            Window.get().camera().add_zoom(amount);
        }
        else if (control)
        {
            var type = down
                ? Event.Type.NEXT_ITEM
                : Event.Type.PREV_ITEM;

            Window.get().event_bus().report_event(Event.input(type));
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
