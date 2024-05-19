package com.controllerface.bvge.ecs.components;

import org.joml.Vector2f;

import java.util.EnumMap;
import java.util.Map;

import static org.lwjgl.glfw.GLFW.*;

public class ControlPoints implements GameComponent
{
    private boolean disabled = false;

    private static final Map<InputBinding, Integer> input_bindings = new EnumMap<>(InputBinding.class);
    private static final Map<InputBinding, Boolean> input_states = new EnumMap<>(InputBinding.class);

    // todo: make this non-static and configurable
    static
    {
        for (var binding : InputBinding.values())
        {
            int input = switch (binding)
            {
                case MOVE_UP         -> GLFW_KEY_W;
                case MOVE_DOWN       -> GLFW_KEY_S;
                case MOVE_LEFT       -> GLFW_KEY_A;
                case MOVE_RIGHT      -> GLFW_KEY_D;
                case JUMP            -> GLFW_KEY_SPACE;
                case MOUSE_PRIMARY   -> GLFW_MOUSE_BUTTON_1;
                case MOUSE_SECONDARY -> GLFW_MOUSE_BUTTON_2;
                case MOUSE_MIDDLE    -> GLFW_MOUSE_BUTTON_3;
                case MOUSE_BACK      -> GLFW_MOUSE_BUTTON_4;
                case MOUSE_FORWARD   -> GLFW_MOUSE_BUTTON_5;
            };
            input_bindings.put(binding, input);
        }
    }

    // target position
    private final Vector2f screen_target = new Vector2f();
    private final Vector2f world_target = new Vector2f();

    public void update_input_states(boolean[] key_down, boolean[] mouse_down)
    {
        for (var binding : InputBinding.values())
        {
            var input = input_bindings.get(binding);
            boolean state = binding.mouse
                ? mouse_down[input]
                : key_down[input];
            input_states.put(binding, state);
        }
    }

    public Map<InputBinding, Boolean> input_states()
    {
        return input_states;
    }

    public boolean is_disabled()
    {
        return disabled;
    }

    public Vector2f get_screen_target()
    {
        return screen_target;
    }

    public Vector2f get_world_target()
    {
        return world_target;
    }
}
