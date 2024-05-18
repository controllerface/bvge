package com.controllerface.bvge.game;

import java.util.EnumMap;
import java.util.Map;

public class AnimationSettings
{
    static final Map<AnimationState, Map<AnimationState, Float>> transitions = define_transitions();

    private static float define_transition(AnimationState from, AnimationState to)
    {
        return switch (from)
        {
            case IDLE -> switch (to)
            {
                case IDLE, UNKNOWN -> 0.0f;
                case WALKING, LAND_HARD, FALLING_SLOW, LAND_SOFT, SWIM_UP, SWIM_DOWN, FALLING_FAST, JUMPING, RUNNING, IN_AIR -> 0.4f;
                case PUNCH, JUMP_START -> 0.1f;
            };

            case WALKING -> switch (to)
            {
                case WALKING, UNKNOWN -> 0.0f;
                case IDLE, LAND_HARD, FALLING_SLOW, LAND_SOFT, SWIM_UP, SWIM_DOWN, FALLING_FAST, JUMPING, RUNNING, IN_AIR -> 0.2f;
                case PUNCH, JUMP_START -> 0.1f;
            };

            case FALLING_FAST -> switch (to)
            {
                case FALLING_FAST, UNKNOWN -> 0.0f;
                case WALKING, LAND_HARD, FALLING_SLOW, LAND_SOFT, SWIM_UP, SWIM_DOWN, IDLE, JUMPING, RUNNING, IN_AIR -> 0.2f;
                case PUNCH, JUMP_START -> 0.1f;
            };

            case IN_AIR -> switch (to)
            {
                case IN_AIR, UNKNOWN -> 0.0f;
                case WALKING, FALLING_SLOW, FALLING_FAST, LAND_SOFT, SWIM_UP, SWIM_DOWN, IDLE, JUMPING, RUNNING, LAND_HARD -> 0.2f;
                case PUNCH, JUMP_START -> 0.1f;
            };

            case FALLING_SLOW -> switch (to)
            {
                case FALLING_SLOW, UNKNOWN -> 0.0f;
                case WALKING, LAND_HARD, FALLING_FAST, LAND_SOFT, SWIM_UP, SWIM_DOWN, IDLE, JUMPING, RUNNING, IN_AIR -> 0.2f;
                case PUNCH, JUMP_START -> 0.1f;
            };

            case SWIM_UP -> switch (to)
            {
                case SWIM_UP, UNKNOWN -> 0.0f;
                case WALKING, LAND_HARD, FALLING_SLOW, LAND_SOFT, SWIM_DOWN, FALLING_FAST, IDLE, JUMPING, RUNNING, IN_AIR -> 0.2f;
                case PUNCH, JUMP_START -> 0.1f;
            };

            case SWIM_DOWN -> switch (to)
            {
                case SWIM_DOWN, UNKNOWN -> 0.0f;
                case WALKING, LAND_HARD, FALLING_SLOW, LAND_SOFT, SWIM_UP, FALLING_FAST, IDLE, JUMPING, RUNNING, IN_AIR -> 0.2f;
                case PUNCH, JUMP_START -> 0.1f;
            };

            case PUNCH -> 0.1f;
            case JUMPING -> 0.5f;

            case RUNNING -> 0.0f; // todo: implement this

            case JUMP_START, LAND_HARD, LAND_SOFT -> 0.0f;

            case UNKNOWN -> 0.0f;
        };
    }
    
    private static Map<AnimationState, Map<AnimationState, Float>> define_transitions()
    {
        EnumMap<AnimationState, Map<AnimationState, Float>> m = new EnumMap<>(AnimationState.class);
        for (var base_state : AnimationState.values())
        {
            EnumMap<AnimationState, Float> n = new EnumMap<>(AnimationState.class);
            for (var next_state :  AnimationState.values())
            {
                n.put(next_state, define_transition(base_state, next_state));
            }
            m.put(base_state, n);
        }
        return m;
    }



    private static String lookup_table = "";

    public static String cl_lookup_table()
    {
        if (lookup_table.isEmpty())
        {
            var values = AnimationState.values();
            int length = values.length;
            var buffer = new StringBuilder();

            buffer.append("constant float transition_table[").append(length).append("][").append(length).append("] = \n{\n");
            for (var base_state : values)
            {
                buffer.append("\t{");
                for (var next_state : values)
                {
                    float value = transitions.get(base_state).get(next_state);
                    buffer.append(value).append("f,");
                }
                buffer.append("},\n");
            }
            buffer.append("};\n\n");
            lookup_table = buffer.toString();
        }
        return lookup_table;
    }


}
