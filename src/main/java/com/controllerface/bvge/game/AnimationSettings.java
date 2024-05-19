package com.controllerface.bvge.game;

import java.util.EnumMap;
import java.util.Map;

public class AnimationSettings
{
    static final Map<AnimationState, Map<AnimationState, Float>> transitions = define_transitions();

    private static String lookup_table = "";

    private static float get_transition(AnimationState from, AnimationState to)
    {
        return switch (from)
        {
            case IDLE -> switch (to)
            {
                case IDLE, UNKNOWN -> 0.0f;
                case WALKING,
                     LAND_HARD,
                     FALLING_SLOW,
                     LAND_SOFT,
                     SWIM_UP,
                     SWIM_DOWN,
                     FALLING_FAST,
                     JUMPING,
                     RUNNING,
                     IN_AIR -> 0.4f;
                case PUNCH, RECOIL -> 0.1f;
            };

            case WALKING -> switch (to)
            {
                case WALKING, UNKNOWN -> 0.0f;
                case IDLE,
                     LAND_HARD,
                     FALLING_SLOW,
                     LAND_SOFT,
                     SWIM_UP,
                     SWIM_DOWN,
                     FALLING_FAST,
                     JUMPING,
                     RUNNING,
                     IN_AIR -> 0.2f;
                case PUNCH, RECOIL -> 0.1f;
            };

            case FALLING_FAST -> switch (to)
            {
                case FALLING_FAST, UNKNOWN -> 0.0f;
                case WALKING,
                     LAND_HARD,
                     FALLING_SLOW,
                     LAND_SOFT,
                     SWIM_UP,
                     SWIM_DOWN,
                     IDLE,
                     JUMPING,
                     RUNNING,
                     IN_AIR -> 0.2f;
                case PUNCH, RECOIL -> 0.1f;
            };

            case IN_AIR -> switch (to)
            {
                case IN_AIR, UNKNOWN -> 0.0f;
                case WALKING,
                     FALLING_SLOW,
                     FALLING_FAST,
                     LAND_SOFT,
                     SWIM_UP,
                     SWIM_DOWN,
                     IDLE,
                     JUMPING,
                     RUNNING,
                     LAND_HARD -> 0.2f;
                case PUNCH, RECOIL -> 0.1f;
            };

            case FALLING_SLOW -> switch (to)
            {
                case FALLING_SLOW, UNKNOWN -> 0.0f;
                case WALKING,
                     LAND_HARD,
                     FALLING_FAST,
                     LAND_SOFT,
                     SWIM_UP,
                     SWIM_DOWN,
                     IDLE,
                     JUMPING,
                     RUNNING,
                     IN_AIR -> 0.2f;
                case PUNCH, RECOIL -> 0.1f;
            };

            case SWIM_UP -> switch (to)
            {
                case SWIM_UP, UNKNOWN -> 0.0f;
                case WALKING,
                     LAND_HARD,
                     FALLING_SLOW,
                     LAND_SOFT,
                     SWIM_DOWN,
                     FALLING_FAST,
                     IDLE,
                     JUMPING,
                     RUNNING,
                     IN_AIR -> 0.2f;
                case PUNCH, RECOIL -> 0.1f;
            };

            case SWIM_DOWN -> switch (to)
            {
                case SWIM_DOWN, UNKNOWN -> 0.0f;
                case WALKING,
                     LAND_HARD,
                     FALLING_SLOW,
                     LAND_SOFT,
                     SWIM_UP,
                     FALLING_FAST,
                     IDLE,
                     JUMPING,
                     RUNNING,
                     IN_AIR -> 0.2f;
                case PUNCH, RECOIL -> 0.1f;
            };

            case PUNCH -> switch (to)
            {
                case PUNCH, UNKNOWN -> 0.0F;
                case IDLE, WALKING, JUMPING -> 0.3F;
                case RUNNING, RECOIL, IN_AIR, LAND_HARD, LAND_SOFT, SWIM_UP, SWIM_DOWN -> 0.2F;
                case FALLING_FAST, FALLING_SLOW -> 0.5F;
            };

            case RUNNING -> switch (to)
            {
                case RUNNING, UNKNOWN -> 0.0f;
                case LAND_HARD,
                     FALLING_SLOW,
                     LAND_SOFT,
                     SWIM_UP,
                     SWIM_DOWN,
                     FALLING_FAST,
                     JUMPING,
                     IN_AIR -> 0.2f;
                case IDLE,
                     WALKING -> 0.4f;
                case PUNCH, RECOIL -> 0.1f;
            };

            case JUMPING -> 0.5f;
            case RECOIL, LAND_HARD, LAND_SOFT -> 0.05f;
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
                n.put(next_state, get_transition(base_state, next_state));
            }
            m.put(base_state, n);
        }
        return m;
    }

    /**
     * Generates an Open CL C lookup table for the transition times of all defined animations. The generated table will look
     * similar to this example at runtime:
     *
     * constant float transition_table[14][14] =
     * {
     *   {0.2f, ... , 0.05f},               // transitions from anim 0, to 0 through n
     *    // ... other timings go here ...
     *   {0.1f, ... , 0.4f},                // transitions from anim n, to 0 through n
     * };
     *
     * Each ordinal of the animation state enum is mapped such that the animation being transitioned from is mapped by ordinal
     * to the first dimension of the array. Each animation being transitioned into is mapped by ordinal to the second dimension
     * of the array. This means the array grows in size exponentially with the number of animation states, as all states must
     * have some defined value (even if it is 0.0f) for a transition time from that state to all others.
     * @return a String containing the generated lookup table.
     */
    public static String cl_lookup_table()
    {
        if (lookup_table.isEmpty())
        {
            var values = AnimationState.values();
            int length = values.length;
            var buffer = new StringBuilder();

            buffer.append(String.format("constant float transition_table[%d][%d] = \n{\n", length, length));
            for (var base_state : values)
            {
                buffer.append("\t{");
                for (var next_state : values)
                {
                    float value = transitions.get(base_state).get(next_state);
                    buffer.append(value).append("f");
                    if (next_state != AnimationState.UNKNOWN)
                    {
                        buffer.append(", ");
                    }
                }
                buffer.append("}");
                if (base_state != AnimationState.UNKNOWN)
                {
                    buffer.append(",");
                }
                buffer.append("\n");
            }
            buffer.append("};\n\n");
            lookup_table = buffer.toString();
        }
        return lookup_table;
    }
}
