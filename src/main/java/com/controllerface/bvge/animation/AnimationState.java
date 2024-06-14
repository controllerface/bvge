package com.controllerface.bvge.animation;

import java.util.Arrays;

public enum AnimationState
{
    IDLE,
    WALKING,
    RUNNING,
    FALLING_FAST,
    RECOIL,
    JUMPING,
    IN_AIR,
    LAND_HARD,
    FALLING_SLOW,
    LAND_SOFT,
    SWIM_UP,
    SWIM_DOWN,
    PUNCH,
    UNKNOWN,

    ;

    private static String lookup_table = "";

    public static AnimationState fuzzy_match(String animation_name)
    {
        return Arrays.stream(values())
            .filter(state -> animation_name.toUpperCase().contains(state.name()))
            .findAny().orElse(UNKNOWN);
    }

    public static String cl_constants()
    {
        if (lookup_table.isEmpty())
        {
            var buffer = new StringBuilder();
            for (var state : values())
            {
                buffer.append("#define ")
                    .append(state.name())
                    .append(" ")
                    .append(state.ordinal())
                    .append("\n");
            }
            buffer.append("\n");
            lookup_table = buffer.toString();
        }
        return lookup_table;
    }
}