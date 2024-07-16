package com.controllerface.bvge.animation;

import java.util.Arrays;

public enum NamedBone
{
    PELVIS  (0),
    TORSO   (1),
    NECK    (1),
    HEAD    (1),
    BICEP   (1),
    FOREARM (1),
    HAND    (1),
    THIGH   (0),
    SHIN    (0),
    FOOT    (0),

    UNKNOWN(0),

    ;

    public final int layer;

    NamedBone(int layer)
    {
        this.layer = layer;
    }

    public static NamedBone fuzzy_match(String bone_name)
    {
        if (bone_name == null) return UNKNOWN;

        return Arrays.stream(values())
            .filter(bone -> bone_name.toUpperCase().contains(bone.name()))
            .findAny().orElse(UNKNOWN);
    }
}
