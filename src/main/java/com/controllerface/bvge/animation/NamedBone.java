package com.controllerface.bvge.animation;

import java.util.Arrays;

public enum NamedBone
{
    PELVIS  (1),
    TORSO   (2),
    NECK    (2),
    HEAD    (2),
    BICEP   (2),
    FOREARM (2),
    HAND    (2),
    THIGH   (1),
    SHIN    (1),
    FOOT    (1),

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
