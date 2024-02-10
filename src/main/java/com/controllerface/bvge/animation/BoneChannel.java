package com.controllerface.bvge.animation;

public record BoneChannel(int anim_timing_id,
                          int pos_start,
                          int pos_end,
                          int rot_start,
                          int rot_end,
                          int scl_start,
                          int scl_end) { }
