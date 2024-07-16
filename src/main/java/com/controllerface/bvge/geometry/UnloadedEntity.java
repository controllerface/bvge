package com.controllerface.bvge.geometry;

public record UnloadedEntity(float x, float y, float z, float w,
                             float anim_time_x, float anim_time_y, float anim_time_z, float anim_time_w,
                             float anim_prev_x, float anim_prev_y, float anim_prev_z, float anim_prev_w,
                             short motion_x, short motion_y,
                             int anim_layer_x, int anim_layer_y, int anim_layer_z, int anim_layer_w,
                             int anim_previous_x, int anim_previous_y, int anim_previous_z, int anim_previous_w,
                             int model_id, int model_transform_id,
                             float mass, int root_hull, int type, int flags,
                             UnloadedHull[] hulls,
                             UnloadedEntityBone[] bones) { }
