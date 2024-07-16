package com.controllerface.bvge.geometry;

public record UnloadedEntity(float x, float y, float z, float w,
                             float anim_time_x, float anim_time_y,
                             short motion_x, short motion_y,
                             int anim_layer_x, int anim_layer_y,
                             int anim_previous_x, int anim_previous_y,
                             int model_id, int model_transform_id,
                             float mass, int root_hull, int type, int flags,
                             UnloadedHull[] hulls,
                             UnloadedEntityBone[] bones) { }
