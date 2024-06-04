package com.controllerface.bvge.geometry;

public record UnloadedEntity(float x, float y, float z, float w,
                             float anim_elapsed_x, float anim_elapsed_y,
                             short motion_x, short motion_y,
                             float anim_index_x, float anim_index_y,
                             int model_id, int model_transform_id,
                             float mass, int root_hull, int flags,
                             UnloadedHull[] hulls,
                             UnloadedEntityBone[] bones) { }
