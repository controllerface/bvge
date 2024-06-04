package com.controllerface.bvge.geometry;

public record UnloadedPoint(float x, float y, float z, float w,
                            int bone_1, int bone_2, int bone_3, int bone_4,
                            int vertex_reference, int hull_index,
                            short hit_count, int flags)
{
}
