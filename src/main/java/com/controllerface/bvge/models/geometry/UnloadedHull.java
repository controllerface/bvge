package com.controllerface.bvge.models.geometry;

public record UnloadedHull(float x, float y, float z, float w,
                           float scale_x, float scale_y,
                           float rotation_x, float rotation_y,
                           float friction, float restitution,
                           int integrity, int mesh_id, int entity_id,
                           int uv_offset, int flags,
                           UnloadedPoint[] points,
                           UnloadedEdge[] edges,
                           UnloadedHullBone[] bones)
{
}
