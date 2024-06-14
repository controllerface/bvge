package com.controllerface.bvge.game.state;

public record GameEntity(Hull[] hulls,
                         float[] position,
                         float scale,
                         float mass,
                         float friction,
                         float restitution,
                         int model_id,
                         int uv_offset,
                         int flags) { }
