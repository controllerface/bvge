package com.controllerface.bvge.ecs.systems.physics;

import com.controllerface.bvge.data.FBody2D;
import org.joml.Vector2f;

public record CollisionManifold(FBody2D vertexObject,
                                FBody2D edgeObject,
                                Vector2f normal,
                                float depth,
                                int edgeA,
                                int edgeB,
                                int vert)
{
}
