package com.controllerface.bvge.ecs.systems.physics;

import com.controllerface.bvge.ecs.components.RigidBody2D;
import org.joml.Vector2f;

public record CollisionManifold(RigidBody2D vertexObject,
                                RigidBody2D edgeObject,
                                Vector2f normal,
                                float depth,
                                int edgeA,
                                int edgeB,
                                int vert)
{
}
