package com.controllerface.bvge.ecs.systems;

import com.controllerface.bvge.Transform;
import com.controllerface.bvge.ecs.*;
import com.controllerface.bvge.util.MathEX;
import org.joml.Vector2f;

import java.util.HashMap;
import java.util.Map;

public class VerletPhysics extends GameSystem
{
    private final float TICK_RATE = 1.0f / 60.0f;
    private final int SUB_STEPS = 8;
    private final int EDGE_STEPS = 8;
    private final float GRAVITY = 9.8f;
    private final float FRICTION = 0.980f;
    private float accumulator = 0.0f;

    private final Vector2f accelBuffer = new Vector2f();
    private final Vector2f diffBuffer = new Vector2f();
    private final Vector2f moveBuffer = new Vector2f();


    public VerletPhysics(ECS ecs)
    {
        super(ecs);
    }

    private void resolveForces(String entity, RigidBody2D body2D)
    {
        var cp = ecs.getComponentFor(entity, Component.ControlPoints);

        if (cp == null) return;

        ControlPoints controlPoints = Component.ControlPoints.coerce(cp);
        accelBuffer.zero();

        if (controlPoints.isLeft())
        {
            accelBuffer.x -= body2D.getForce();
        }
        if (controlPoints.isRight())
        {
            accelBuffer.x += body2D.getForce();
        }
        if (controlPoints.isUp())
        {
            accelBuffer.y += body2D.getForce();
        }
        if (controlPoints.isDown())
        {
            accelBuffer.y -= body2D.getForce();
        }
        body2D.getAcc().x = accelBuffer.x;
        body2D.getAcc().y = accelBuffer.y;
    }

    private void integrate(String entitiy, RigidBody2D body2D, float dt)
    {
        var t = ecs.getComponentFor(entitiy, Component.Transform);
        Transform transform = Component.Transform.coerce(t);
        var displacement = body2D.getAcc().mul(dt * dt);
        for (Point2D point : body2D.getVerts())
        {
            diffBuffer.zero();
            moveBuffer.zero();
            point.pos().sub(point.prv(), diffBuffer);
            diffBuffer.mul(FRICTION);
            displacement.add(diffBuffer, moveBuffer);
            point.prv().set(point.pos());
            point.pos().add(moveBuffer);
        }
        MathEX.centroid(body2D.getVerts(), transform.position);
    }

    private final Map<String, RigidBody2D> bodyBuffer = new HashMap<>();

    @Override
    public void run(float dt)
    {
        // 0: get objects to calculate positions for
//        var kdtree = new KDTree<String>(2);
        var bodies = ecs.getComponents(Component.RigidBody2D);
        if (bodies == null || bodies.isEmpty()) return;

        // Pass 1: Resolve forces and integrate
        for (Map.Entry<String, GameComponent> entry : bodies.entrySet())
        {
            String entity = entry.getKey();
            GameComponent component = entry.getValue();
            RigidBody2D body2D = Component.RigidBody2D.coerce(component);
            resolveForces(entity, body2D); // 1: resolve forces
            integrate(entity, body2D, dt); // 2: integrate
            bodyBuffer.put(entity, body2D);
        }

        // Pass 2: Detect collisions
        for (Map.Entry<String, RigidBody2D> entry : bodyBuffer.entrySet())
        {
            String entity = entry.getKey();
            RigidBody2D body = entry.getValue();

        }

        // 3: find collisions
        // 4: resolve collisions
        // 5: resolve constraints
    }
}
