package com.controllerface.bvge.ecs;

import com.controllerface.bvge.TransformEX;
import com.controllerface.bvge.util.JMath;
import org.joml.Vector2f;

public class PhysicsSystem extends SystemEX
{
    private final float TICK_RATE = 1.0f / 60.0f;
    private final int SUB_STEPS = 8;
    private final int EDGE_STEPS = 8;
    private final float GRAVITY = 9.8f;
    private final float FRICTION = 0.997f;
    private float accumulator = 0.0f;

    private final Vector2f accelBuffer = new Vector2f();
    private final Vector2f diffBuffer = new Vector2f();
    private final Vector2f moveBuffer = new Vector2f();


    public PhysicsSystem(ECS ecs)
    {
        super(ecs);
    }

    private void resolveForces(String entity, RigidBody2D body2D)
    {
        var cp = ecs.getComponentFor(entity, Component.ControlPoints);
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
        TransformEX transform = Component.Transform.coerce(t);
        //if (body.static) return;
        var displacement = body2D.getAcc().mul(dt * dt);

        for (Point2D point : body2D.getVerts())
        {
            diffBuffer.zero();
            moveBuffer.zero();
            point.getPos().sub(point.getPrv(), diffBuffer);
            displacement.add(diffBuffer, moveBuffer);
            point.getPrv().set(point.getPos());
            point.getPos().add(moveBuffer);
        }
        JMath.centroid(body2D.getVerts(), transform.position);

//        if (acc.x != 0.0f || acc.y != 0)
//        {
//            //var norm = acc.normalize();
//            transform.position.add(acc.mul(dt * dt));
//        }

        //unit = vectors.multiply(unit, dt);
        //var movement_vector = vectors.multiply(unit, dt ^2);
//        body.verts.forEach(vertex =>
//            {
//                let diff = vectors.subtract(vertex.pos, vertex.prv)
//                diff = vectors.multiply(diff, FRICTION);
//        let add = vectors.add(movement_vector, diff);
//        let new_pos = vectors.add(vertex.pos, add);
//        vertex.prv = vertex.pos;
//        vertex.pos = new_pos;
//        });
//
//        // the body should always be re-calculated as the geometric center of all points
//        // note that for circle bodies, this will always effectively be the single entiry
//        // in the verts[] array, though the positions are distinct for bookkeeping purtposes.
//        body.pos = vectors.centroid(body.verts);
    }

    @Override
    public void run(float dt)
    {
        var bodies = ecs.getComponents(Component.RigidBody2D);
        if (bodies == null || bodies.isEmpty()) return;
        bodies.forEach((entity, component) ->
        {
            RigidBody2D body2D = Component.RigidBody2D.coerce(component);
            resolveForces(entity, body2D);
            integrate(entity, body2D, dt);
        });
        // 0: get objects to calculate positions for
        // 1: resolve forces
        // 2: integrate
        // 3: find collisions
        // 4: resolve collisions
        // 5: resolve constraints
    }
}
