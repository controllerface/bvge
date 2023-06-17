package com.controllerface.bvge.ecs.systems.physics;

import com.controllerface.bvge.Transform;
import com.controllerface.bvge.ecs.*;
import com.controllerface.bvge.ecs.systems.GameSystem;
import com.controllerface.bvge.util.MathEX;
import org.joml.Vector2f;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class VerletPhysics extends GameSystem
{
    private final float TICK_RATE = 1.0f / 60.0f;
    private final int SUB_STEPS = 8;
    private final int EDGE_STEPS = 8;
    private final float GRAVITY = 9.8f;
    private final float FRICTION = 0.980f;
    private float accumulator = 0.0f;

    /**
     * These buffers are reused each tick to avoid creating a new one every frame and for each object.
     * They should always be zeroed before each use.
     */
    private final Map<String, RigidBody2D> bodyBuffer = new HashMap<>();
    private final List<CollisionManifold> collisionBuffer = new ArrayList<>();
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


    private PolygonProjection projectPolygon(List<Point2D> verts, Vector2f normal)
    {
        float min =  normal.dot(verts.get(0).pos()); //vectors.dot(axis, verts[0].pos);
        float max = min;
        int index = 0;
        for (int i = 0; i < verts.size(); i++)
        {
            var v = verts.get(i);
            float proj = v.pos().dot(normal); //vectors.dot(v.pos, axis);
            if (proj < min)
            {
                min = proj;
                index = i;
            }
            if (proj > max)
            {
                max = proj;
            }
        }
        return new PolygonProjection(min, max, index);
    }

    private float polygonDistance(PolygonProjection projA, PolygonProjection projB)
    {
        if (projA.min() < projB.min())
        {
            return projB.min() - projA.max();
        }
        else
        {
            return projA.min() - projB.max();
        }
    }

    private CollisionManifold polygonCollision(RigidBody2D bodyA, RigidBody2D bodyB)
    {
        float min_distance = 10000;
        Vector2f normal = null;

        var verts1 = bodyA.getVerts();
        var verts2 = bodyB.getVerts();

        RigidBody2D vertex_o = null;
        RigidBody2D edge_o = null;
        //let collision_edge = { a: null, b: null };
        int edge_indexA = 0;
        int edge_indexB = 0;
        int vert_index = 0;
        boolean invert = false;

        for (int i = 0; i < verts1.size(); i++)
        {
            var b_index = (i + 1) == verts1.size() ? 0 : i + 1;
            var va = verts1.get(i);
            var vb = verts1.get(b_index);

            var edge = vb.pos().sub(va.pos(), new Vector2f());//vectors.subtract(vb.pos, va.pos);
            edge.perpendicular();  //let axis = vectors.normal(edge);
            edge.normalize();      //axis = vectors.unit(axis);
            var proj_a = projectPolygon(verts1, edge); //utils.polygon_project(verts1, axis);
            var proj_b = projectPolygon(verts2, edge); //utils.polygon_project(verts2, axis);
            var distance = polygonDistance(proj_a, proj_b); //utils.polygon_distance(proj_a, proj_b);
            if (distance > 0)
            {
                return null;
            }
            var abs_distance = Math.abs(distance);
            if (abs_distance < min_distance)
            {
                vertex_o = bodyB;
                invert = true;
                edge_o = bodyA;
                min_distance = abs_distance;
                normal = edge;
                edge_indexA = i;
                edge_indexB = b_index;
//                collision_edge.a = i;
//                collision_edge.b = b_index;
            }
        }


        for (int i = 0; i < verts2.size(); i++)
        {
            var b_index = (i + 1) == verts2.size() ? 0 : i + 1;
            var va = verts2.get(i);
            var vb = verts2.get(b_index);
            var edge = vb.pos().sub(va.pos(), new Vector2f());
            edge.perpendicular();
            edge.normalize();
            var proj_a = projectPolygon(verts1, edge);
            var proj_b = projectPolygon(verts2, edge);
            var distance = polygonDistance(proj_a, proj_b);
            if (distance > 0)
            {
                return null;
            }
            var abs_distance = Math.abs(distance);
            if (abs_distance < min_distance)
            {
                vertex_o = bodyA;
                invert = false;
                edge_o = bodyB;
                min_distance = abs_distance;
                normal = edge;
                edge_indexA = i;
                edge_indexB = b_index;
            }
        }

        var pr = projectPolygon(vertex_o.getVerts(), normal); //utils.polygon_project(vertex_o.verts, normal);
        vert_index = pr.index();

        min_distance = min_distance / normal.length();
        normal.normalize(); // = vectors.unit(normal);

        var a = invert ? bodyB : bodyA;
        var b = invert ? bodyA : bodyB;

        var centerA = new Vector2f();
        var centerB = new Vector2f();
        MathEX.centroid(a.getVerts(), centerA);
        MathEX.centroid(b.getVerts(), centerB);

        var direction = centerA.sub(centerB); //vectors.subtract(a.pos, b.pos);
        var dirdot = direction.dot(normal); //vectors.dot(direction, normal);
        if (dirdot < 0)
        {
            normal.mul(-1); //vectors.multiply(normal, -1);
        }

//        let result =
//            {
//                vertex_o: vertex_o,
//                edge_o: edge_o,
//                depth: min_distance,
//                norm: normal,
//                edge_index: collision_edge,
//                vert_index: vert_index,
//                type: 'polygon-polygon'
//            };
        return new CollisionManifold(vertex_o,
            edge_o, normal, min_distance,
            edge_indexA, edge_indexB, vert_index);
    }




    private CollisionManifold checkCollision(RigidBody2D bodyA, RigidBody2D bodyB)
    {
        return polygonCollision(bodyA, bodyB);
    }

    private void findCollisions(RigidBody2D target)
    {
        for (RigidBody2D candidate : bodyBuffer.values())
        {
            if (target == candidate) continue;
            var collision = checkCollision(target, candidate);
            if (collision == null) continue;
            collisionBuffer.add(collision);
        }
    }

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

        collisionBuffer.clear();

        // Pass 2: Detect collisions
        for (RigidBody2D body : bodyBuffer.values())
        {
            findCollisions(body); // 3: find collisions
        }

        // 4: resolve collisions
        // 5: resolve constraints
    }
}
