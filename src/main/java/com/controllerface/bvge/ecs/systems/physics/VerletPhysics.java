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
    private final float FRICTION = 0.997f;
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
        float min = normal.dot(verts.get(0).pos()); //vectors.dot(axis, verts[0].pos);
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

    private final Vector2f edgeBuffer = new Vector2f();
    private final Vector2f normalBuffer = new Vector2f();

    private CollisionManifold polygonCollision(RigidBody2D bodyA, RigidBody2D bodyB)
    {
        float min_distance = Float.MAX_VALUE;

        var verts1 = bodyA.getVerts();
        var verts2 = bodyB.getVerts();

        RigidBody2D vertex_o = null;
        RigidBody2D edge_o = null;
        int edge_indexA = 0;
        int edge_indexB = 0;
        int vert_index = 0;
        boolean invert = false;

        for (int i = 0; i < verts1.size(); i++)
        {
            var b_index = (i + 1) == verts1.size()
                ? 0
                : i + 1;
            var va = verts1.get(i);
            var vb = verts1.get(b_index);

            vb.pos().sub(va.pos(), edgeBuffer);
            edgeBuffer.perpendicular();
            edgeBuffer.normalize();

            var proj_a = projectPolygon(verts1, edgeBuffer);
            var proj_b = projectPolygon(verts2, edgeBuffer);
            var distance = polygonDistance(proj_a, proj_b);
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
                normalBuffer.set(edgeBuffer);
                edge_indexA = i;
                edge_indexB = b_index;
            }
        }


        for (int i = 0; i < verts2.size(); i++)
        {
            var b_index = (i + 1) == verts2.size()
                ? 0
                : i + 1;
            var va = verts2.get(i);
            var vb = verts2.get(b_index);

            vb.pos().sub(va.pos(), edgeBuffer);
            edgeBuffer.perpendicular();
            edgeBuffer.normalize();

            var proj_a = projectPolygon(verts1, edgeBuffer);
            var proj_b = projectPolygon(verts2, edgeBuffer);
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
                normalBuffer.set(edgeBuffer);
                edge_indexA = i;
                edge_indexB = b_index;
            }
        }

        var pr = projectPolygon(vertex_o.getVerts(), normalBuffer);
        vert_index = pr.index();

        min_distance = min_distance / normalBuffer.length();
        normalBuffer.normalize();

        var a = invert
            ? bodyB
            : bodyA;

        var b = invert
            ? bodyA
            : bodyB;

        var tA = ecs.getComponentFor(a.getEntitiy(), Component.Transform);
        var tB = ecs.getComponentFor(b.getEntitiy(), Component.Transform);
        Transform transformA = Component.Transform.coerce(tA);
        Transform transformB = Component.Transform.coerce(tB);

        var direction = new Vector2f();
        transformA.position.sub(transformB.position, direction);
        var dirdot = direction.dot(normalBuffer);
        if (dirdot < 0)
        {
            normalBuffer.mul(-1);
        }
        direction.set(normalBuffer);
        return new CollisionManifold(vertex_o,
            edge_o, direction, min_distance,
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

    private float edgeContact(Point2D e1, Point2D e2, Point2D collision_vertex, Vector2f collision_vector)
    {
        float contact;
        float x_dist = e1.pos().x - e2.pos().x;
        float y_dist = e1.pos().y - e2.pos().y;
        if (Math.abs(x_dist) > Math.abs(y_dist))
        {
            float x_offset = (collision_vertex.pos().x - collision_vector.x - e1.pos().x);
            float x_diff = (e2.pos().x - e1.pos().x);
            contact = x_offset / x_diff;
        }
        else
        {
            float y_offset = (collision_vertex.pos().y - collision_vector.y - e1.pos().y);
            float y_diff = (e2.pos().y - e1.pos().y);
            contact = y_offset / y_diff;
        }
        return contact;
    }

    private final Vector2f vectorBuffer1 = new Vector2f();
    private final Vector2f vectorBuffer2 = new Vector2f();
    private final Vector2f vectorBuffer3 = new Vector2f();

    private void reactPolygon(CollisionManifold collision)
    {
        var collision_vector = collision.normal().mul(collision.depth());
        float vertex_magnitude = .5f;
        float edge_magnitude = .5f;
        var verts = collision.vertexObject().getVerts();
        var collision_vertex = verts.get(collision.vert());

        // vertex object
        if (vertex_magnitude > 0)
        {
            collision_vector.mul(vertex_magnitude, vectorBuffer1);
            collision_vertex.pos().add(vectorBuffer1);
        }

        // edge object
        if (edge_magnitude > 0)
        {
            var edge_verts = collision.edgeObject().getVerts();
            var e1 = edge_verts.get(collision.edgeA());
            var e2 = edge_verts.get(collision.edgeB());
            var edge_contact = edgeContact(e1, e2, collision_vertex,collision_vector);
            float edge_scale = 1.0f / ( edge_contact * edge_contact + (1 - edge_contact) * (1 - edge_contact));
            collision_vector.mul((1 - edge_contact) * edge_magnitude * edge_scale, vectorBuffer1);
            collision_vector.mul(edge_contact * edge_magnitude *edge_scale, vectorBuffer2);
            e1.pos().sub(vectorBuffer1);
            e2.pos().sub(vectorBuffer2);
        }
    }

    private void resolveConstraints(RigidBody2D body2D)
    {
        for (Edge2D edge : body2D.getEdges())
        {
            edge.p2().pos().sub(edge.p1().pos(), vectorBuffer1);
            var length = edge.p2().pos().sub(edge.p1().pos(), vectorBuffer2).length();
            float diff = length - edge.length();
                vectorBuffer1.normalize();
                vectorBuffer1.mul(diff * 0.5f);
            edge.p1().pos().add(vectorBuffer1);
            edge.p2().pos().sub(vectorBuffer1);
        }
    }




    private void tickSimulation(float dt)
    {
        //        var kdtree = new KDTree<String>(2);
        var bodies = ecs.getComponents(Component.RigidBody2D);
        if (bodies == null || bodies.isEmpty()) return;

        bodyBuffer.clear();
        for (Map.Entry<String, GameComponent> entry : bodies.entrySet())
        {
            String entity = entry.getKey();
            GameComponent component = entry.getValue();
            RigidBody2D body2D = Component.RigidBody2D.coerce(component);
            resolveForces(entity, body2D);
            integrate(entity, body2D, dt);
            bodyBuffer.put(entity, body2D);
        }

        collisionBuffer.clear();

        for (RigidBody2D body : bodyBuffer.values())
        {
            findCollisions(body);
        }

        for (CollisionManifold c : collisionBuffer)
        {
            reactPolygon(c);
        }

        for (RigidBody2D body : bodyBuffer.values())
        {
            for (int i =0; i< EDGE_STEPS; i++)
            {
                resolveConstraints(body);
            }
        }
    }

    private void simulate(float dt)
    {
//        if (!this.last_timestamp)
//        {
//            this.last_timestamp = 0.0;
//        }

        //let frameTime = (now - this.last_timestamp) / 1000;

        //this.last_timestamp = now;
        this.accumulator += dt;
        while ( this.accumulator >= TICK_RATE )
        {
            float sub_step = TICK_RATE / SUB_STEPS;
            for (int i = 0; i < SUB_STEPS; i++)
            {
                this.tickSimulation(sub_step);
                this.accumulator -= sub_step;
            }
        }

        float drift = this.accumulator / TICK_RATE;
        if (drift != 0)
        {
            //this.lerp(drift);
        }
    }


    @Override
    public void run(float dt)
    {
        simulate(dt);
    }
}
