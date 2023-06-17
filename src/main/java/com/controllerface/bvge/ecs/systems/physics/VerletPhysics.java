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

    private CollisionManifold polygonCollision(RigidBody2D bodyA, RigidBody2D bodyB)
    {
        float min_distance = 10000;
        Vector2f normal = null;

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

            var edge = vb.pos().sub(va.pos(), new Vector2f());
            var axis = new Vector2f(edge).perpendicular();
            axis.normalize();

            var proj_a = projectPolygon(verts1, axis);
            var proj_b = projectPolygon(verts2, axis);
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
                normal = axis;
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

            var edge = vb.pos().sub(va.pos(), new Vector2f());
            var axis = new Vector2f(edge).perpendicular();
            axis.normalize();

            var proj_a = projectPolygon(verts1, axis);
            var proj_b = projectPolygon(verts2, axis);
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
                normal = axis;
                edge_indexA = i;
                edge_indexB = b_index;
            }
        }

        var pr = projectPolygon(vertex_o.getVerts(), normal);
        vert_index = pr.index();

        min_distance = min_distance / normal.length();
        normal.normalize();

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

        var direction = new Vector2f();//centerA.sub(centerB);
        transformA.position.sub(transformB.position, direction);
        var dirdot = direction.dot(normal);
        if (dirdot < 0)
        {
            normal.mul(-1);
        }
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


    private void reactPolygon(CollisionManifold collision)
    {

        var collision_vector = collision.normal().mul(collision.depth(), new Vector2f());
        //vectors.multiply(collision.norm, collision.depth);

        float vertex_magnitude = .5f;
//            collision.edge_o.static
//            ? 1
//            : (collision.vertex_o.static)
//                ? 0
//                : .5;

        float edge_magnitude = .5f;
//            collision.vertex_o.static
//            ? 1
//            : (collision.edge_o.static)
//                ? 0
//                : .5;

        var verts = collision.vertexObject().getVerts();
        var collision_vertex = verts.get(collision.vert());

        var vt = ecs.getComponentFor(collision.vertexObject().getEntitiy(), Component.Transform);
        var et = ecs.getComponentFor(collision.edgeObject().getEntitiy(), Component.Transform);
        Transform vTransform = Component.Transform.coerce(vt);
        Transform eTransform = Component.Transform.coerce(et);

        // vertex object
        if (vertex_magnitude > 0)
        {
            var vertex_reaction = collision_vector.mul(vertex_magnitude, new Vector2f());
                //vectors.multiply(collision_vector, vertex_magnitude);
            var new_v = collision_vertex.pos().add(vertex_reaction, new Vector2f());
                //vectors.add(collision_vertex.pos, vertex_reaction)
            verts.get(collision.vert()).pos().set(new_v);
                //verts[collision.vert_index].pos = new_v;
            MathEX.centroid(verts, vTransform.position);
                //collision.vertex_o.pos = vectors.centroid(verts);
                //utils.clamp(verts[collision.vert_index]);
        }

        // edge object
        if (edge_magnitude > 0)
        {
            var edge_verts = collision.edgeObject().getVerts();
            var e1 = edge_verts.get(collision.edgeA());
            var e2 = edge_verts.get(collision.edgeB());
            var edge_contact = edgeContact(e1, e2, collision_vertex,collision_vector);
            //utils.edge_contact(e1, e2, collision_vertex, collision_vector);
            float edge_scale = 1.0f / ( edge_contact * edge_contact + (1 - edge_contact) * (1 - edge_contact));
            var e1_reaction = collision_vector.mul((1 - edge_contact) * edge_magnitude * edge_scale, new Vector2f());
                //vectors.multiply(collision_vector, (1 - edge_contact) * edge_magnitude * edge_scale);
            var e2_reaction = collision_vector.mul(edge_contact * edge_magnitude *edge_scale, new Vector2f());
                //vectors.multiply(collision_vector, edge_contact * edge_magnitude *edge_scale);
            var new_e1 = e1.pos().sub(e1_reaction, new Vector2f());
                //vectors.subtract(e1.pos, e1_reaction);
            var new_e2 = e2.pos().sub(e2_reaction, new Vector2f());
                //vectors.subtract(e2.pos, e2_reaction);

            e1.pos().set(new_e1);
            e2.pos().set(new_e2);

//            edge_verts[edge_index.a].pos = new_e1;
//            edge_verts[edge_index.b].pos = new_e2;
//            collision.edge_o.pos = vectors.centroid(edge_verts);
//            utils.clamp(edge_verts[edge_index.a]);
//            utils.clamp(edge_verts[edge_index.b]);
        }
    }

    private void resolveConstraints(RigidBody2D body2D)
    {
        for (Edge2D edge : body2D.getEdges())
        {
            var v1v2 = edge.p2().pos().sub(edge.p1().pos(), new Vector2f());
                //vectors.subtract(object.verts[edge.b].pos, object.verts[edge.a].pos);
            var length = edge.p2().pos().sub(edge.p1().pos(), new Vector2f()).length();
                //vectors.magnitude(v1v2);
            float diff = length - edge.length();
            var vn = v1v2.normalize(new Vector2f());
                //vectors.unit(v1v2);
            var offset = vn.mul(diff * 0.5f, new Vector2f());
                //vectors.multiply(vn, diff * 0.5);
            var new_a = edge.p1().pos().add(offset, new Vector2f());
                //vectors.add(object.verts[edge.a].pos, offset);
            var new_b = edge.p2().pos().sub(offset, new Vector2f());
                //vectors.subtract(object.verts[edge.b].pos, offset);
            edge.p1().pos().set(new_a);
            edge.p2().pos().set(new_b);
            //object.verts[edge.a].pos = new_a;
            //object.verts[edge.b].pos = new_b;
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

        // Pass 3: React to collisions
        for (CollisionManifold c : collisionBuffer)
        {
            reactPolygon(c);         // 4: resolve collisions
        }

        for (RigidBody2D body : bodyBuffer.values())
        {
            resolveConstraints(body); // 5: resolve constraints
        }


    }
}
