package com.controllerface.bvge.ecs.systems.physics;

import com.controllerface.bvge.Transform;
import com.controllerface.bvge.ecs.*;
import com.controllerface.bvge.ecs.systems.GameSystem;
import com.controllerface.bvge.util.MathEX;
import com.controllerface.bvge.util.quadtree.QuadRectangle;
import com.controllerface.bvge.util.quadtree.QuadTree;
import org.joml.Vector2f;

import java.util.*;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class VerletPhysics extends GameSystem
{
    private final float TICK_RATE = 1.0f / 60.0f;
    private final int SUB_STEPS = 1;
    private final int EDGE_STEPS = 1;
    private final float GRAVITY = 9.8f;
    private final float FRICTION = 0.991f;
    private float accumulator = 0.0f;

    /**
     * These buffers are reused each tick to avoid creating a new one every frame and for each object.
     * They should always be zeroed before each use.
     */
    private final Map<String, RigidBody2D> bodyBuffer = new HashMap<>();
    private final List<CollisionManifold> collisionBuffer = new ArrayList<>();
    private final ThreadLocal<Vector2f> vectorBuffer1 = new ThreadLocal<>();
    private final ThreadLocal<Vector2f> vectorBuffer2 = new ThreadLocal<>();
//    private final Vector2f vectorBuffer1 = new Vector2f();
//    private final Vector2f vectorBuffer2 = new Vector2f();
//    private final Vector2f vectorBuffer3 = new Vector2f();
//    private final Vector2f vectorBuffer4 = new Vector2f();

    public VerletPhysics(ECS ecs)
    {
        super(ecs);
    }

    private void resolveForces(String entity, RigidBody2D body2D)
    {
        var cp = ecs.getComponentFor(entity, Component.ControlPoints);

        if (cp == null) return;

        ControlPoints controlPoints = Component.ControlPoints.coerce(cp);
        vectorBuffer1.get().zero();

        if (controlPoints.isLeft())
        {
            vectorBuffer1.get().x -= body2D.getForce();
        }
        if (controlPoints.isRight())
        {
            vectorBuffer1.get().x += body2D.getForce();
        }
        if (controlPoints.isUp())
        {
            vectorBuffer1.get().y += body2D.getForce();
        }
        if (controlPoints.isDown())
        {
            vectorBuffer1.get().y -= body2D.getForce();
        }
        body2D.getAcc().x = vectorBuffer1.get().x;
        body2D.getAcc().y = vectorBuffer1.get().y;
    }

    private void integrate(String entitiy, RigidBody2D body2D, float dt)
    {
        var t = ecs.getComponentFor(entitiy, Component.Transform);
        Transform transform = Component.Transform.coerce(t);
        var displacement = body2D.getAcc().mul(dt * dt);
        for (Point2D point : body2D.getVerts())
        {
            vectorBuffer1.get().zero();
            vectorBuffer2.get().zero();
            point.pos().sub(point.prv(), vectorBuffer1.get());
            vectorBuffer1.get().mul(FRICTION);
            displacement.add(vectorBuffer1.get(), vectorBuffer2.get());
            point.prv().set(point.pos());
            point.pos().add(vectorBuffer2.get());
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

            vb.pos().sub(va.pos(), vectorBuffer1.get());
            vectorBuffer1.get().perpendicular();
            vectorBuffer1.get().normalize();

            var proj_a = projectPolygon(verts1, vectorBuffer1.get());
            var proj_b = projectPolygon(verts2, vectorBuffer1.get());
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
                vectorBuffer2.get().set(vectorBuffer1.get());
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

            vb.pos().sub(va.pos(), vectorBuffer1.get());
            vectorBuffer1.get().perpendicular();
            vectorBuffer1.get().normalize();

            var proj_a = projectPolygon(verts1, vectorBuffer1.get());
            var proj_b = projectPolygon(verts2, vectorBuffer1.get());
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
                vectorBuffer2.get().set(vectorBuffer1.get());
                edge_indexA = i;
                edge_indexB = b_index;
            }
        }

        var pr = projectPolygon(vertex_o.getVerts(), vectorBuffer2.get());
        vert_index = pr.index();

        min_distance = min_distance / vectorBuffer2.get().length();
        vectorBuffer2.get().normalize();

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
        var dirdot = direction.dot(vectorBuffer2.get());
        if (dirdot < 0)
        {
            vectorBuffer2.get().mul(-1);
        }
        direction.set(vectorBuffer2.get());
        return new CollisionManifold(vertex_o,
            edge_o, direction, min_distance,
            edge_indexA, edge_indexB, vert_index);
    }


    private CollisionManifold checkCollision(RigidBody2D bodyA, RigidBody2D bodyB)
    {
        return polygonCollision(bodyA, bodyB);
    }

    private final Set<String> keyCache = new HashSet<>();

    private void findCollisions(RigidBody2D target, QuadTree<RigidBody2D> quadTree)
    {
        keyCache.clear();
        var b = ecs.getComponentFor(target.getEntitiy(), Component.BoundingBox);
        QuadRectangle targetBox = Component.BoundingBox.coerce(b);
        var candidates = new ArrayList<RigidBody2D>();
        quadTree.getElements(candidates, targetBox);
        System.out.println("dropped: " + (bodyBuffer.size() - candidates.size()));

        // todo: get candidates from quadtree
        for (RigidBody2D candidate : candidates)
        {
            if (target == candidate) continue;
            if (target.getEntitiy().equals(candidate.getEntitiy())) continue;
            var k1 = target.getEntitiy() + candidate.getEntitiy();
            var k2 = candidate.getEntitiy() + target.getEntitiy();
            if (keyCache.contains(k1) || keyCache.contains(k2))
            {
                return;
            }

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
        var collision_vector = collision.normal().mul(collision.depth());
        float vertex_magnitude = .5f;
        float edge_magnitude = .5f;
        var verts = collision.vertexObject().getVerts();
        var collision_vertex = verts.get(collision.vert());

        // vertex object
        if (vertex_magnitude > 0)
        {

            collision_vector.mul(vertex_magnitude, vectorBuffer1.get());
            //collision_vertex.pos().sub(collision_vertex.prv(), vectorBuffer2.get()); // diff
            collision_vertex.pos().add(vectorBuffer1.get());
            //collision_vertex.pos().sub(vectorBuffer2.get(), collision_vertex.prv());
        }

        // edge object
        if (edge_magnitude > 0)
        {
            var edge_verts = collision.edgeObject().getVerts();
            var e1 = edge_verts.get(collision.edgeA());
            var e2 = edge_verts.get(collision.edgeB());
            var edge_contact = edgeContact(e1, e2, collision_vertex,collision_vector);
            float edge_scale = 1.0f / ( edge_contact * edge_contact + (1 - edge_contact) * (1 - edge_contact));
            collision_vector.mul((1 - edge_contact) * edge_magnitude * edge_scale, vectorBuffer1.get());
            collision_vector.mul(edge_contact * edge_magnitude *edge_scale, vectorBuffer2.get());
            //e1.prv().sub( e1.pos(), vectorBuffer3);
            //e2.prv().sub( e2.pos(), vectorBuffer4);
            e1.pos().sub(vectorBuffer1.get());
            e2.pos().sub(vectorBuffer2.get());
            //e1.pos().sub(vectorBuffer3, e1.prv());
            //e2.pos().sub(vectorBuffer4, e2.prv());

        }
    }

    private void resolveConstraints(RigidBody2D body2D)
    {
        for (Edge2D edge : body2D.getEdges())
        {
            edge.p2().pos().sub(edge.p1().pos(), vectorBuffer1.get());
            var length = edge.p2().pos().sub(edge.p1().pos(), vectorBuffer2.get()).length();
            float diff = length - edge.length();
            vectorBuffer1.get().normalize();
            vectorBuffer1.get().mul(diff * 0.5f);
            edge.p1().pos().add(vectorBuffer1.get());
            edge.p2().pos().sub(vectorBuffer1.get());
        }
    }

    private void updateBoundBox(RigidBody2D body, QuadRectangle boundingBox)
    {
        var max_x = Float.MIN_VALUE;
        var min_x = Float.MAX_VALUE;
        var max_y = Float.MIN_VALUE;
        var min_y = Float.MAX_VALUE;

        var verts = body.getVerts();

        for (Point2D vertex : verts)
        {
            if (vertex.pos().x > max_x)
            {
                max_x = vertex.pos().x;
            }
            if (vertex.pos().x < min_x)
            {
                min_x = vertex.pos().x;
            }
            if (vertex.pos().y > max_y)
            {
                max_y = vertex.pos().y;
            }
            if (vertex.pos().y < min_y)
            {
                min_y = vertex.pos().y;
            }
        }

        var px = min_x;
        var py = min_y;
        var w = Math.abs(max_x - min_x) + 20;
        var h = Math.abs(max_y - min_y) + 20;

        boundingBox.setX(px);
        boundingBox.setY(py);
        boundingBox.setWidth(w);
        boundingBox.setHeight(h);
    }

    private void setThreadvectorBuffers()
    {
        if (vectorBuffer1.get() == null)
        {
            vectorBuffer1.set(new Vector2f());
        }
        if (vectorBuffer2.get() == null)
        {
            vectorBuffer2.set(new Vector2f());
        }
    }

    private void tickSimulation(float dt)
    {
        setThreadvectorBuffers();

        QuadTree<RigidBody2D> quadTree = new QuadTree<>(new QuadRectangle(0, 0, 1920, 1080), 0);
        var bodies = ecs.getComponents(Component.RigidBody2D);
        if (bodies == null || bodies.isEmpty()) return;

        bodyBuffer.clear();
        for (Map.Entry<String, GameComponent> entry : bodies.entrySet())
        {
            String entity = entry.getKey();
            var b = ecs.getComponentFor(entity, Component.BoundingBox);
            QuadRectangle box = Component.BoundingBox.coerce(b);
            GameComponent component = entry.getValue();
            RigidBody2D body2D = Component.RigidBody2D.coerce(component);
            resolveForces(entity, body2D);
            integrate(entity, body2D, dt);
            bodyBuffer.put(entity, body2D);
            updateBoundBox(body2D, box);
            quadTree.insert(box, body2D);
        }

        collisionBuffer.clear();

        for (RigidBody2D body : bodyBuffer.values())
        {
            findCollisions(body, quadTree);
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
