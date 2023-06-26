package com.controllerface.bvge.ecs.systems.physics;

import com.controllerface.bvge.Main;
import com.controllerface.bvge.cl.OpenCL;
import com.controllerface.bvge.cl.OpenCL_EX;
import com.controllerface.bvge.data.FBody2D;
import com.controllerface.bvge.data.FEdge2D;
import com.controllerface.bvge.data.FPoint2D;
import com.controllerface.bvge.data.FTransform;
import com.controllerface.bvge.ecs.ECS;
import com.controllerface.bvge.ecs.Point2D;
import com.controllerface.bvge.ecs.components.*;
import com.controllerface.bvge.ecs.systems.GameSystem;
import com.controllerface.bvge.util.MathEX;
import com.controllerface.bvge.window.Window;
import org.joml.Vector2f;

import java.nio.FloatBuffer;
import java.util.*;

public class VerletPhysics extends GameSystem
{
    private final float TICK_RATE = 1.0f / 24.0f;
    private final int SUB_STEPS = 1;
    private final int EDGE_STEPS = 1;
    private final float GRAVITY = 9.8f;
    private final float FRICTION = .980f;
    private float accumulator = 0.0f;

    private SpatialMap spatialMap = new SpatialMap();


    /**
     * These buffers are reused each tick p2 avoid creating a new one every frame and for each object.
     * They should always be zeroed before each use.
     */
    private final List<CollisionManifold> collisionBuffer = new ArrayList<>();
    private final Map<String, Set<String>> collisionProgress = new HashMap<>();
    private final Vector2f vectorBuffer1 = new Vector2f();
    private final Vector2f vectorBuffer2 = new Vector2f();
    private final Vector2f vectorBuffer3 = new Vector2f();

    public VerletPhysics(ECS ecs)
    {
        super(ecs);
    }

    private void resolveForces(String entity, FBody2D body2D)
    {
        var cp = ecs.getComponentFor(entity, Component.ControlPoints);

        if (cp == null)
        {
            return;
        }

        ControlPoints controlPoints = Component.ControlPoints.coerce(cp);
        vectorBuffer1.zero();

        if (controlPoints.isLeft())
        {
            vectorBuffer1.x -= body2D.force();
        }
        if (controlPoints.isRight())
        {
            vectorBuffer1.x += body2D.force();
        }
        if (controlPoints.isUp())
        {
            vectorBuffer1.y += body2D.force();
        }
        if (controlPoints.isDown())
        {
            vectorBuffer1.y -= body2D.force();
        }
        body2D.setAcc(vectorBuffer1);
        //body2D.getAcc().x = vectorBuffer1.x;
        //body2D.getAcc().y = vectorBuffer1.y;
    }

    private void integrate(String entitiy, FBody2D body2D, float dt)
    {
        var t = ecs.getComponentFor(entitiy, Component.Transform);
        FTransform transform = Component.Transform.coerce(t);

        body2D.mulAcc(dt * dt);
        vectorBuffer3.x = body2D.acc_x();
        vectorBuffer3.y = body2D.acc_y();
        var displacement = vectorBuffer3;

        //var displacement = body2D.getAcc().mul(dt * dt);

        for (FPoint2D point : body2D.points())
        {
            vectorBuffer1.zero();
            vectorBuffer2.zero();
            point.frameDiff(vectorBuffer1);
            //point.pos().sub(point.prv(), vectorBuffer1);
            vectorBuffer1.mul(FRICTION);
            displacement.add(vectorBuffer1, vectorBuffer2);
            point.frameSwap();
            //point.prv().set(point.pos());
            point.addPos(vectorBuffer2);
            //point.pos().add(vectorBuffer2);
        }
        //MathEX.centroid(body2D.points(), transform.position);
        //body2D.setPos(transform.position);
    }


    private PolygonProjection projectPolygon(FPoint2D[] verts, Vector2f normal)
    {
        boolean minYet = false;
        boolean maxYet = false;
        float min = 0;
        float max = 0;
        int minIndex = 0;
        for (int i = 0; i < verts.length; i++)
        {
            var v = verts[i];
            float proj = v.dotPos(normal); //v.pos().dot(normal);
            if (proj < min || !minYet)
            {
                min = proj;
                minIndex = i;
                minYet = true;
            }
            if (proj > max || !maxYet)
            {
                max = proj;
                maxYet = true;
            }
        }
        return new PolygonProjection(min, max, minIndex);
    }

    private PolygonProjection cl_projectPolygon(FPoint2D[] verts, float[] dstTest, int offset)
    {
        boolean minYet = false;
        boolean maxYet = false;
        float min = 0;
        float max = 0;
        int minIndex = 0;
        int innerOffset = offset;
        for (int i = 0; i < verts.length; i++)
        {
            float proj = dstTest[innerOffset];
            innerOffset++;
            if (proj < min || !minYet)
            {
                min = proj;
                minIndex = i;
                minYet = true;
            }
            if (proj > max || !maxYet)
            {
                max = proj;
                maxYet = true;
            }
        }
        return new PolygonProjection(min, max, minIndex);
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

    private CollisionManifold polygonCollision(FBody2D bodyA, FBody2D bodyB)
    {
        float min_distance = Float.MAX_VALUE;

        var verts1 = bodyA.points();
        var verts2 = bodyB.points();

        FBody2D vertex_o = null;
        FBody2D edge_o = null;
        int edge_indexA = 0;
        int edge_indexB = 0;
        int vert_index = 0;
        boolean invert = false;

        for (int i = 0; i < verts1.length; i++)
        {
            var b_index = (i + 1) == verts1.length
                ? 0
                : i + 1;
            var va = verts1[i];
            var vb = verts1[b_index];

            vb.subPos(va, vectorBuffer1);
            //vb.pos().sub(va.pos(), vectorBuffer1);
            vectorBuffer1.perpendicular();
            vectorBuffer1.normalize();

            var proj_a = projectPolygon(verts1, vectorBuffer1);
            var proj_b = projectPolygon(verts2, vectorBuffer1);
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
                vectorBuffer2.set(vectorBuffer1);
                edge_indexA = i;
                edge_indexB = b_index;
            }
        }

        for (int i = 0; i < verts2.length; i++)
        {
            var b_index = (i + 1) == verts2.length
                ? 0
                : i + 1;
            var va = verts2[i];
            var vb = verts2[b_index];

            vb.subPos(va, vectorBuffer1);
            //vb.pos().sub(va.pos(), vectorBuffer1);
            vectorBuffer1.perpendicular();
            vectorBuffer1.normalize();

            var proj_a = projectPolygon(verts1, vectorBuffer1);
            var proj_b = projectPolygon(verts2, vectorBuffer1);
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
                vectorBuffer2.set(vectorBuffer1);
                edge_indexA = i;
                edge_indexB = b_index;
            }
        }

        var pr = projectPolygon(vertex_o.points(), vectorBuffer2);
        vert_index = pr.index();

        min_distance = min_distance / vectorBuffer2.length();
        vectorBuffer2.normalize();

        var a = invert
            ? bodyB
            : bodyA;

        var b = invert
            ? bodyA
            : bodyB;

        var tA = ecs.getComponentFor(a.entity(), Component.Transform);
        var tB = ecs.getComponentFor(b.entity(), Component.Transform);
        FTransform transformA = Component.Transform.coerce(tA);
        FTransform transformB = Component.Transform.coerce(tB);

        var direction = new Vector2f();
        transformA.subPos(transformB, direction);
        //transformA.position.sub(transformB.position, direction);
        var dirdot = direction.dot(vectorBuffer2);
        if (dirdot < 0)
        {
            vectorBuffer2.mul(-1);
        }
        direction.set(vectorBuffer2);
        return new CollisionManifold(vertex_o,
            edge_o, direction, min_distance,
            edge_indexA, edge_indexB, vert_index);
    }


    private CollisionManifold checkCollision(FBody2D bodyA, FBody2D bodyB)
    {
        return polygonCollision(bodyA, bodyB);
    }




    private void findCollisions(FBody2D target, SpatialMap spatialMap)
    {
        var b = ecs.getComponentFor(target.entity(), Component.BoundingBox);
        QuadRectangle targetBox = Component.BoundingBox.coerce(b);
        var candidates = spatialMap.getMatches(targetBox);

        for (FBody2D candidate : candidates)
        {
            if (target == candidate)
            {
                continue;
            }
            if (target.entity().equals(candidate.entity()))
            {
                continue;
            }

            var keyA = "";
            var keyB = "";
            if (target.entity().compareTo(candidate.entity()) < 0)
            {
                keyA = target.entity();
                keyB = candidate.entity();
            }
            else
            {
                keyA = candidate.entity();
                keyB = target.entity();
            }



            if (collisionProgress.computeIfAbsent(keyA, (k)-> new HashSet<>()).contains(keyB))
            {
                continue;
            }

            collisionProgress.get(keyA).add(keyB);
            //System.out.println("keyA: " + keyA + " keyB: " + keyB);

            var bx = ecs.getComponentFor(candidate.entity(), Component.BoundingBox);
            QuadRectangle candidateBox = Component.BoundingBox.coerce(bx);
            boolean ch = doBoxesIntersect(targetBox, candidateBox);
            if (!ch)
            {
                continue;
            }

            var collision = checkCollision(target, candidate);
            if (collision == null)
            {
                continue;
            }
            collisionBuffer.add(collision);
        }
    }

    private float edgeContact(FPoint2D e1, FPoint2D e2, FPoint2D collision_vertex, Vector2f collision_vector)
    {
        float contact;
        float x_dist = e1.pos_x() - e2.pos_x();
        float y_dist = e1.pos_y() - e2.pos_y();
        if (Math.abs(x_dist) > Math.abs(y_dist))
        {
            float x_offset = (collision_vertex.pos_x() - collision_vector.x - e1.pos_x());
            float x_diff = (e2.pos_x() - e1.pos_x());
            contact = x_offset / x_diff;
        }
        else
        {
            float y_offset = (collision_vertex.pos_y() - collision_vector.y - e1.pos_y());
            float y_diff = (e2.pos_y() - e1.pos_y());
            contact = y_offset / y_diff;
        }
        return contact;
    }

    private void reactPolygon(CollisionManifold collision)
    {
        var collision_vector = collision.normal().mul(collision.depth());
        float vertex_magnitude = .5f;
        float edge_magnitude = .5f;
        var verts = collision.vertexObject().points();
        var collision_vertex = verts[collision.vert()];

        // vertex object
        if (vertex_magnitude > 0)
        {

            collision_vector.mul(vertex_magnitude, vectorBuffer1);
            //collision_vertex.pos().sub(collision_vertex.prv(), vectorBuffer2); // diff
            collision_vertex.addPos(vectorBuffer1);
            //collision_vertex.pos().sub(vectorBuffer2, collision_vertex.prv());
        }

        // edge object
        if (edge_magnitude > 0)
        {
            var edge_verts = collision.edgeObject().points();
            var e1 = edge_verts[collision.edgeA()];
            var e2 = edge_verts[collision.edgeB()];
            var edge_contact = edgeContact(e1, e2, collision_vertex, collision_vector);
            float edge_scale = 1.0f / (edge_contact * edge_contact + (1 - edge_contact) * (1 - edge_contact));
            collision_vector.mul((1 - edge_contact) * edge_magnitude * edge_scale, vectorBuffer1);
            collision_vector.mul(edge_contact * edge_magnitude * edge_scale, vectorBuffer2);
            //e1.prv().sub( e1.pos(), vectorBuffer3);
            //e2.prv().sub( e2.pos(), vectorBuffer4);
            e1.subPos(vectorBuffer1);
            e2.subPos(vectorBuffer2);
            //e1.pos().sub(vectorBuffer3, e1.prv());
            //e2.pos().sub(vectorBuffer4, e2.prv());

        }
    }

    private void updateBoundBox(FBody2D body, QuadRectangle boundingBox)
    {
        var max_x = Float.MIN_VALUE;
        var min_x = Float.MAX_VALUE;
        var max_y = Float.MIN_VALUE;
        var min_y = Float.MAX_VALUE;

        var verts = body.points();

        for (FPoint2D vertex : verts)
        {
            if (vertex.pos_x() > max_x)
            {
                max_x = vertex.pos_x();
            }
            if (vertex.pos_x() < min_x)
            {
                min_x = vertex.pos_x();
            }
            if (vertex.pos_y() > max_y)
            {
                max_y = vertex.pos_y();
            }
            if (vertex.pos_y() < min_y)
            {
                min_y = vertex.pos_y();
            }
        }

        var px = min_x;
        var py = min_y;
        var w = Math.abs(max_x - min_x);
        var h = Math.abs(max_y - min_y);

        boundingBox.setX(px);
        boundingBox.setY(py);
        boundingBox.setWidth(w);
        boundingBox.setHeight(h);
        boundingBox.setMax_x(max_x);
        boundingBox.setMin_x(min_x);
        boundingBox.setMax_y(max_y);
        boundingBox.setMin_y(min_y);
    }

    private void tickEdges()
    {
//        var buf = bodyBuffer.size() * 6;
//        var buf2 = buf * 2;

        var bodies = ecs.getComponents(Component.RigidBody2D);

        for (int i = 0; i < EDGE_STEPS; i++)
        {
//            float[] arr1 = new float[buf2];
//            float[] arr2 = new float[buf2];
//            float[] dest2 = new float[buf];
//
//            float[] offsets = new float[bodyBuffer.size() * 6 * 2];
//
//            int offset = 0;
//            for (RigidBody2D body : bodyBuffer.values())
//            {
//                var edges = body.getEdges();
//                for (Edge2D e : edges)
//                {
//                    e.p2().pos().sub(e.p1().pos(), vectorBuffer1);
//                    offsets[offset] = vectorBuffer1.x;
//                    offsets[offset + 1] = vectorBuffer1.y;
//                    arr1[offset] = e.p1().pos().x;
//                    arr1[offset + 1] = e.p1().pos().y;
//                    arr2[offset] = e.p2().pos().x;
//                    arr2[offset + 1] = e.p2().pos().y;
//                }
//
//                //resolveConstraints(body);
//            }

            // todo: come back p2 doing this with CL after getting SAT stuff further along
            //CLInstance.vectorDistance(arr1, arr2, dest2);
            for (Map.Entry<String, GameComponent> entry : bodies.entrySet())
            {
                FBody2D body = Component.RigidBody2D.coerce(entry.getValue());
                for (FEdge2D edge : body.edges())
                {
                    edge.p2().subPos(edge.p1(), vectorBuffer1);
                    //edge.p2_index().pos().sub(edge.p1_index().pos(), vectorBuffer1);
                    var length = vectorBuffer1.length();
                    float diff = length - edge.length();
                    vectorBuffer1.normalize();
                    vectorBuffer1.mul(diff * 0.5f);
                    edge.p1().addPos(vectorBuffer1);
                    edge.p2().subPos(vectorBuffer1);
                    //edge.p1_index().pos().add(vectorBuffer1);
                    //edge.p2_index().pos().sub(vectorBuffer1);
                }
            }
        }
    }
    private static boolean doBoxesIntersect(QuadRectangle a, QuadRectangle b)
    {
//        return !(a.max_x < b.x ||
//            a.max_y < b.y ||
//            a.x > b.max_x ||
//            a.y > b.max_y);

        return a.x < b.x + b.width &&
            a.x + a.width > b.x &&
            a.y < b.y + b.height &&
            a.y + a.height > b.y;
    }

    private void tickSimulation(float dt)
    {
        spatialMap.clear();

        var bodies = ecs.getComponents(Component.RigidBody2D);
        if (bodies == null || bodies.isEmpty())
        {
            return;
        }



        var bd = ecs.getComponentFor("player", Component.RigidBody2D);
        FBody2D body = Component.RigidBody2D.coerce(bd);
        resolveForces(body.entity(), body);

        var start = System.nanoTime();
        OpenCL_EX.integrate(dt);
        System.out.println("Frame time: " + (System.nanoTime() - start));

        for (Map.Entry<String, GameComponent> entry : bodies.entrySet())
        {
            String entity = entry.getKey();

            // todo: replace quad rect with FBounds
            var b = ecs.getComponentFor(entity, Component.BoundingBox);
            QuadRectangle box = Component.BoundingBox.coerce(b);

            GameComponent component = entry.getValue();
            FBody2D body2D = Component.RigidBody2D.coerce(component);

//            var t = ecs.getComponentFor(entity, Component.Transform);
//            FTransform transform = Component.Transform.coerce(t);
//            transform.position.x = body2D.pos_x();
//            transform.position.y = body2D.pos_y();

            box.setX(body2D.bounds().x());
            box.setY(body2D.bounds().y());
            box.setWidth(body2D.bounds().w());
            box.setHeight(body2D.bounds().h());
            box.setMin_x(body2D.bounds().min_x());
            box.setMax_x(body2D.bounds().max_x());
            box.setMin_y(body2D.bounds().min_y());
            box.setMax_y(body2D.bounds().max_y());

            spatialMap.add(body2D, box);
        }


        collisionProgress.clear();
        collisionBuffer.clear();

        //tickCollisions();

        for (GameComponent component : bodies.values())
        {
            FBody2D body2D = Component.RigidBody2D.coerce(component);
            findCollisions(body2D, spatialMap);
        }

        for (CollisionManifold c : collisionBuffer)
        {
            reactPolygon(c);
        }

        tickEdges();

        Window.setSP(spatialMap);
    }

    private void simulate(float dt)
    {
        this.accumulator += dt;
        while (this.accumulator >= TICK_RATE)
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
