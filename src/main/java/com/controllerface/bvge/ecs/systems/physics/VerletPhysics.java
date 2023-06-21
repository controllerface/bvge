package com.controllerface.bvge.ecs.systems.physics;

import com.controllerface.bvge.CLInstance;
import com.controllerface.bvge.Transform;
import com.controllerface.bvge.ecs.*;
import com.controllerface.bvge.ecs.systems.GameSystem;
import com.controllerface.bvge.util.MathEX;
import com.controllerface.bvge.util.quadtree.QuadRectangle;
import com.controllerface.bvge.util.quadtree.QuadTree;
import com.controllerface.bvge.window.Window;
import org.joml.Vector2f;

import java.nio.FloatBuffer;
import java.util.*;

public class VerletPhysics extends GameSystem
{
    private final float TICK_RATE = 1.0f / 60.0f;
    private final int SUB_STEPS = 1;
    private final int EDGE_STEPS = 4;
    private final float GRAVITY = 9.8f;
    private final float FRICTION = .970f;
    private float accumulator = 0.0f;

    /**
     * These buffers are reused each tick to avoid creating a new one every frame and for each object.
     * They should always be zeroed before each use.
     */
    private final Map<String, RigidBody2D> bodyBuffer = new HashMap<>();
    private final List<CollisionManifold> collisionBuffer = new ArrayList<>();
    private final Vector2f vectorBuffer1 = new Vector2f();
    private final Vector2f vectorBuffer2 = new Vector2f();
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

        if (cp == null)
        {
            return;
        }

        ControlPoints controlPoints = Component.ControlPoints.coerce(cp);
        vectorBuffer1.zero();

        if (controlPoints.isLeft())
        {
            vectorBuffer1.x -= body2D.getForce();
        }
        if (controlPoints.isRight())
        {
            vectorBuffer1.x += body2D.getForce();
        }
        if (controlPoints.isUp())
        {
            vectorBuffer1.y += body2D.getForce();
        }
        if (controlPoints.isDown())
        {
            vectorBuffer1.y -= body2D.getForce();
        }
        body2D.getAcc().x = vectorBuffer1.x;
        body2D.getAcc().y = vectorBuffer1.y;
    }

    private void integrate(String entitiy, RigidBody2D body2D, float dt)
    {
        var t = ecs.getComponentFor(entitiy, Component.Transform);
        Transform transform = Component.Transform.coerce(t);
        var displacement = body2D.getAcc().mul(dt * dt);
        for (Point2D point : body2D.getVerts())
        {
            vectorBuffer1.zero();
            vectorBuffer2.zero();
            point.pos().sub(point.prv(), vectorBuffer1);
            vectorBuffer1.mul(FRICTION);
            displacement.add(vectorBuffer1, vectorBuffer2);
            point.prv().set(point.pos());
            point.pos().add(vectorBuffer2);
        }
        MathEX.centroid(body2D.getVerts(), transform.position);
    }


    private PolygonProjection projectPolygon(List<Point2D> verts, Vector2f normal)
    {
        boolean minYet = false;
        boolean maxYet = false;
        float min = 0;
        float max = 0;
        int minIndex = 0;
        for (int i = 0; i < verts.size(); i++)
        {
            var v = verts.get(i);
            float proj = v.pos().dot(normal);
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

    private PolygonProjection projectPolygon2(List<Point2D> verts, Vector2f normal, float[] dstTest, int offset)
    {
        boolean minYet = false;
        boolean maxYet = false;
        float min = 0;
        float max = 0;
        int minIndex = 0;
        int innerOffset = offset;
        for (int i = 0; i < verts.size(); i++)
        {
            var v = verts.get(i);
            float dstTestVal = dstTest[innerOffset];
            float proj = dstTestVal;//v.pos().dot(normal);
            if (dstTestVal != proj)
            {
                //System.out.println("error: expected: " + dstTestVal + " was: " + proj);
            }
            if (dstTestVal == proj)
            {
                //System.out.println("match: " + dstTestVal + " was: " + proj);
            }
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

    int cl_buffer_offset = 0;

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

            vb.pos().sub(va.pos(), vectorBuffer1);
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

        for (int i = 0; i < verts2.size(); i++)
        {
            var b_index = (i + 1) == verts2.size()
                ? 0
                : i + 1;
            var va = verts2.get(i);
            var vb = verts2.get(b_index);

            vb.pos().sub(va.pos(), vectorBuffer1);
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

        var pr = projectPolygon(vertex_o.getVerts(), vectorBuffer2);
        vert_index = pr.index();

        min_distance = min_distance / vectorBuffer2.length();
        vectorBuffer2.normalize();

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


    private CollisionManifold checkCollision(RigidBody2D bodyA, RigidBody2D bodyB)
    {
        return polygonCollision(bodyA, bodyB);
    }

    private final Set<String> keyCache = new HashSet<>();

    private void findCollisions(RigidBody2D target, QuadTree<RigidBody2D> quadTree, SpatialMap spatialMap)
    {
        keyCache.clear();
        var b = ecs.getComponentFor(target.getEntitiy(), Component.BoundingBox);
        QuadRectangle targetBox = Component.BoundingBox.coerce(b);
        //var candidates = new ArrayList<RigidBody2D>();
        //quadTree.getElements(candidates, targetBox);
        var c2s = spatialMap.getMatches(targetBox);
        //System.out.println("c2: " + c2s.size() + " c1: " + candidates.size());

        for (RigidBody2D candidate : c2s)
        {
            if (target == candidate)
            {
                continue;
            }
            if (target.getEntitiy().equals(candidate.getEntitiy()))
            {
                continue;
            }
            var k1 = target.getEntitiy() + candidate.getEntitiy();
            var k2 = candidate.getEntitiy() + target.getEntitiy();
            if (keyCache.contains(k1) || keyCache.contains(k2))
            {
                continue;
            }

            var bx = ecs.getComponentFor(candidate.getEntitiy(), Component.BoundingBox);
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

            collision_vector.mul(vertex_magnitude, vectorBuffer1);
            //collision_vertex.pos().sub(collision_vertex.prv(), vectorBuffer2); // diff
            collision_vertex.pos().add(vectorBuffer1);
            //collision_vertex.pos().sub(vectorBuffer2, collision_vertex.prv());
        }

        // edge object
        if (edge_magnitude > 0)
        {
            var edge_verts = collision.edgeObject().getVerts();
            var e1 = edge_verts.get(collision.edgeA());
            var e2 = edge_verts.get(collision.edgeB());
            var edge_contact = edgeContact(e1, e2, collision_vertex, collision_vector);
            float edge_scale = 1.0f / (edge_contact * edge_contact + (1 - edge_contact) * (1 - edge_contact));
            collision_vector.mul((1 - edge_contact) * edge_magnitude * edge_scale, vectorBuffer1);
            collision_vector.mul(edge_contact * edge_magnitude * edge_scale, vectorBuffer2);
            //e1.prv().sub( e1.pos(), vectorBuffer3);
            //e2.prv().sub( e2.pos(), vectorBuffer4);
            e1.pos().sub(vectorBuffer1);
            e2.pos().sub(vectorBuffer2);
            //e1.pos().sub(vectorBuffer3, e1.prv());
            //e2.pos().sub(vectorBuffer4, e2.prv());

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

    boolean runyet = false;
    QuadTree<RigidBody2D> quadTree = new QuadTree<>(new QuadRectangle(0, 0, 1920, 1080), 0);

    private void tickEdges()
    {
        var buf = bodyBuffer.size() * 6;
        var buf2 = buf * 2;

        for (int i = 0; i < EDGE_STEPS; i++)
        {
            float[] arr1 = new float[buf2];
            float[] arr2 = new float[buf2];
            float[] dest2 = new float[buf];

            float[] offsets = new float[bodyBuffer.size() * 6 * 2];

            int offset = 0;
            for (RigidBody2D body : bodyBuffer.values())
            {
                var edges = body.getEdges();
                for (Edge2D e : edges)
                {
                    e.p2().pos().sub(e.p1().pos(), vectorBuffer1);
                    offsets[offset] = vectorBuffer1.x;
                    offsets[offset + 1] = vectorBuffer1.y;
                    arr1[offset] = e.p1().pos().x;
                    arr1[offset + 1] = e.p1().pos().y;
                    arr2[offset] = e.p2().pos().x;
                    arr2[offset + 1] = e.p2().pos().y;
                }

                //resolveConstraints(body);
            }

            //CLInstance.vectorDistance(arr1, arr2, dest2);


            for (RigidBody2D body : bodyBuffer.values())
            {
                offset = 0;
                for (Edge2D edge : body.getEdges())
                {
                    edge.p2().pos().sub(edge.p1().pos(), vectorBuffer1);
                    var length = vectorBuffer1.length();

//                    var x1 = dest2[offset];
//                    var x2 = dest2[offset];

//                    if (!runyet)
//                    {
//                        System.out.println("Diff: x1:" + (x1 - length) + " x2:" + (x2 - length));
//                    }

                    float diff = length - edge.length();
                    vectorBuffer1.normalize();
                    vectorBuffer1.mul(diff * 0.5f);
                    edge.p1().pos().add(vectorBuffer1);
                    edge.p2().pos().sub(vectorBuffer1);
                }
            }
        }

        runyet = true;
    }

    private Map<RigidBody2D, List<RigidBody2D>> checkMap = new LinkedHashMap<>();
    FloatBuffer bA = FloatBuffer.allocate(125_000_000);
    FloatBuffer bB = FloatBuffer.allocate(125_000_000);

    private void tickCollisions()
    {
        long start = System.nanoTime();
        bA.clear();
        bB.clear();

        int runningCount = 0;
        for (RigidBody2D body : bodyBuffer.values())
        {
            keyCache.clear();
            var b = ecs.getComponentFor(body.getEntitiy(), Component.BoundingBox);
            QuadRectangle targetBox = Component.BoundingBox.coerce(b);
            var candidates = new ArrayList<RigidBody2D>();
            quadTree.getElements(candidates, targetBox);
            //System.out.println("dropped: " + (bodyBuffer.size() - candidates.size()));

            List<RigidBody2D> filtered = new ArrayList<>();

            for (RigidBody2D candidate : candidates)
            {
                if (body.equals(candidate))
                {
                    continue;
                }
                if (body.getEntitiy().equals(candidate.getEntitiy()))
                {
                    continue;
                }
                var k1 = body.getEntitiy() + candidate.getEntitiy();
                var k2 = candidate.getEntitiy() + body.getEntitiy();
                if (keyCache.contains(k1) || keyCache.contains(k2))
                {
                    continue;
                }

                int vertexCount = body.getVerts().size() + candidate.getVerts().size();
                int resultCount = body.getVerts().size() * vertexCount;
                resultCount += candidate.getVerts().size() * vertexCount;
                int inputCount = resultCount * 2;
                float[] arr1 = new float[inputCount];
                float[] arr2 = new float[inputCount];
                cl_buffer_offset = 0;
                for (int i = 0; i < body.getVerts().size(); i++)
                {
                    var b_index = (i + 1) == body.getVerts().size()
                        ? 0
                        : i + 1;
                    var va = body.getVerts().get(i);
                    var vb = body.getVerts().get(b_index);

                    vb.pos().sub(va.pos(), vectorBuffer1);
                    vectorBuffer1.perpendicular();
                    vectorBuffer1.normalize();

                    for (int j = 0; j < body.getVerts().size(); j++)
                    {
                        arr1[cl_buffer_offset] = body.getVerts().get(j).pos().x;
                        arr1[cl_buffer_offset + 1] = body.getVerts().get(j).pos().y;
                        arr2[cl_buffer_offset] = vectorBuffer1.x;
                        arr2[cl_buffer_offset + 1] = vectorBuffer1.y;
                        cl_buffer_offset += 2;
                    }

                    for (int j = 0; j < candidate.getVerts().size(); j++)
                    {
                        arr1[cl_buffer_offset] = candidate.getVerts().get(j).pos().x;
                        arr1[cl_buffer_offset + 1] = candidate.getVerts().get(j).pos().y;
                        arr2[cl_buffer_offset] = vectorBuffer1.x;
                        arr2[cl_buffer_offset + 1] = vectorBuffer1.y;
                        cl_buffer_offset += 2;
                    }
                }

                // 2nd
                for (int i = 0; i < candidate.getVerts().size(); i++)
                {
                    var b_index = (i + 1) == candidate.getVerts().size()
                        ? 0
                        : i + 1;
                    var va = candidate.getVerts().get(i);
                    var vb = candidate.getVerts().get(b_index);

                    vb.pos().sub(va.pos(), vectorBuffer1);
                    vectorBuffer1.perpendicular();
                    vectorBuffer1.normalize();

                    for (int j = 0; j < body.getVerts().size(); j++)
                    {
                        arr1[cl_buffer_offset] = body.getVerts().get(j).pos().x;
                        arr1[cl_buffer_offset + 1] = body.getVerts().get(j).pos().y;
                        arr2[cl_buffer_offset] = vectorBuffer1.x;
                        arr2[cl_buffer_offset + 1] = vectorBuffer1.y;
                        cl_buffer_offset += 2;
                    }

                    for (int j = 0; j < candidate.getVerts().size(); j++)
                    {
                        arr1[cl_buffer_offset] = candidate.getVerts().get(j).pos().x;
                        arr1[cl_buffer_offset + 1] = candidate.getVerts().get(j).pos().y;
                        arr2[cl_buffer_offset] = vectorBuffer1.x;
                        arr2[cl_buffer_offset + 1] = vectorBuffer1.y;
                        cl_buffer_offset += 2;
                    }
                }

                //bufferA.add(arr1);
                //bufferB.add(arr2);

                bA.put(arr1);
                bB.put(arr2);

                runningCount += resultCount;
                filtered.add(candidate);
            }
            if (filtered.isEmpty())
            {
                continue;
            }

            checkMap.put(body, filtered);
        }

        long stop = System.nanoTime();

        long t1 = stop - start;
        start = System.nanoTime();

        float[] r = new float[runningCount];

        int outerOffset = 0;

        bA.flip();
        bB.flip();

        CLInstance.vectorDotProduct(bA, bB, FloatBuffer.wrap(r));

        int secondCount = 0;

        stop = System.nanoTime();

        long t2 = stop - start;

        start = System.nanoTime();

        outerOffset = 0;
        for (Map.Entry<RigidBody2D, List<RigidBody2D>> entry : checkMap.entrySet())
        {
            RigidBody2D body = entry.getKey();
            List<RigidBody2D> candidates = entry.getValue();

            for (RigidBody2D candidate : candidates)
            {
                int middleOffset = outerOffset;

                float min_distance = Float.MAX_VALUE;

                var verts1 = body.getVerts();
                var verts2 = candidate.getVerts();

                int vertexCount = body.getVerts().size() + candidate.getVerts().size();
                int resultCount = body.getVerts().size() * vertexCount;
                resultCount += candidate.getVerts().size() * vertexCount;

                int nextIncrease = resultCount;

                secondCount += nextIncrease;

                RigidBody2D vertex_o = null;
                RigidBody2D edge_o = null;
                int edge_indexA = 0;
                int edge_indexB = 0;
                int vert_index = 0;
                boolean invert = false;

                boolean breakOut = false;

                for (int i = 0; i < verts1.size(); i++)
                {
                    var b_index = (i + 1) == verts1.size()
                        ? 0
                        : i + 1;
                    var va = verts1.get(i);
                    var vb = verts1.get(b_index);

                    vb.pos().sub(va.pos(), vectorBuffer1);
                    vectorBuffer1.perpendicular();
                    vectorBuffer1.normalize();

                    var proj_a = projectPolygon2(verts1, vectorBuffer1, r, middleOffset);
                    middleOffset += verts1.size();
                    var proj_b = projectPolygon2(verts2, vectorBuffer1, r, middleOffset);
                    middleOffset += verts2.size();
                    var distance = polygonDistance(proj_a, proj_b);
                    if (distance > 0)
                    {
                        breakOut = true;
                        break;
                    }
                    var abs_distance = Math.abs(distance);
                    if (abs_distance < min_distance)
                    {
                        vertex_o = candidate;
                        invert = true;
                        edge_o = body;
                        min_distance = abs_distance;
                        vectorBuffer2.set(vectorBuffer1);
                        edge_indexA = i;
                        edge_indexB = b_index;
                    }
                }

                if (breakOut)
                {
                    outerOffset += nextIncrease;
                    continue;
                }

                for (int i = 0; i < verts2.size(); i++)
                {
                    var b_index = (i + 1) == verts2.size()
                        ? 0
                        : i + 1;
                    var va = verts2.get(i);
                    var vb = verts2.get(b_index);

                    vb.pos().sub(va.pos(), vectorBuffer1);
                    vectorBuffer1.perpendicular();
                    vectorBuffer1.normalize();

                    var proj_a = projectPolygon2(verts1, vectorBuffer1, r, middleOffset);
                    middleOffset += verts1.size();
                    var proj_b = projectPolygon2(verts2, vectorBuffer1, r, middleOffset);
                    middleOffset += verts2.size();
                    var distance = polygonDistance(proj_a, proj_b);
                    if (distance > 0)
                    {
                        breakOut = true;
                        break;
                    }
                    var abs_distance = Math.abs(distance);
                    if (abs_distance < min_distance)
                    {
                        vertex_o = body;
                        invert = false;
                        edge_o = candidate;
                        min_distance = abs_distance;
                        vectorBuffer2.set(vectorBuffer1);
                        edge_indexA = i;
                        edge_indexB = b_index;
                    }
                }

                if (breakOut)
                {
                    outerOffset += nextIncrease;
                    continue;
                }

                var pr = projectPolygon(vertex_o.getVerts(), vectorBuffer2);
                vert_index = pr.index();

                min_distance = min_distance / vectorBuffer2.length();
                vectorBuffer2.normalize();

                var a = invert
                    ? candidate
                    : body;

                var b = invert
                    ? body
                    : candidate;

                var tA = ecs.getComponentFor(a.getEntitiy(), Component.Transform);
                var tB = ecs.getComponentFor(b.getEntitiy(), Component.Transform);
                Transform transformA = Component.Transform.coerce(tA);
                Transform transformB = Component.Transform.coerce(tB);

                var direction = new Vector2f();
                transformA.position.sub(transformB.position, direction);
                var dirdot = direction.dot(vectorBuffer2);
                if (dirdot < 0)
                {
                    vectorBuffer2.mul(-1);
                }
                direction.set(vectorBuffer2);
                collisionBuffer.add(new CollisionManifold(vertex_o,
                    edge_o, direction, min_distance,
                    edge_indexA, edge_indexB, vert_index));
                outerOffset += nextIncrease;
            }
        }

        stop = System.nanoTime();

        long t3 = stop - start;

        System.out.println("T: \n1=" + t1 + "\n2=" + t2 + "\n3=" + t3);

        if (secondCount != runningCount)
        {
            System.out.println("diff: " + (runningCount - secondCount));
        }
    }


    private static boolean doBoxesIntersect(QuadRectangle a, QuadRectangle b)
    {
        return !(a.max_x < b.min_x ||
            a.max_y < b.min_y ||
            a.min_x > b.max_x ||
            a.min_y > b.max_y);
    }


    private void tickCollisions2()
    {
        //List<float[]> bufferA = new ArrayList<>();
        //List<float[]> bufferB = new ArrayList<>();

        //int runningCount = 0;
        for (RigidBody2D body : bodyBuffer.values())
        {
            keyCache.clear();
            var b = ecs.getComponentFor(body.getEntitiy(), Component.BoundingBox);
            QuadRectangle targetBox = Component.BoundingBox.coerce(b);
            var candidates = new ArrayList<RigidBody2D>();
            quadTree.getElements(candidates, targetBox);

            List<RigidBody2D> filtered = new ArrayList<>();

            for (RigidBody2D candidate : candidates)
            {
                if (body.equals(candidate))
                {
                    continue;
                }
                if (body.getEntitiy().equals(candidate.getEntitiy()))
                {
                    continue;
                }
                var k1 = body.getEntitiy() + candidate.getEntitiy();
                var k2 = candidate.getEntitiy() + body.getEntitiy();
                if (keyCache.contains(k1) || keyCache.contains(k2))
                {
                    continue;
                }

                int vertexCount = body.getVerts().size() + candidate.getVerts().size();
                int resultCount = body.getVerts().size() * vertexCount;
                resultCount += candidate.getVerts().size() * vertexCount;
                int inputCount = resultCount * 2;
                float[] arr1 = new float[inputCount];
                float[] arr2 = new float[inputCount];
                cl_buffer_offset = 0;
                for (int i = 0; i < body.getVerts().size(); i++)
                {
                    var b_index = (i + 1) == body.getVerts().size()
                        ? 0
                        : i + 1;
                    var va = body.getVerts().get(i);
                    var vb = body.getVerts().get(b_index);

                    vb.pos().sub(va.pos(), vectorBuffer1);
                    vectorBuffer1.perpendicular();
                    vectorBuffer1.normalize();

                    for (int j = 0; j < body.getVerts().size(); j++)
                    {
                        arr1[cl_buffer_offset] = body.getVerts().get(j).pos().x;
                        arr1[cl_buffer_offset + 1] = body.getVerts().get(j).pos().y;
                        arr2[cl_buffer_offset] = vectorBuffer1.x;
                        arr2[cl_buffer_offset + 1] = vectorBuffer1.y;
                        cl_buffer_offset += 2;
                    }

                    for (int j = 0; j < candidate.getVerts().size(); j++)
                    {
                        arr1[cl_buffer_offset] = candidate.getVerts().get(j).pos().x;
                        arr1[cl_buffer_offset + 1] = candidate.getVerts().get(j).pos().y;
                        arr2[cl_buffer_offset] = vectorBuffer1.x;
                        arr2[cl_buffer_offset + 1] = vectorBuffer1.y;
                        cl_buffer_offset += 2;
                    }
                }

                // 2nd
                for (int i = 0; i < candidate.getVerts().size(); i++)
                {
                    var b_index = (i + 1) == candidate.getVerts().size()
                        ? 0
                        : i + 1;
                    var va = candidate.getVerts().get(i);
                    var vb = candidate.getVerts().get(b_index);

                    vb.pos().sub(va.pos(), vectorBuffer1);
                    vectorBuffer1.perpendicular();
                    vectorBuffer1.normalize();

                    for (int j = 0; j < body.getVerts().size(); j++)
                    {
                        arr1[cl_buffer_offset] = body.getVerts().get(j).pos().x;
                        arr1[cl_buffer_offset + 1] = body.getVerts().get(j).pos().y;
                        arr2[cl_buffer_offset] = vectorBuffer1.x;
                        arr2[cl_buffer_offset + 1] = vectorBuffer1.y;
                        cl_buffer_offset += 2;
                    }

                    for (int j = 0; j < candidate.getVerts().size(); j++)
                    {
                        arr1[cl_buffer_offset] = candidate.getVerts().get(j).pos().x;
                        arr1[cl_buffer_offset + 1] = candidate.getVerts().get(j).pos().y;
                        arr2[cl_buffer_offset] = vectorBuffer1.x;
                        arr2[cl_buffer_offset + 1] = vectorBuffer1.y;
                        cl_buffer_offset += 2;
                    }
                }

                //bufferA.add(arr1);
                //bufferB.add(arr2);

                float[] a1 = new float[resultCount * 2];
                float[] a2 = new float[resultCount * 2];
                float[] r = new float[resultCount];

                var x = polygonCollision(body, candidate);
                if (x != null)
                {
                    collisionBuffer.add(x);
                }

//                runningCount += resultCount;
                //filtered.add(candidate);
            }
//            if (filtered.isEmpty())
//            {
//                continue;
//            }

            //checkMap.put(body, filtered);
        }

//
//        float[] a1 = new float[runningCount * 2];
//        float[] a2 = new float[runningCount * 2];
//        float[] r = new float[runningCount];
//
//        assert bufferA.size() == bufferB.size() : "error, bad sizes";
//
//        int outerOffset = 0;
//        for (int i = 0; i < bufferA.size(); i++)
//        {
//            float[] cA = bufferA.get(i);
//            float[] cB = bufferB.get(i);
//            System.arraycopy(cA, 0, a1, outerOffset, cA.length);
//            System.arraycopy(cB, 0, a2, outerOffset, cB.length);
//            outerOffset += cA.length;
//        }
//
//        CLInstance.vectorDotProduct(a1, a2, r);
//
//        //System.out.println("lol");
//
//        int secondCount = 0;
//
//        outerOffset = 0;
//        for (Map.Entry<RigidBody2D, List<RigidBody2D>> entry : checkMap.entrySet())
//        {
//            RigidBody2D body = entry.getKey();
//            List<RigidBody2D> candidates = entry.getValue();
//
//            for (RigidBody2D candidate : candidates)
//            {
//                int middleOffset = outerOffset;
//
//                float min_distance = Float.MAX_VALUE;
//
//                var verts1 = body.getVerts();
//                var verts2 = candidate.getVerts();
//
//
//
//                int vertexCount = body.getVerts().size() + candidate.getVerts().size();
//                int resultCount = body.getVerts().size() * vertexCount;
//                resultCount += candidate.getVerts().size() * vertexCount;
//
//                int nextIncrease = resultCount;
//
//                secondCount+=nextIncrease;
//
//                RigidBody2D vertex_o = null;
//                RigidBody2D edge_o = null;
//                int edge_indexA = 0;
//                int edge_indexB = 0;
//                int vert_index = 0;
//                boolean invert = false;
//
//                boolean breakOut = false;
//
//                for (int i = 0; i < verts1.size(); i++)
//                {
//                    var b_index = (i + 1) == verts1.size()
//                        ? 0
//                        : i + 1;
//                    var va = verts1.get(i);
//                    var vb = verts1.get(b_index);
//
//                    vb.pos().sub(va.pos(), vectorBuffer1);
//                    vectorBuffer1.perpendicular();
//                    vectorBuffer1.normalize();
//
//                    var proj_a = projectPolygon2(verts1, vectorBuffer1, r, middleOffset);
//                    middleOffset += verts1.size();
//                    var proj_b = projectPolygon2(verts2, vectorBuffer1, r, middleOffset);
//                    middleOffset += verts2.size();
//                    var distance = polygonDistance(proj_a, proj_b);
//                    if (distance > 0)
//                    {
//                        breakOut = true;
//                        break;
//                    }
//                    var abs_distance = Math.abs(distance);
//                    if (abs_distance < min_distance)
//                    {
//                        vertex_o = candidate;
//                        invert = true;
//                        edge_o = body;
//                        min_distance = abs_distance;
//                        vectorBuffer2.set(vectorBuffer1);
//                        edge_indexA = i;
//                        edge_indexB = b_index;
//                    }
//                }
//
//                if (breakOut)
//                {
//                    outerOffset += nextIncrease;
//                    continue;
//                }
//
//                for (int i = 0; i < verts2.size(); i++)
//                {
//                    var b_index = (i + 1) == verts2.size()
//                        ? 0
//                        : i + 1;
//                    var va = verts2.get(i);
//                    var vb = verts2.get(b_index);
//
//                    vb.pos().sub(va.pos(), vectorBuffer1);
//                    vectorBuffer1.perpendicular();
//                    vectorBuffer1.normalize();
//
//                    var proj_a = projectPolygon2(verts1, vectorBuffer1, r, middleOffset);
//                    middleOffset += verts1.size();
//                    var proj_b = projectPolygon2(verts2, vectorBuffer1, r, middleOffset);
//                    middleOffset += verts2.size();
//                    var distance = polygonDistance(proj_a, proj_b);
//                    if (distance > 0)
//                    {
//                        breakOut = true;
//                        break;
//                    }
//                    var abs_distance = Math.abs(distance);
//                    if (abs_distance < min_distance)
//                    {
//                        vertex_o = body;
//                        invert = false;
//                        edge_o = candidate;
//                        min_distance = abs_distance;
//                        vectorBuffer2.set(vectorBuffer1);
//                        edge_indexA = i;
//                        edge_indexB = b_index;
//                    }
//                }
//
//                if (breakOut)
//                {
//                    outerOffset += nextIncrease;
//                    continue;
//                }
//
//                var pr = projectPolygon(vertex_o.getVerts(), vectorBuffer2);
//                vert_index = pr.index();
//
//                min_distance = min_distance / vectorBuffer2.length();
//                vectorBuffer2.normalize();
//
//                var a = invert
//                    ? candidate
//                    : body;
//
//                var b = invert
//                    ? body
//                    : candidate;
//
//                var tA = ecs.getComponentFor(a.getEntitiy(), Component.Transform);
//                var tB = ecs.getComponentFor(b.getEntitiy(), Component.Transform);
//                Transform transformA = Component.Transform.coerce(tA);
//                Transform transformB = Component.Transform.coerce(tB);
//
//                var direction = new Vector2f();
//                transformA.position.sub(transformB.position, direction);
//                var dirdot = direction.dot(vectorBuffer2);
//                if (dirdot < 0)
//                {
//                    vectorBuffer2.mul(-1);
//                }
//                direction.set(vectorBuffer2);
//                collisionBuffer.add(new CollisionManifold(vertex_o,
//                    edge_o, direction, min_distance,
//                    edge_indexA, edge_indexB, vert_index));
//                outerOffset += nextIncrease;
//            }
//        }
//
//        if (secondCount != runningCount)
//        {
//            System.out.println("diff: " + (runningCount - secondCount));
//        }
    }

    public static class BoxKey
    {
        public final int x;
        public final int y;

        public BoxKey(int x, int y)
        {
            this.x = x;
            this.y = y;
        }

        @Override
        public boolean equals(Object o)
        {
            if (this == o)
            {
                return true;
            }
            if (o == null || getClass() != o.getClass())
            {
                return false;
            }

            BoxKey boxKey = (BoxKey) o;

            if (x != boxKey.x)
            {
                return false;
            }
            return y == boxKey.y;
        }

        @Override
        public int hashCode()
        {
            int result = x;
            result = 31 * result + y;
            return result;
        }
    }

    public class SpatialMap
    {
        private int width = 1920;
        private int height = 1080;
        private int xsubdivisions = 80;
        private int ysubdivisions = 55;
        private float x_spacing = 0;
        private float y_spacing = 0;
        public List<QuadRectangle> rects = new ArrayList<>();

        Map<Integer, Map<Integer, BoxKey>> keyMap = new HashMap<>();
        Map<BoxKey, Set<RigidBody2D>> boxMap = new HashMap<>();

        public SpatialMap()
        {
            init();
        }

        public void init()
        {
            x_spacing = (float)width / (float) xsubdivisions;
            y_spacing = (float)height / (float)ysubdivisions;

            var currentX = 0;
            var currentY = 0;
            for (int i = 0; i < xsubdivisions; i++)
            {
                for (int j = 0; j < ysubdivisions; j++)
                {
                    var k = new BoxKey(i, j);
                    keyMap.computeIfAbsent(i, (_i) -> new HashMap<>()).put(j, k);
                    boxMap.put(k, new HashSet<>());
                    rects.add(new QuadRectangle(currentX, currentY, x_spacing, y_spacing));
                    currentY+=y_spacing;
                }
                currentX+=x_spacing;
                currentY=0;
            }
        }

        public Set<RigidBody2D> getMatches(QuadRectangle box)
        {
            var rSet = new HashSet<RigidBody2D>();
            for (BoxKey k : box.getKeys())
            {
                rSet.addAll(boxMap.get(k));
            }
            return rSet;
        }

        public void add(RigidBody2D body, QuadRectangle box)
        {
            box.resetKeys();
            var t = ecs.getComponentFor(body.getEntitiy(), Component.Transform);
            Transform transform = Component.Transform.coerce(t);
            var k0 = getKeyForPoint(transform.position.x, transform.position.y);
            var k1 = getKeyForPoint(box.x, box.y);
            var k2 = getKeyForPoint(box.x + box.width, box.y);
            var k3 = getKeyForPoint(box.x + box.width, box.y + box.height);
            var k4 = getKeyForPoint(box.x, box.y + box.height);

            if (k0 == null
                && k1 == null
                && k2 == null
                && k3 == null
                && k4 == null)
            {
                return;
            }

            // is within only one cell
            if (k0 == k1 && k0 == k2 && k0 == k3 && k0 == k4)
            {
                boxMap.get(k0).add(body);
                box.addkey(k0);
                return;
            }

            if (k0 != null)
            {
                boxMap.get(k0).add(body);
                box.addkey(k0);
            }

            if (k1 != null)
            {
                boxMap.get(k1).add(body);
                box.addkey(k1);
            }
            if (k2 != null)
            {
                boxMap.get(k2).add(body);
                box.addkey(k2);
            }
            if (k3 != null)
            {
                boxMap.get(k3).add(body);
                box.addkey(k3);
            }
            if (k4 != null)
            {
                boxMap.get(k4).add(body);
                box.addkey(k4);
            }
        }

        private BoxKey getKeyForPoint(float px, float py)
        {
            int index_x = ((int) Math.floor(px / x_spacing));
            int index_y = ((int) Math.floor(py / y_spacing));
            if (!keyMap.containsKey(index_x))
            {
                return null;
            }
            var ymp = keyMap.get(index_x);
            if (!ymp.containsKey(index_y))
            {
                return null;
            }
            return ymp.get(index_y);
        }
    }

    private void tickSimulation(float dt)
    {
//        setThreadvectorBuffers();
        quadTree.clear();

        var spaceMap = new SpatialMap();

        var bodies = ecs.getComponents(Component.RigidBody2D);
        if (bodies == null || bodies.isEmpty())
        {
            return;
        }

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
            updateBoundBox(body2D, box);
            //quadTree.insert(box, body2D);

            spaceMap.add(body2D, box);

            bodyBuffer.put(entity, body2D);
        }

        collisionBuffer.clear();
        tickEdges();
        //tickCollisions();
        for (RigidBody2D body2D : bodyBuffer.values())
        {
            findCollisions(body2D, quadTree, spaceMap);
        }

        for (CollisionManifold c : collisionBuffer)
        {
            reactPolygon(c);
        }

        //tickEdges();

        //Window.setQT(quadTree);
        Window.setSP(spaceMap);


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
