package com.controllerface.bvge.ecs.systems.physics;

import com.controllerface.bvge.Main;
import com.controllerface.bvge.cl.OpenCL;
import com.controllerface.bvge.cl.OpenCL_EX;
import com.controllerface.bvge.data.FBody2D;
import com.controllerface.bvge.data.FEdge2D;
import com.controllerface.bvge.data.FPoint2D;
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
    private final float TICK_RATE = 1.0f / 60.0f;
    private final int SUB_STEPS = 1;
    private final int EDGE_STEPS = 4;
    private final float GRAVITY = 9.8f;
    private final float FRICTION = .980f;
    private float accumulator = 0.0f;

    private SpatialMap spatialMap = new SpatialMap();


    /**
     * These buffers are reused each tick p2 avoid creating a new one every frame and for each object.
     * They should always be zeroed before each use.
     */
    private final Map<String, FBody2D> bodyBuffer = new HashMap<>();
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
        Transform transform = Component.Transform.coerce(t);

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
        MathEX.centroid(body2D.points(), transform.position);
        body2D.setPos(transform.position);
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
            for (FBody2D body : bodyBuffer.values())
            {
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

    private record CheckHit(FBody2D body2D, List<FBody2D> candidates)
    {
    }

    //region WORKING AREA

    // this is the CL enabled variant of the collision check todo: explore this more
    // a linked map is used now p2 ensure order is preserved, but this is expensive
    //private Map<RigidBody2D, List<RigidBody2D>> checkMap = new LinkedHashMap<>();
    private List<CheckHit> checkList = new ArrayList<>();


    FloatBuffer v_dot_bufferA = FloatBuffer.allocate(125_000_000);
    FloatBuffer v_dot_bufferB = FloatBuffer.allocate(125_000_000);

    FloatBuffer v_norm_buffer = FloatBuffer.allocate(125_000_000);

    private void tickCollisions()
    {
        //checkMap.clear();
        checkList.clear();

        v_dot_bufferA.clear();
        v_dot_bufferB.clear();
        v_norm_buffer.clear();

        int runningVDotCount = 0;
        int runningNormCount = 0;



        // broad phase collision check p2 filter in worthwhile collision checks
        //Map<RigidBody2D, List<RigidBody2D>> broadPhaseHits = new HashMap<>();
        List<CheckHit> broadPhaseHits = new ArrayList<>();

        for (FBody2D body : bodyBuffer.values())
        {
            var b = ecs.getComponentFor(body.entity(), Component.BoundingBox);
            QuadRectangle targetBox = Component.BoundingBox.coerce(b);
            var candidates = spatialMap.getMatches(targetBox);
            List<FBody2D> filtered = new ArrayList<>();

            for (FBody2D candidate : candidates)
            {
                if (body.equals(candidate))
                {
                    continue;
                }
                if (body.entity().equals(candidate.entity()))
                {
                    continue;
                }

                var bx = ecs.getComponentFor(candidate.entity(), Component.BoundingBox);
                QuadRectangle candidateBox = Component.BoundingBox.coerce(bx);
                boolean ch = doBoxesIntersect(targetBox, candidateBox);
                if (!ch)
                {
                    continue;
                }

                filtered.add(candidate);
            }
            if (filtered.isEmpty())
            {
                continue;
            }

            //broadPhaseHits.put(body, filtered);
            broadPhaseHits.add(new CheckHit(body, filtered));
        }


        // new method starts here
        //Map<RigidBody2D, List<RigidBody2D>> normalStage = new LinkedHashMap<>();
        List<CheckHit> normalStage = new ArrayList<>();
        for (CheckHit e : broadPhaseHits)
        //for (Map.Entry<RigidBody2D, List<RigidBody2D>> e : broadPhaseHits.entrySet())
        {
            FBody2D body = e.body2D();
            List<FBody2D> candidates = e.candidates();

            for (FBody2D candidate : candidates)
            {
                var test_buffer_offset = 0;
                int inputCount = (body.points().length * 2) + (candidate.points().length * 2);
                //inputCount *= 2;
                float[] arr1 = new float[inputCount];


                // first object
                for (int i = 0; i < body.points().length; i++)
                {
                    var b_index = (i + 1) == body.points().length
                        ? 0
                        : i + 1;
                    var va = body.points()[i];
                    var vb = body.points()[b_index];

                    int xOffset = test_buffer_offset;
                    int yOffset = test_buffer_offset + 1;

                    vb.subPos(va, vectorBuffer1);
                    //vb.pos().sub(va.pos(), vectorBuffer1);
                    vectorBuffer1.perpendicular();
                    arr1[xOffset] = vectorBuffer1.x;
                    arr1[yOffset] = vectorBuffer1.y;
                    test_buffer_offset += 2;
                }

                // second object
                for (int i = 0; i < candidate.points().length; i++)
                {
                    var b_index = (i + 1) == candidate.points().length
                        ? 0
                        : i + 1;
                    var va = candidate.points()[i];
                    var vb = candidate.points()[b_index];

                    int xOffset = test_buffer_offset;
                    int yOffset = test_buffer_offset + 1;

                    vb.subPos(va, vectorBuffer1);
                    //vb.pos().sub(va.pos(), vectorBuffer1);
                    vectorBuffer1.perpendicular();
                    arr1[xOffset] = vectorBuffer1.x;
                    arr1[yOffset] = vectorBuffer1.y;
                    test_buffer_offset += 2;
                }

                // add all of the vector data p2 the global buffers
                v_norm_buffer.put(arr1);
                runningNormCount += inputCount;
            }
            //normalStage.put(body, candidates);
            normalStage.add(new CheckHit(body, candidates));
        }


        v_norm_buffer.flip();
        float[] v_norm_results = new float[runningNormCount];
        OpenCL.vectorNormalize(v_norm_buffer, FloatBuffer.wrap(v_norm_results));


        int result2Offset = 0;
        // next new method here
        for (CheckHit e : normalStage)
        //for (Map.Entry<RigidBody2D, List<RigidBody2D>> e : normalStage.entrySet())
        {
            FBody2D body = e.body2D();
            List<FBody2D> candidates = e.candidates();

            for (FBody2D candidate : candidates)
            {
                int vertexCount = body.points().length + candidate.points().length;
                int resultCount = body.points().length * vertexCount;
                resultCount += candidate.points().length * vertexCount;
                int inputCount = resultCount * 2;
                float[] arr1 = new float[inputCount];
                float[] arr2 = new float[inputCount];

                var local_buffer_offset = 0;

                // first object
                for (int i = 0; i < body.points().length; i++)
                {
                    var x = v_norm_results[result2Offset];
                    var y = v_norm_results[result2Offset + 1];

                    result2Offset += 2;

                    for (int j = 0; j < body.points().length; j++)
                    {
                        int xOffset = local_buffer_offset;
                        int yOffset = local_buffer_offset + 1;

                        arr1[xOffset] = body.points()[j].pos_x();
                        arr1[yOffset] = body.points()[j].pos_y();
                        arr2[xOffset] = x;
                        arr2[yOffset] = y;

                        local_buffer_offset += 2;
                    }

                    for (int j = 0; j < candidate.points().length; j++)
                    {
                        arr1[local_buffer_offset] = candidate.points()[j].pos_x();
                        arr1[local_buffer_offset + 1] = candidate.points()[j].pos_y();
                        arr2[local_buffer_offset] = x;
                        arr2[local_buffer_offset + 1] = y;
                        local_buffer_offset += 2;
                    }
                }

                // second object
                for (int i = 0; i < candidate.points().length; i++)
                {

                    var x = v_norm_results[result2Offset];
                    var y = v_norm_results[result2Offset + 1];
                    result2Offset += 2;
                    for (int j = 0; j < body.points().length; j++)
                    {
                        arr1[local_buffer_offset] = body.points()[j].pos_x();
                        arr1[local_buffer_offset + 1] = body.points()[j].pos_y();
                        arr2[local_buffer_offset] = x;
                        arr2[local_buffer_offset + 1] = y;
                        local_buffer_offset += 2;
                    }

                    for (int j = 0; j < candidate.points().length; j++)
                    {
                        arr1[local_buffer_offset] = candidate.points()[j].pos_x();
                        arr1[local_buffer_offset + 1] = candidate.points()[j].pos_y();
                        arr2[local_buffer_offset] = x;
                        arr2[local_buffer_offset + 1] = y;
                        local_buffer_offset += 2;
                    }
                }

                // add all of the vector data p2 the global buffers
                v_dot_bufferA.put(arr1);
                v_dot_bufferB.put(arr2);

                runningVDotCount += resultCount;
            }
            checkList.add(new CheckHit(body, candidates));
            //checkMap.put(body, candidates);
        }








        // todo: refactor p2 use pre-computed normals
//        for (Map.Entry<RigidBody2D, List<RigidBody2D>> e : innerMap.entrySet())
//        {
//            RigidBody2D body = e.getKey();
//            List<RigidBody2D> candidates = e.getValue();
//
//            for (RigidBody2D candidate : candidates)
//            {
//                int vertexCount = body.getVerts().size() + candidate.getVerts().size();
//                int resultCount = body.getVerts().size() * vertexCount;
//                resultCount += candidate.getVerts().size() * vertexCount;
//                int inputCount = resultCount * 2;
//                float[] arr1 = new float[inputCount];
//                float[] arr2 = new float[inputCount];
//
//                var local_buffer_offset = 0;
//
//                // first object
//                for (int i = 0; i < body.getVerts().size(); i++)
//                {
//                    var b_index = (i + 1) == body.getVerts().size()
//                        ? 0
//                        : i + 1;
//                    var va = body.getVerts().get(i);
//                    var vb = body.getVerts().get(b_index);
//
//                    // todo: parallelize these steps as a first stage
//                    vb.pos().sub(va.pos(), vectorBuffer1);
//                    vectorBuffer1.perpendicular();
//                    vectorBuffer1.normalize();
//
//                    for (int j = 0; j < body.getVerts().size(); j++)
//                    {
//                        int xOffset = local_buffer_offset;
//                        int yOffset = local_buffer_offset + 1;
//
//                        arr1[xOffset] = body.getVerts().get(j).pos().x;
//                        arr1[yOffset] = body.getVerts().get(j).pos().y;
//                        arr2[xOffset] = vectorBuffer1.x;
//                        arr2[yOffset] = vectorBuffer1.y;
//
//                        local_buffer_offset += 2;
//                    }
//
//                    for (int j = 0; j < candidate.getVerts().size(); j++)
//                    {
//                        arr1[local_buffer_offset] = candidate.getVerts().get(j).pos().x;
//                        arr1[local_buffer_offset + 1] = candidate.getVerts().get(j).pos().y;
//                        arr2[local_buffer_offset] = vectorBuffer1.x;
//                        arr2[local_buffer_offset + 1] = vectorBuffer1.y;
//                        local_buffer_offset += 2;
//                    }
//                }
//
//                // second object
//                for (int i = 0; i < candidate.getVerts().size(); i++)
//                {
//                    var b_index = (i + 1) == candidate.getVerts().size()
//                        ? 0
//                        : i + 1;
//                    var va = candidate.getVerts().get(i);
//                    var vb = candidate.getVerts().get(b_index);
//
//                    vb.pos().sub(va.pos(), vectorBuffer1);
//                    vectorBuffer1.perpendicular();
//                    vectorBuffer1.normalize();
//
//                    for (int j = 0; j < body.getVerts().size(); j++)
//                    {
//                        arr1[local_buffer_offset] = body.getVerts().get(j).pos().x;
//                        arr1[local_buffer_offset + 1] = body.getVerts().get(j).pos().y;
//                        arr2[local_buffer_offset] = vectorBuffer1.x;
//                        arr2[local_buffer_offset + 1] = vectorBuffer1.y;
//                        local_buffer_offset += 2;
//                    }
//
//                    for (int j = 0; j < candidate.getVerts().size(); j++)
//                    {
//                        arr1[local_buffer_offset] = candidate.getVerts().get(j).pos().x;
//                        arr1[local_buffer_offset + 1] = candidate.getVerts().get(j).pos().y;
//                        arr2[local_buffer_offset] = vectorBuffer1.x;
//                        arr2[local_buffer_offset + 1] = vectorBuffer1.y;
//                        local_buffer_offset += 2;
//                    }
//                }
//
//                // add all of the vector data p2 the global buffers
//                v_dot_bufferA.put(arr1);
//                v_dot_bufferB.put(arr2);
//
//                runningVDotCount += resultCount;
//            }
//            checkMap.put(body, candidates);
//        }

        // Open CL: flip the buffers and create the result array with the matching size
        // for the dot product results.
        v_dot_bufferA.flip();
        v_dot_bufferB.flip();
        float[] v_dot_results = new float[runningVDotCount];
        OpenCL.vectorDotProduct(v_dot_bufferA, v_dot_bufferB, FloatBuffer.wrap(v_dot_results));

        int secondCount = 0;

        // this offset maps p2 the pre-computed results array, as the map entries
        // are traversed, it is updated p2 point p2 the next offset into the array
        int outerOffset = 0;

        for (CheckHit entry : checkList)
        //for (Map.Entry<RigidBody2D, List<RigidBody2D>> entry : checkMap.entrySet())
        {
            FBody2D body = entry.body2D();
            List<FBody2D> candidates = entry.candidates();

            for (FBody2D candidate : candidates)
            {
                int middleOffset = outerOffset;

                float min_distance = Float.MAX_VALUE;

                var verts1 = body.points();
                var verts2 = candidate.points();

                int vertexCount = body.points().length + candidate.points().length;
                int resultCount = body.points().length * vertexCount;
                resultCount += candidate.points().length * vertexCount;

                int nextIncrease = resultCount;

                secondCount += nextIncrease;

                FBody2D vertex_o = null;
                FBody2D edge_o = null;
                int edge_indexA = 0;
                int edge_indexB = 0;
                int vert_index = 0;
                boolean invert = false;

                boolean breakOut = false;

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

                    var proj_a = cl_projectPolygon(verts1, v_dot_results, middleOffset);
                    middleOffset += verts1.length;
                    var proj_b = cl_projectPolygon(verts2, v_dot_results, middleOffset);
                    middleOffset += verts2.length;
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

                    var proj_a = cl_projectPolygon(verts1, v_dot_results, middleOffset);
                    middleOffset += verts1.length;
                    var proj_b = cl_projectPolygon(verts2, v_dot_results, middleOffset);
                    middleOffset += verts2.length;
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

                var pr = projectPolygon(vertex_o.points(), vectorBuffer2);
                vert_index = pr.index();

                min_distance = min_distance / vectorBuffer2.length();
                vectorBuffer2.normalize();

                var a = invert
                    ? candidate
                    : body;

                var b = invert
                    ? body
                    : candidate;

                var tA = ecs.getComponentFor(a.entity(), Component.Transform);
                var tB = ecs.getComponentFor(b.entity(), Component.Transform);
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

        if (secondCount != runningVDotCount)
        {
            System.out.println("diff: " + (runningVDotCount - secondCount));
        }
    }

    //endregion

    private static boolean doBoxesIntersect(QuadRectangle a, QuadRectangle b)
    {
        return !(a.max_x < b.min_x ||
            a.max_y < b.min_y ||
            a.min_x > b.max_x ||
            a.min_y > b.max_y);
    }

    //private CountDownLatch collLatch;

    //private ExecutorService collThreads = Executors.newFixedThreadPool(2);

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

        bodyBuffer.clear();
        OpenCL_EX.integrate(Main.Memory.body_buffer, Main.Memory.point_buffer, dt);
        for (Map.Entry<String, GameComponent> entry : bodies.entrySet())
        {
            String entity = entry.getKey();
            var b = ecs.getComponentFor(entity, Component.BoundingBox);
            QuadRectangle box = Component.BoundingBox.coerce(b);
            GameComponent component = entry.getValue();
            FBody2D body2D = Component.RigidBody2D.coerce(component);
            var t = ecs.getComponentFor(entity, Component.Transform);
            Transform transform = Component.Transform.coerce(t);
            transform.position.x = body2D.pos_x();
            transform.position.y = body2D.pos_y();
            box.setX(body2D.bounds_x());
            box.setY(body2D.bounds_y());
            box.setWidth(body2D.bounds_w());
            box.setHeight(body2D.bounds_h());

            //resolveForces(entity, body2D);
            //integrate(entity, body2D, dt);
            updateBoundBox(body2D, box);

            spatialMap.add(body2D, box);
            bodyBuffer.put(entity, body2D);
        }

        collisionProgress.clear();
        collisionBuffer.clear();

        //tickCollisions();

        for (FBody2D body2D : bodyBuffer.values())
        {
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
