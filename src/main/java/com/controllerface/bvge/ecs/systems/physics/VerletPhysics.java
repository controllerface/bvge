package com.controllerface.bvge.ecs.systems.physics;

import com.controllerface.bvge.Main;
import com.controllerface.bvge.cl.OpenCL_EX;
import com.controllerface.bvge.data.*;
import com.controllerface.bvge.ecs.ECS;
import com.controllerface.bvge.ecs.components.*;
import com.controllerface.bvge.ecs.systems.GameSystem;
import org.joml.Vector2f;

import java.nio.FloatBuffer;
import java.util.*;

public class VerletPhysics extends GameSystem
{
    private final float TICK_RATE = 1.0f / 60.0f;
    private final int SUB_STEPS = 1;
    private final int EDGE_STEPS = 1;
    private final float GRAVITY = 9.8f;
    private final float FRICTION = .995f;
    private float accumulator = 0.0f;


    /**
     * These buffers are reused each tick p2 avoid creating a new one every frame and for each object.
     * They should always be zeroed before each use.
     */
    private final List<CollisionManifold> collisionBuffer = new ArrayList<>();
    private final Map<String, Set<String>> collisionProgress = new HashMap<>();
    private final Vector2f vectorBuffer1 = new Vector2f();
    private final Vector2f vectorBuffer2 = new Vector2f();

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

    private float edgeContact(FPoint2D e1, FPoint2D e2, Vector2f collision_vertex, Vector2f collision_vector)
    {
        float contact;
        float x_dist = e1.pos_x() - e2.pos_x();
        float y_dist = e1.pos_y() - e2.pos_y();
        if (Math.abs(x_dist) > Math.abs(y_dist))
        {
            float x_offset = (collision_vertex.x() - collision_vector.x - e1.pos_x());
            float x_diff = (e2.pos_x() - e1.pos_x());
            contact = x_offset / x_diff;
        }
        else
        {
            float y_offset = (collision_vertex.y() - collision_vector.y - e1.pos_y());
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
        var x_index = collision.vert() * Main.Memory.Width.POINT;
        var y_index = x_index + 1;
        var collision_vertex_x = Main.Memory.point_buffer[x_index];
        var collision_vertex_y = Main.Memory.point_buffer[y_index];

        //System.out.println("DEBUG V X: " + collision_vertex_x + " Y: " + collision_vertex_y);

        // vertex object
        if (vertex_magnitude > 0)
        {
            collision_vector.mul(vertex_magnitude, vectorBuffer1);
            Main.Memory.point_buffer[x_index] += vectorBuffer1.x;
            Main.Memory.point_buffer[y_index] += vectorBuffer1.y;
        }

        // edge object
        if (edge_magnitude > 0)
        {
            var edge_verts = collision.edgeObject().points();
            var e1 = edge_verts[collision.edgeA()];
            var e2 = edge_verts[collision.edgeB()];
            vectorBuffer1.x = collision_vertex_x;
            vectorBuffer1.y = collision_vertex_y;
            var edge_contact = edgeContact(e1, e2, vectorBuffer1, collision_vector);
            //System.out.println("DEBUG E1 X: " + e1.pos_x() + " Y: " + e1.pos_y());
            //System.out.println("DEBUG E2 X: " + e2.pos_x() + " Y: " + e2.pos_y());

            float edge_scale = 1.0f / (edge_contact * edge_contact + (1 - edge_contact) * (1 - edge_contact));
            collision_vector.mul((1 - edge_contact) * edge_magnitude * edge_scale, vectorBuffer1);
            collision_vector.mul(edge_contact * edge_magnitude * edge_scale, vectorBuffer2);
            e1.subPos(vectorBuffer1);
            e2.subPos(vectorBuffer2);
        }
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

    private SpatialMapEX spatialMap = new SpatialMapEX();

    private void tickSimulation(float dt)
    {
        var bodies = ecs.getComponents(Component.RigidBody2D);
        if (bodies == null || bodies.isEmpty())
        {
            return;
        }

        var bd = ecs.getComponentFor("player", Component.RigidBody2D);
        FBody2D body = Component.RigidBody2D.coerce(bd);
        resolveForces(body.entity(), body);

        // integrate
        OpenCL_EX.integrate(dt);

        // broad phase collision
        // todo: split this up into two phases, one that calculates the space needed
        //  for each object, storing it in the object's structure, then a quick pass
        //  locally to create the appropriately sized buffer, then push that back up
        //  to be calculated on the GPU. Essentially, only return to the CPU when we
        //  need to dynamically generate a sized buffer.
        var key_directory = spatialMap.rebuildIndex();
        var candidates = spatialMap.computeCandidates(key_directory);

        // narrow phase collision
        if (candidates.limit() > 0)
        {
            var count = candidates.limit() / Main.Memory.Width.COLLISION;
            var manifold_size = count * Main.Memory.Width.MANIFOLD;
            var manifolds = new float[manifold_size];
            var manifoldBuffer = FloatBuffer.wrap(manifolds);
            OpenCL_EX.collide(candidates, manifoldBuffer);

            var reaction_size = count * Main.Memory.Width.REACTION;
            var reactions = new float[reaction_size];
            var reactionBuffer = FloatBuffer.wrap(reactions);
            OpenCL_EX.react(manifoldBuffer, reactionBuffer);

            // todo: replace loop below with CL call
            for (int i = 0; i < count; i++)
            {
                var next = i * Main.Memory.Width.MANIFOLD;
                if (manifolds[next] == -1) // separating axis was found
                {
                    continue;
                }

                float[] manifold = new float[Main.Memory.Width.MANIFOLD];
                manifold[0] = manifolds[next];
                manifold[1] = manifolds[next + 1];
                manifold[2] = manifolds[next + 2];
                manifold[3] = manifolds[next + 3];
                manifold[4] = manifolds[next + 4];
                manifold[5] = manifolds[next + 5];
                manifold[6] = manifolds[next + 6];
                manifold[7] = manifolds[next + 7];
                var vo = Main.Memory.bodyByIndex((int)manifold[0]);
                var eo = Main.Memory.bodyByIndex((int)manifold[1]);
                var norm = new Vector2f(manifold[2], manifold[3]);
                var _manifold = new CollisionManifold(vo, eo, norm, manifold[4], (int)manifold[5], (int)manifold[6],
                    (int)manifold[7]);
                reactPolygon(_manifold); // todo: cut out the intermediate manifold object
            }
            //System.out.println("----");
        }

        tickEdges();
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
