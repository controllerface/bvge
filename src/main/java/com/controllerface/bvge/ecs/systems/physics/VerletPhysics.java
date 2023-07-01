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
    private final Vector2f vectorBuffer1 = new Vector2f();

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
    }

    private void reactPolygon(float[] reaction)
    {
        // vertex object
        var x_index = (int)reaction[7] * Main.Memory.Width.POINT;
        var y_index = x_index + 1;
        Main.Memory.point_buffer[x_index] += reaction[8];
        Main.Memory.point_buffer[y_index] += reaction[9];

        // edge object
        var x_index_a = (int)reaction[5] * Main.Memory.Width.POINT;
        var y_index_a = x_index_a + 1;
        var x_index_b = (int)reaction[6] * Main.Memory.Width.POINT;
        var y_index_b = x_index_b + 1;
        Main.Memory.point_buffer[x_index_a] -= reaction[10];
        Main.Memory.point_buffer[y_index_a] -= reaction[11];
        Main.Memory.point_buffer[x_index_b] -= reaction[12];
        Main.Memory.point_buffer[y_index_b] -= reaction[13];
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

        // if somehow there are no bodies, just bail. something is probably really wrong
        if (bodies == null || bodies.isEmpty())
        {
            return;
        }

        // update the player's acc values, don't bother doing this in the CL call
        // as it only applies to one object and would waste cycles on all other objects
        var bd = ecs.getComponentFor("player", Component.RigidBody2D);
        FBody2D body = Component.RigidBody2D.coerce(bd);
        resolveForces(body.entity(), body);

        // integrate in CL
        OpenCL_EX.integrate(dt);

        // broad phase collision

        // NEW way (WIP)
        // todo: split some of these into two phases, one that calculates the space needed
        //  for a phase, then a quick pass locally to create the appropriately sized buffer,
        //  then push that back up to be calculated on the GPU. Essentially, only return to
        //  the CPU when we need to generate a dynamically sized buffer.

        // 1: calculate needed size and offset for each body from the si_key_bank_size values.
        // the offsets are stored within the body's associated bounds object
        var keyBankSize = spatialMap.calculateKeyBankSize();

        // 2: create a buffer of the required size, this takes the place of the local key buffer,
        // and will contain the keys for each object. the layout should be identical as well
        int[] keyBank = new int[keyBankSize];

        // 3: fill the key buffer using the pre-computed sizes and offsets from previous steps.
        // after this method completes, the key buffer will contain all the keys for each object.
        // The returned object contains two important things:
        // 1. a fixed size array that matches
        // the size of the key directory and contains the total count of keys (among all objects)
        // that is stored for each cell index.
        // 2. a count of the total number of keys that were stored in the key bank buffer. This
        // can be used to size the final array that should replace the hashmap
        SpatialMapEX.IndexEx mapCounts = spatialMap.rebuildKeyBank(keyBank);

        // store in a variable for easier debugging/visibility for implementation
        int[] keyMapCounts = mapCounts.mapCounts();

        // 4: in addition to the counts, we also need to know the offset into the key map buffer
        // where the body indices for each key will be stored. Together with the map counts,
        // these structures will provide a means to determine how many bodies are a match for a given
        // key, replicating the functionality of a HashMap but with a fixed size array
        int[] keyMapOffsets = spatialMap.calculateMapOffsets(mapCounts);

        // 5: create a buffer that will contain the raw data that will take the place of the local pointer
        // buffer, it will hold the computed mapping data instead, allowing removal of the Java Map object
        int[] keyMapBuffer = new int[mapCounts.keyTotal()];

        // 6: todo: figure out how to get the key matches into the map buffer matching the current pointer
        //     buffer functionality (updateKeyDirectory)



        // OLD way

        // todo: this only sets up the map for the next step now,
        spatialMap.rebuildIndex();

        // this should be replaced by the key map structure
        spatialMap.updateKeyDirectory();

        var candidates = spatialMap.computeCandidates(spatialMap.keyDirectory(), keyBank);

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

            // todo: replace loop below with CL call. will need to calculate offsets for
            //  each collision pair and store the reactions, then sum them into a single
            //  reaction, before finally adding them to the appropriate objects
            for (int i = 0; i < count; i++)
            {
                var next_reaction = i * Main.Memory.Width.REACTION;
                if (reactions[next_reaction] == -1) // separating axis was found
                {
                    continue;
                }
                float[] reaction = new float[Main.Memory.Width.REACTION];
                System.arraycopy(reactions, next_reaction, reaction, 0, Main.Memory.Width.REACTION);
                reactPolygon(reaction);
            }
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
