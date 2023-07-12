package com.controllerface.bvge.ecs.systems.physics;

import com.controllerface.bvge.Main;
import com.controllerface.bvge.cl.OCLFunctions;
import com.controllerface.bvge.data.*;
import com.controllerface.bvge.ecs.ECS;
import com.controllerface.bvge.ecs.components.*;
import com.controllerface.bvge.ecs.systems.GameSystem;
import org.joml.Vector2f;

import java.nio.FloatBuffer;
import java.util.*;

import static com.controllerface.bvge.data.PhysicsObjects.FLAG_STATIC;

public class VerletPhysics extends GameSystem
{
    private final float TICK_RATE = 1.0f / 60.0f;
    private final int SUB_STEPS = 1;
    private final int EDGE_STEPS = 2;
    private final float GRAVITY = 9.8f;
    private final float FRICTION = .995f;
    private float accumulator = 0.0f;

    private final SpatialPartition spatialPartition;
    /**
     * These buffers are reused each tick p2 avoid creating a new one every frame and for each object.
     * They should always be zeroed before each use.
     */
    private final Vector2f vectorBuffer1 = new Vector2f();

    public VerletPhysics(ECS ecs, SpatialPartition spatialPartition)
    {
        super(ecs);
        this.spatialPartition = spatialPartition;
    }

    private void resolveForces(String entity, FBody2D body2D)
    {
        if ((body2D.flags() & FLAG_STATIC) != 0) return;
        vectorBuffer1.zero();

        var cp = ecs.getComponentFor(entity, Component.ControlPoints);
        if (cp != null)
        {
            ControlPoints controlPoints = Component.ControlPoints.coerce(cp);

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
        }
        vectorBuffer1.y -= 9.8 * 100;
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

        // uncomment these lines to force inelastic collisions
//        Main.Memory.point_buffer[x_index+2] = Main.Memory.point_buffer[x_index];
//        Main.Memory.point_buffer[y_index+2] = Main.Memory.point_buffer[y_index];
//        Main.Memory.point_buffer[x_index_a+2] = Main.Memory.point_buffer[x_index_a];
//        Main.Memory.point_buffer[y_index_a+2] = Main.Memory.point_buffer[y_index_a];
//        Main.Memory.point_buffer[x_index_b+2] = Main.Memory.point_buffer[x_index_b];
//        Main.Memory.point_buffer[y_index_b+2] = Main.Memory.point_buffer[y_index_b];
    }


    private void tickEdges()
    {
        var bodies = ecs.getComponents(Component.RigidBody2D);
        for (int i = 0; i < EDGE_STEPS; i++)
        {
            // todo: would be worth trying to iterate on each edge but in parallel for each body
            //  solving ALL edges at once doesn't work, but in series it may
            for (Map.Entry<String, GameComponent> entry : bodies.entrySet())
            {
                FBody2D body = Component.RigidBody2D.coerce(entry.getValue());
                for (FEdge2D edge : body.edges())
                {
                    edge.p2().subPos(edge.p1(), vectorBuffer1);
                    var length = vectorBuffer1.length();
                    float diff = length - edge.length();
                    vectorBuffer1.normalize();
                    vectorBuffer1.mul(diff * 0.5f);
                    edge.p1().addPos(vectorBuffer1);
                    edge.p2().subPos(vectorBuffer1);
                }
            }
        }
    }


    private void tickSimulation(float dt)
    {
        var bodies = ecs.getComponents(Component.RigidBody2D);

        // if somehow there are no bodies, just bail. something is probably really wrong
        if (bodies == null || bodies.isEmpty())
        {
            return;
        }

        // todo: this can be moved into CL
        bodies.forEach((key, value)->
        {
            FBody2D body = Component.RigidBody2D.coerce(value);
            resolveForces(body.entity(), body);
        });

        // perform integration step
        OCLFunctions.integrate(dt, spatialPartition);

        // todo #0: replace this with 3 OCL calls
        spatialPartition.calculateKeyBankSize();
        // todo #1: create OCL function to scan the bounding boxes, using the si bank
        //  size as the value to sum. These values are stored in an array aligned to
        //  the number of bounding boxes

        // todo #2: create OCL function to take scan sums from step 1 and forward
        //  them into the corresponding bounding boxes' offset values

        // todo #3: create OCL function to reduce the body key sizes, calculating the
        //  space needed for the key bank and key map. These arrays can then be generated
        //  via a host call.


        // todo #0: replace this with a single OCL call
        spatialPartition.buildKeyBank();
        // todo #1: create an OCL function that accepts the key counts array and the empty,
        //  key map, and operates on every tracked body/bounds pair. The function should
        //  iterate the keys for each object and place them in the appropriate keymap
        //  location. During this process, it should atomically increment the key count
        //  array.


        // todo #0: replace this with 2 OCL calls
        spatialPartition.calculateMapOffsets();
        // todo #1: do an exclusive scan on the key counts array as one CL call, and then
        //  make a second call that forwards the values to the key offsets array


        // todo #0: replace this with 1 OCL call
        spatialPartition.buildKeyMap();
        // todo #1: this one will be tricky, as there needs to be one atomic counter per
        //  key-map index that threads can increment to get their relative offset in order
        //  to store their index into the proper part of the array. When run sequentially,
        //  the order is effectively always ascending and preserved, but this is not strictly
        //  needed.



        // todo #0: replace this with 2 OCL calls
        var candidates = spatialPartition.computeCandidatesEX();
        // todo #1: need to pull out the intermediate list used as a buffer in this method
        //  and instead do two CL passes, one to detect size needed and one to actually get
        //  candidate pairs stored into the sized array. probably will need a global atomic
        //  counter for this one, otherwise a more complex solution will be needed.

        // narrow phase collision
        if (candidates.limit() > 0)
        {
            var count = candidates.limit() / Main.Memory.Width.COLLISION;
            var reaction_size = count * Main.Memory.Width.REACTION;
            var reactions = new float[reaction_size];
            var reaction_buffer = FloatBuffer.wrap(reactions);
            OCLFunctions.collide(candidates, reaction_buffer);

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
