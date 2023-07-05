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

        // update the player's acc values, don't bother doing this in the CL call
        // as it only applies to one object and would waste cycles on all other objects
        var bd = ecs.getComponentFor("player", Component.RigidBody2D);
        FBody2D body = Component.RigidBody2D.coerce(bd);
        resolveForces(body.entity(), body);

        // integrate in CL
        OCLFunctions.integrate(dt,
                spatialPartition.getX_spacing(),
                spatialPartition.getY_spacing(),
                spatialPartition.getX_origin(),
                spatialPartition.getY_origin(),
                spatialPartition.getWidth(),
                spatialPartition.getHeight(),
                spatialPartition.getX_subdivisions(),
                spatialPartition.getY_subdivisions());

        spatialPartition.calculateKeyBankSize();
        spatialPartition.buildKeyBank();
        spatialPartition.calculateMapOffsets();
        spatialPartition.buildKeyMap();

        // todo: now need to pull out the intermediate list used as a buffer in this method
        //  may need to do two passes, one to detect size needed and one to actually get candidates
        //  best to do on GPU.
        var candidates = spatialPartition.computeCandidatesEX();

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
