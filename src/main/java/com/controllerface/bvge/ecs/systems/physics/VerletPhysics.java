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
    private final float TARGET_FPS = 60.0f;
    private final float TICK_RATE = 1.0f / TARGET_FPS;
    private final int SUB_STEPS = 1;
    private final int EDGE_STEPS = 2;
    private final float GRAVITY_X = 0;
    private final float GRAVITY_Y = 0;//-(9.8f * 100);
    private final float FRICTION = .980f;
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

    private void updateControllableBodies()
    {
        var cntro = ecs.getComponents(Component.ControlPoints);
        for (Map.Entry<String, GameComponent> entry : cntro.entrySet())
        {
            String entity = entry.getKey();
            var b = ecs.getComponentFor(entity, Component.RigidBody2D);
            FBody2D body = Component.RigidBody2D.coerce(b);
            GameComponent component = entry.getValue();
            ControlPoints controlPoints = Component.ControlPoints.coerce(component);
            vectorBuffer1.zero();
            if (controlPoints.isLeft())
            {
                vectorBuffer1.x -= body.force();
            }
            if (controlPoints.isRight())
            {
                vectorBuffer1.x += body.force();
            }
            if (controlPoints.isUp())
            {
                vectorBuffer1.y += body.force();
            }
            if (controlPoints.isDown())
            {
                vectorBuffer1.y -= body.force();
            }
            body.addAcc(vectorBuffer1);
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

        updateControllableBodies();

        // todo: (LARGE)
        //  the memory where the objects reside should reside on the GPU and the CPU/
        //  host code should "query" this memory when necessary. Instead of being host
        //  local and transferring to the GPU every frame, as is the current design,
        //  this would drastically cut down on the amount of memory transferred back
        //  and forth from the GPU.
        //  Also, once this is in place, the vertex data should be prepared on the GPU
        //  as well, so CL can prep the data for use by GL.

        // todo: the buffers generated during these OCL calls can be carried forward
        //  and only pulled off the GPU at the very end.
        OCLFunctions.integrate(dt, GRAVITY_X, GRAVITY_Y, FRICTION, spatialPartition);
        OCLFunctions.calculate_key_bank_offsets();

        // todo: calculate this during bank scan. It should be possible on the final
        //  scan phase to calculate one further value, as if the scan were inclusive.
        //  The value can the be read back into the host so a buffer can be allocated.
        spatialPartition.calculateKeyBankSize();

        OCLFunctions.generate_key_bank(spatialPartition);
        OCLFunctions.scan_key_offsets(spatialPartition);
        OCLFunctions.generate_key_map(spatialPartition);

        // todo #0: replace this with OCL calls
        var candidates = spatialPartition.computeCandidatesEX();
        // todo #1: need to make a kernel that determines the key sums of each body
        //  as well as the total size needed for the entire candidate buffer. Then
        //  a separate kernel will be needed to query the key map for every body and
        //  compute their candidates, porting the current logic into that kernel.
        //  The output of the calculation just needs to conform to the structure
        //  assumed below, which is effectively an array of 2 dimensional int vectors.
        //  May make sense to split "compute candidates" into "count candidates" and
        //  "find candidates" phases. See if buffer can be created in GPU and not
        //  passed back out, having the next phase start immediately after

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
