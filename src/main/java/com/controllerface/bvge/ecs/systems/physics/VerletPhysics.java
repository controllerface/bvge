package com.controllerface.bvge.ecs.systems.physics;

import com.controllerface.bvge.Main;
import com.controllerface.bvge.cl.OpenCL;
import com.controllerface.bvge.data.*;
import com.controllerface.bvge.ecs.ECS;
import com.controllerface.bvge.ecs.components.*;
import com.controllerface.bvge.ecs.systems.GameSystem;
import org.joml.Vector2f;

import java.util.*;

public class VerletPhysics extends GameSystem
{
    private final float TARGET_FPS = 60.0f;
    private final float TICK_RATE = 1.0f / TARGET_FPS;
    private final int SUB_STEPS = 2;
    private final int EDGE_STEPS = 2;
    private float accumulator = 0.0f;

    // todo: these values should not be global, but per-object.
    //  When an object is considered "in-contact" with a static object,
    //  it should be assigned friction based on that object. There should
    //  be some type of ambient friction (i.e. friction of the "air" or ambient medium)
    //  that applies when no other friction value is set, and then friction
    //  values applied by objects could not go above the ambient friction.
    //  In this way, friction is a "status effect" that is cleared every frame
    //  and applied when contact occurs.
    private final float GRAVITY_X = 0;
    private final float GRAVITY_Y = 0;//-(9.8f * 100);
    private final float FRICTION = .980f;


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

    private void tickEdges()
    {
        var bodies = ecs.getComponents(Component.RigidBody2D);
        boolean lastStep;
        for (int i = 0; i < EDGE_STEPS; i++)
        {
            lastStep = i == EDGE_STEPS - 1;
            // todo: would be worth trying to iterate on each edge but in parallel for each body
            //  solving ALL edges at once doesn't work, but in series it may
            for (Map.Entry<String, GameComponent> entry : bodies.entrySet())
            {
                FBody2D body = Component.RigidBody2D.coerce(entry.getValue());
                if (lastStep || body.bounds().si_bank_size() > 0)
                {
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

        var physicsBuffer = new PhysicsBuffer();

        // integration
        OpenCL.integrate(physicsBuffer, dt, GRAVITY_X, GRAVITY_Y, FRICTION, spatialPartition);

        // broad phase collision
        OpenCL.calculate_bank_offsets(physicsBuffer, spatialPartition);
        OpenCL.generate_key_bank(physicsBuffer, spatialPartition);
        OpenCL.calculate_map_offsets(physicsBuffer, spatialPartition);
        OpenCL.generate_key_map(physicsBuffer, spatialPartition);
        OpenCL.locate_in_bounds(physicsBuffer, spatialPartition);

        // narrow phase collision
        OpenCL.collide(physicsBuffer);

        // todo: avoid transfer, use existing buffer in new kernel
        physicsBuffer.transferAll();

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
