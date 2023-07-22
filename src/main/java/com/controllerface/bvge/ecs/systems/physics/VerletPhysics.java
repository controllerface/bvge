package com.controllerface.bvge.ecs.systems.physics;

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
    private final int EDGE_STEPS = 4;
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

    private void tickSimulation(float dt, PhysicsBuffer physicsBuffer)
    {
        // integration
        OpenCL.integrate(physicsBuffer, dt, GRAVITY_X, GRAVITY_Y, FRICTION, spatialPartition);

        // broad phase collision
        OpenCL.calculate_bank_offsets(physicsBuffer, spatialPartition);
        OpenCL.generate_key_bank(physicsBuffer, spatialPartition);
        OpenCL.calculate_map_offsets(physicsBuffer, spatialPartition);
        OpenCL.generate_key_map(physicsBuffer, spatialPartition);
        OpenCL.locate_in_bounds(physicsBuffer, spatialPartition);

        // narrow phase collision and reaction
        OpenCL.collide(physicsBuffer);

        // resolve edges
        OpenCL.resolve_constraints(physicsBuffer, EDGE_STEPS);

        // todo: avoid transfer, use existing buffer in new kernel
        physicsBuffer.transferAll();
    }


    private void simulate(float dt)
    {
        var bodies = ecs.getComponents(Component.RigidBody2D);

        // if somehow there are no bodies, just bail. something is probably really wrong
        if (bodies == null || bodies.isEmpty())
        {
            return;
        }

        // todo: need to account for this in the kernel somehow so it can be
        //  updated inside the sub-steps. Right now this is the last point before
        //  the memory is transferred out.
        updateControllableBodies();

        var physicsBuffer = new PhysicsBuffer();

        this.accumulator += dt;
        while (this.accumulator >= TICK_RATE)
        {
            float sub_step = TICK_RATE / SUB_STEPS;
            for (int i = 0; i < SUB_STEPS; i++)
            {
                this.tickSimulation(sub_step, physicsBuffer);
                this.accumulator -= sub_step;
            }
        }

        physicsBuffer.transferFinish();

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
