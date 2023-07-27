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
    private final int SUB_STEPS = 4;
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
    private final float GRAVITY_Y = 0;//-(9.8f * 50) * SUB_STEPS;
    private final float FRICTION = .995f;

    private final SpatialPartition spatialPartition;
    private PhysicsBuffer physicsBuffer;

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

    private void updateControllableBodies(float dt)
    {
        var components = ecs.getComponents(Component.ControlPoints);
        for (Map.Entry<String, GameComponent> entry : components.entrySet())
        {
            String entity = entry.getKey();
            GameComponent component = entry.getValue();
            ControlPoints controlPoints = Component.ControlPoints.coerce(component);

            var b = ecs.getComponentFor(entity, Component.RigidBody2D);
            BodyIndex body = Component.RigidBody2D.coerce(b);
            vectorBuffer1.zero();
            if (controlPoints.is_moving_left())
            {
                vectorBuffer1.x -= body.force();
            }
            if (controlPoints.is_moving_right())
            {
                vectorBuffer1.x += body.force();
            }
            if (controlPoints.is_moving_up())
            {
                vectorBuffer1.y += body.force();
            }
            if (controlPoints.is_moving_down())
            {
                vectorBuffer1.y -= body.force();
            }

            if (vectorBuffer1.x != 0f || vectorBuffer1.y != 0)
            {
                OpenCL.update_accel(body.bodyIndex(), vectorBuffer1.x, vectorBuffer1.y);
            }

            if (controlPoints.is_rotating_right() ^ controlPoints.is_rotating_left())
            {
                float angle = controlPoints.is_rotating_right() ? -500 : 500;
                OpenCL.rotate_body(body.bodyIndex(), angle * dt * dt);
            }
        }
    }

    private void tickSimulation(float dt)
    {
        // todo: need to account for this in the kernel somehow so it can be
        //  updated inside the sub-steps. Right now this is the last point before
        //  the memory is transferred out.
        updateControllableBodies(dt);

        // integration
        OpenCL.integrate(dt, spatialPartition);

        // broad phase collision
        OpenCL.calculate_bank_offsets(spatialPartition);

        if (spatialPartition.getKey_bank_size() == 0)
        {
            return;
        }

        OpenCL.generate_key_bank(spatialPartition);
        OpenCL.calculate_map_offsets(spatialPartition);
        OpenCL.generate_key_map(spatialPartition);
        OpenCL.locate_in_bounds(spatialPartition);
        OpenCL.count_candidates();
        OpenCL.count_matches();
        OpenCL.aabb_collide();
        OpenCL.finalize_candidates();

        // narrow phase collision/reaction
        OpenCL.sat_collide();

        // resolve edges
        OpenCL.resolve_constraints(EDGE_STEPS);
    }

    //boolean run_once = false;

    private void simulate(float dt)
    {
        var bodies = ecs.getComponents(Component.RigidBody2D);

        // if somehow there are no bodies, just bail. something is probably really wrong
        if (bodies == null || bodies.isEmpty())
        {
            return;
        }

        this.accumulator += dt;
        while (this.accumulator >= TICK_RATE)// || !run_once)
        {
            //run_once = true;
            float sub_step = TICK_RATE / SUB_STEPS;
            for (int i = 0; i < SUB_STEPS; i++)
            {
                this.tickSimulation(sub_step);
                this.accumulator -= sub_step;
                physicsBuffer.finishTick();
            }
        }

        physicsBuffer.finishLoop();
        float drift = this.accumulator / TICK_RATE;
        if (drift != 0)
        {
            // todo: once work starts in on renderer in earnest, check if this needs to be done or not
            //  initial visuals without it don't look bad, but would be good to see if there's some
            //  kind of improvement if the lerp is done. It should only affect the visual location of
            //  objects, not their actual location.
            //this.lerp(drift);
        }
    }


    @Override
    public void run(float dt)
    {
        if (physicsBuffer == null)
        {
            this.physicsBuffer = new PhysicsBuffer();
            this.physicsBuffer.set_gravity_x(GRAVITY_X);
            this.physicsBuffer.set_gravity_y(GRAVITY_Y);
            this.physicsBuffer.set_friction(FRICTION);
            OpenCL.setPhysicsBuffer(physicsBuffer);
        }

        simulate(dt);
    }

    @Override
    public void shutdown()
    {
        physicsBuffer.shutdown();
    }
}
