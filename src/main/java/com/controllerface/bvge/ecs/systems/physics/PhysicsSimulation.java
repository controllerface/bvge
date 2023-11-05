package com.controllerface.bvge.ecs.systems.physics;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.ecs.ECS;
import com.controllerface.bvge.ecs.components.*;
import com.controllerface.bvge.ecs.systems.GameSystem;
import org.joml.Vector2f;

import java.util.*;

public class PhysicsSimulation extends GameSystem
{
    private final static float TARGET_FPS = 60.0f;
    private final static float TICK_RATE = 1.0f / TARGET_FPS;
    private final static int SUB_STEPS = 4;
    private final static int EDGE_STEPS = 4;
    private float accumulator = 0.0f;

    // todo: gravity should not be a constant but calculated based on proximity next to planets and other large bodies
    private final static float GRAVITY_X = 0;
    private final static float GRAVITY_Y = -(9.8f * 50) * SUB_STEPS;

    // todo: investigate if this should be variable as well. It may make sense to increase damping in some cases,
    //  and lower it in others, for example in space vs on a planet. It may also be useful to set the direction
    //  or make damping interact with the gravity vector in some way.
    private final static float MOTION_DAMPING = .990f;

    private final UniformGrid uniform_grid;
    private PhysicsBuffer physics_buffer;

    /**
     * This buffer is reused each tick to avoid creating a new one every frame and for each object.
     * It should always be zeroed before each use.
     */
    private final Vector2f vectorBuffer = new Vector2f();

    public PhysicsSimulation(ECS ecs, UniformGrid uniform_grid)
    {
        super(ecs);
        this.uniform_grid = uniform_grid;
    }

    private void updateControllableBodies(float dt)
    {
        var components = ecs.getComponents(Component.ControlPoints);
        for (Map.Entry<String, GameComponent> entry : components.entrySet())
        {
            String entity = entry.getKey();
            GameComponent component = entry.getValue();
            ControlPoints controlPoints = Component.ControlPoints.coerce(component);
            ArmatureIndex armature = Component.Armature.forEntity(ecs, entity);
            LinearForce force = Component.LinearForce.forEntity(ecs, entity);

            Objects.requireNonNull(controlPoints);
            Objects.requireNonNull(armature);
            Objects.requireNonNull(force);

            vectorBuffer.zero();
            if (controlPoints.is_moving_left())
            {
                vectorBuffer.x -= force.magnitude();
            }
            if (controlPoints.is_moving_right())
            {
                vectorBuffer.x += force.magnitude();
            }
            if (controlPoints.is_moving_up())
            {
                vectorBuffer.y += force.magnitude();
            }
            if (controlPoints.is_moving_down())
            {
                vectorBuffer.y -= force.magnitude();
            }
            if (controlPoints.is_space_bar_down())
            {
                vectorBuffer.y -= GRAVITY_Y;
            }

            if (vectorBuffer.x != 0f || vectorBuffer.y != 0)
            {
                GPU.update_accel(armature.index(), vectorBuffer.x, vectorBuffer.y);
            }

            // todo: implement this for armatures
//            if (controlPoints.is_rotating_right() ^ controlPoints.is_rotating_left())
//            {
//                float angle = controlPoints.is_rotating_right() ? -200f : 200f;
//                OpenCL.rotate_hull(hull.index(), angle * dt * dt);
//            }
        }
    }


    /**
     * This is the core of the physics simulation. Upon return from this method, the simulation is
     * advanced one tick. Note that this class uses a fixed time step, so the time delta should always
     * be the same. Most work done within this method is delegated to the GPU for performance.
     *
     * @param dt amount of time that is simulated during the physics tick.
     */
    private void tickSimulation(float dt)
    {
        /*
        * CPU Side - Setup
        * */

        // Before the GPU begins the simulation cycle, player input is handled and the memory structures
        // in the GPU are updated with the proper values.
        updateControllableBodies(dt);

        /*
        * GPU Side - Physics
        * */

        // The first step is to animate the vertices of bone-tracked hulls. This ensures that all tracked objects
        // that have animation will have their hulls moved into position for the current frame. It may seem odd
        // to process animations as part of physics and not rendering, however this is required as the animated
        // objects need to be accounted for in physical space. The hulls representing the rendered meshes are
        // what is actually moved, and the result of the hull movement is used to position the original mesh for
        // rendering. This separation is necessary as model geometry is too complex to use as a collision boundary.
        GPU.animate_hulls();


        // Now that all animated hulls are in their initial frame positions, we perform the mathematical steps
        // to calculate where the individual points of each hull currently are. When this call returns, all
        // tracked physics objects will be in their new locations, and points will have their current and
        // previous location values updated for the current tick.
        GPU.integrate(dt, uniform_grid);

        GPU.locate_out_of_bounds();

        GPU.delete_and_compact();


        // Once positions are adjusted, edge constraints are enforced to ensure that rigid bodies maintain their
        // defined shapes. Without this step, the individual points of the tracked physics hulls will deform on
        // impact, and may fly off in random directions, typically causing simulation failure. The number of steps
        // that are performed each tick has an impact on the accuracy of the hull boundaries within the simulation.
        GPU.resolve_constraints(EDGE_STEPS);

        /*
        - Broad Phase Collision -
        =========================
        Before the final and more computationally expensive collision checks are performed, A broad phase check
        is done to narrow down the potential collision candidates. Because this is implemented using parallelized
        compute kernels, the process is more verbose than a traditional CPU based approach. At a high level, this
        process is used in place of more complex constructs like Map<> and Set<>, but with capacities of backing
        structures being pre-computed, to fulfill the fixed memory size requirements of the GPU kernel.

        There are two top-level "conceptual" structures, a key bank and a key map, and there is the concept
        of a key itself.

        Keys in this context are simply two-dimensional integer vectors that point to a "cell" of the uniform
        grid, which is a structure that imposes a coarse grid over the viewable area of the screen. For every hull,
        if it is within view, it will be inside, or overlapping one or more of these cells. A "key" value simply
        describes this cell location.

        The key bank is a large block of memory that holds the actual key data for each object. Objects with entries
        in the key bank will have their corresponding key bank tables updated to point to their start and offset
        within this key bank. It is recomputed every tick, because the values and number of keys change depending
        on object location and orientation. Objects that are off-screen are handled such that they always have empty
        key banks, removing them from consideration before the broad phase even starts.

        The key map is a structure that maps each individual spatial key to the objects that have that key within
        their local key bank. Or to put it another way, every object that is touching the cell associated with a
        particular key, has its index stored in this map under that key. This makes it possible to then query
        the map by key, effectively getting a list of all objects that could be colliding with other objects,
        by virtue of the fact that they share a key.
        */

        // The first task before checking boundaries is to calculate the bank offsets for this frame. These offsets
        // determine how much space in the key bank is allocated to each possible collision candidate. The amount of
        // space varies based on the size and orientation of the object within the uniform grid.
        GPU.calculate_bank_offsets(uniform_grid);

        // As a fail-safe, if the total bank size is zero, it means there's no tracked objects, so simply return.
        // This condition is unlikely to occur accept when the simulation is first starting up.
        if (uniform_grid.get_key_bank_size() == 0)
        {
            return;
        }

        // Once we know there are some objects to track, we can generate the keys needed to further process
        // the tracked objects. This call generates the keys for each object, and stores them in the global
        // key bank. Hull bounds tables are updated with the correct offsets and counts as needed.
        GPU.generate_keys(uniform_grid);

        // After keys are generated, the next step is to calculate the space needed for the key map. This is
        // a similar process to calculating the bank offsets.
        GPU.calculate_map_offsets(uniform_grid);

        // Now, the keymap itself is built. This is the structure that provides the ability to query
        // objects within the uniform grid structure.
        GPU.build_key_map(uniform_grid);

        // Hulls are now filtered to ensure that only objects that are within the uniform grid boundary
        // are considered for collisions.
        GPU.locate_in_bounds(uniform_grid);

        // In a first pass, the number of total possible candidates is calculated for each hull. This is
        // necessary to correctly size a match table buffer which is used in the GPU kernel.
        GPU.count_candidates();

        // In a second pass, candidate sizes are scanned to calculate the required size of the match table.
        GPU.count_matches();

        // Finally, the actual broad phase collision check is performed. Once complete, the match table will
        // be filled in with all matches. There may be some unused sections of the table, because some objects
        // may be eliminated as candidates during the check.
        GPU.aabb_collide();

        // This last step cleans up the match table, retaining only the used sections of the buffer.
        // After this step, the matches are ready for the narrow phase check.
        GPU.finalize_candidates();

        // If there were no candidate collisions, there's nothing left to do
        if (physics_buffer.candidates == null)
        {
            return;
        }

        /*
        - Narrow Phase Collision/Reaction -
        ===================================
        Objects found to be close enough for a narrow check are now fully examined to determine if they are
        actually colliding. Any collisions that are detected will have reactions calculated and forwarded
        to a series of kernels that work together to scan and sort the reactions, then ultimately apply them
        to the appropriate points.

        Collision is detected using the separating axis theorem for all collisions that involve polygons.
        Circle-to-circle collisions are handled using a simple distance/radius check. Because of this, circles
        are significantly less demanding to simulate.

        After all collision reactions have been applied, there is a final step that applies all hull movements
        to their parent armatures. This last step is needed for complex models, to ensure the groups of hulls
        are moved together as a unit. Without this step, armature based objects will not collide correctly.
        */

        // Using the candidates generated by the AABB checks, we now do a full collision check. For any objects
        // that are found to be colliding, appropriate reaction vectors are generated and stored.
        GPU.sat_collide();

        // It is possible that after the check, no objects are found to be colliding. If that happens, exit.
        if (physics_buffer.get_reaction_count() == 0)
        {
            return;
        }

        // Since we did have some reactions, we need to figure out what points were affected. This is needed
        // so that reactions can be accumulated in series and applied to the point they affect.
        GPU.scan_reactions();

        // After the initial scan, the reaction buffers are sorted to match the layout computed in the scan
        // step. After this call, the buffers are in ascending order by point index.
        GPU.sort_reactions();

        // Now all points with reactions are able to sum all their reactions and apply them, as well as
        // enforcing constraints on the velocities of the affected points.
        GPU.apply_reactions();

        // Once all points have been relocated, all hulls are in their required positions for this frame.
        // Movements applied to hulls are now accumulated and applied to their parent armatures.
        GPU.move_armatures();

//        GPU.locate_out_of_bounds();
//
//        GPU.delete_and_compact();
    }

    private void simulate(float dt)
    {
        var armatures = ecs.getComponents(Component.Armature);

        // if there are no armatures, just bail. things may still be setting up
        if (armatures == null || armatures.isEmpty())
        {
            return;
        }

        this.accumulator += dt;
        while (this.accumulator >= TICK_RATE)
        {
            //run_once = true;
            float sub_step = TICK_RATE / SUB_STEPS;
            for (int i = 0; i < SUB_STEPS; i++)
            {
                this.tickSimulation(sub_step);
                this.accumulator -= sub_step;
                physics_buffer.finishTick();
            }
        }

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
    public void tick(float dt)
    {
        if (physics_buffer == null)
        {
            this.physics_buffer = new PhysicsBuffer();
            this.physics_buffer.set_gravity_x(GRAVITY_X);
            this.physics_buffer.set_gravity_y(GRAVITY_Y);
            this.physics_buffer.set_damping(MOTION_DAMPING);
            GPU.set_physics_buffer(physics_buffer);
        }

        simulate(dt);
    }
}
