package com.controllerface.bvge.ecs.systems.physics;

import com.controllerface.bvge.cl.OpenCL;
import org.jocl.Pointer;

public class PhysicsBuffer
{
    public MemoryBuffer bounds;
    public MemoryBuffer bodies;
    public MemoryBuffer points;
    public MemoryBuffer edges;

    public MemoryBuffer key_map;
    public MemoryBuffer key_bank;
    public MemoryBuffer key_counts;
    public MemoryBuffer key_offsets;
    public MemoryBuffer in_bounds;
    public MemoryBuffer candidates;
    public MemoryBuffer candidate_counts;
    public MemoryBuffer candidate_offsets;
    public MemoryBuffer matches;
    public MemoryBuffer matches_used;

    public Pointer x_sub_divisions;
    public Pointer key_count_length;

    private int candidate_buffer_count = 0;
    private int match_buffer_count = 0;
    private int candidate_count = 0;

    private float gravity_x = 0f;
    private float gravity_y = 0f;
    private float friction = 0f;

    public float get_gravity_x()
    {
        return gravity_x;
    }

    public void set_gravity_x(float gravity_x)
    {
        this.gravity_x = gravity_x;
    }

    public float get_gravity_y()
    {
        return gravity_y;
    }

    public void set_gravity_y(float gravity_y)
    {
        this.gravity_y = gravity_y;
    }

    public float get_friction()
    {
        return friction;
    }

    public void set_friction(float friction)
    {
        this.friction = friction;
    }

    public PhysicsBuffer()
    {
        OpenCL.initPhysicsBuffer(this);
    }

    public void transferAll()
    {
        key_map.setCopyBuffer(false);
        key_bank.setCopyBuffer(false);
        //key_counts.setDoTransfer(false);
        key_offsets.setCopyBuffer(false);
        in_bounds.setCopyBuffer(false);
        candidate_counts.setCopyBuffer(false);
        candidate_offsets.setCopyBuffer(false);
        matches_used.setCopyBuffer(false);
        matches.setCopyBuffer(false);
        if (candidates != null) candidates.setCopyBuffer(false);

//        bounds.transfer();
//        bodies.transfer();
//        points.transfer();
        key_map.transfer();
        key_bank.transfer();
        key_counts.transfer();
        key_offsets.transfer();
        in_bounds.transfer();
        candidate_counts.transfer();
        candidate_offsets.transfer();
        matches_used.transfer();
        matches.transfer();
        if (candidates != null) candidates.transfer();

//        bodies = null;
//        bounds = null;
//        points = null;
        key_map = null;
        key_bank = null;
        key_counts = null;
        key_offsets = null;
        candidates = null;
        in_bounds = null;
        candidate_counts = null;
        candidate_offsets = null;
        matches_used = null;
        matches = null;
    }

    public void transferFinish()
    {
        bounds.setReleaseAfterTransfer(false);
        bodies.setReleaseAfterTransfer(false);
        points.setReleaseAfterTransfer(false);

        bounds.transfer();
        bodies.transfer();
        points.transfer();
    }

    public void shutdown()
    {
        if (bounds != null)
        {
            bounds.release();
        }
        if (bodies != null)
        {
            bodies.release();
        }
        if (points != null)
        {
            points.release();
        }
        if (key_map != null)
        {
            key_map.release();
        }
        if (key_bank != null)
        {
            key_bank.release();
        }
        if (key_counts != null)
        {
            key_counts.release();
        }
        if (key_offsets != null)
        {
            key_offsets.release();
        }
        if (candidates != null)
        {
            candidates.release();
        }
    }

    public int get_candidate_buffer_count()
    {
        return candidate_buffer_count;
    }

    public void set_candidate_buffer_count(int candidate_count)
    {
        this.candidate_buffer_count = candidate_count;
    }

    public int get_candidate_match_count()
    {
        return match_buffer_count;
    }

    public void set_candidate_match_count(int match_count)
    {
        this.match_buffer_count = match_count;
    }

    public int get_candidate_count()
    {
        return candidate_count;
    }

    public void set_candidate_count(int candidate_count)
    {
        this.candidate_count = candidate_count;
    }
}
