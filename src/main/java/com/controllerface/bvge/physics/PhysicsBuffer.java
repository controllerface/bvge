package com.controllerface.bvge.physics;

import com.controllerface.bvge.cl.GPUMemory;
import org.jocl.Pointer;

public class PhysicsBuffer
{
    public GPUMemory key_map;
    public GPUMemory key_bank;
    public GPUMemory key_counts;
    public GPUMemory key_offsets;
    public GPUMemory in_bounds;
    public GPUMemory candidates;
    public GPUMemory candidate_counts;
    public GPUMemory candidate_offsets;
    public GPUMemory matches;
    public GPUMemory matches_used;

    public GPUMemory reactions_in;
    public GPUMemory reactions_out;
    public GPUMemory reaction_index;

    public int x_sub_divisions;
    public int key_count_length;

    private int candidate_buffer_count = 0;
    private int match_buffer_count = 0;
    private int candidate_count = 0;
    private int reaction_count = 0;
    private long final_size = 0;

    private float gravity_x = 0f;
    private float gravity_y = 0f;
    private float damping = 0f;

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

    public float get_damping()
    {
        return damping;
    }

    public void set_damping(float friction)
    {
        this.damping = friction;
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

    public int get_reaction_count()
    {
        return reaction_count;
    }

    public void set_reaction_count(int reaction_count)
    {
        this.reaction_count = reaction_count;
    }

    public long get_final_size()
    {
        return final_size;
    }

    public void set_final_size(long final_size)
    {
        this.final_size = final_size;
    }

    public void finishTick()
    {
        // todo: some of these could be reused between frames, as long as the spatial partition
        //  size does not change. If it does, we would just need to release and re-acquire with
        //  the new size. This should be easy to do on the frame tick.
        if (key_map != null) key_map.release();
        if (key_bank != null) key_bank.release();
        if (key_counts != null) key_counts.release();
        if (key_offsets != null) key_offsets.release();
        if (in_bounds != null) in_bounds.release();
        if (candidate_counts != null) candidate_counts.release();
        if (candidate_offsets != null) candidate_offsets.release();
        if (matches_used != null) matches_used.release();
        if (matches != null) matches.release();
        if (candidates != null) candidates.release();
        if (reactions_in != null) reactions_in.release();
        if (reactions_out != null) reactions_out.release();
        if (reaction_index != null) reaction_index.release();

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
        reactions_in = null;
        reactions_out = null;
        reaction_index = null;
    }
}
