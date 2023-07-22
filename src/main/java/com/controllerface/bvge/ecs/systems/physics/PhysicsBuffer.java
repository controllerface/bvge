package com.controllerface.bvge.ecs.systems.physics;

import com.controllerface.bvge.Main;
import com.controllerface.bvge.cl.OpenCL;
import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_mem;

import java.nio.FloatBuffer;

import static org.jocl.CL.*;

public class PhysicsBuffer
{
    public MemoryBuffer bounds;
    public MemoryBuffer bodies;
    public MemoryBuffer points;
    public MemoryBuffer key_map;
    public MemoryBuffer key_bank;
    public MemoryBuffer key_counts;
    public MemoryBuffer key_offsets;
    public MemoryBuffer candidates;

    public PhysicsBuffer()
    {
        OpenCL.makeBuffers(this);
    }

    public void transferAll()
    {
        key_map.setDoTransfer(false);
        key_bank.setDoTransfer(false);
        //key_counts.setDoTransfer(false);
        key_offsets.setDoTransfer(false);
        if (candidates != null) candidates.setDoTransfer(false);

//        bounds.transfer();
//        bodies.transfer();
//        points.transfer();
        key_map.transfer();
        key_bank.transfer();
        key_counts.transfer();
        key_offsets.transfer();
        if (candidates != null) candidates.transfer();

//        bodies = null;
//        bounds = null;
//        points = null;
        key_map = null;
        key_bank = null;
        key_counts = null;
        key_offsets = null;
        candidates = null;
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
}
