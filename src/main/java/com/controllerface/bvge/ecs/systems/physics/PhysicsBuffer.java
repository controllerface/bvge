package com.controllerface.bvge.ecs.systems.physics;

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

    public void transferAll()
    {
        key_map.setDoTransfer(false);
        key_bank.setDoTransfer(false);
        key_counts.setDoTransfer(false);
        key_offsets.setDoTransfer(false);
        if (candidates != null) candidates.setDoTransfer(false);

        bounds.transfer();
        bodies.transfer();
        points.transfer();
        key_map.transfer();
        key_bank.transfer();
        key_counts.transfer();
        key_offsets.transfer();
        if (candidates != null) candidates.transfer();
    }
}