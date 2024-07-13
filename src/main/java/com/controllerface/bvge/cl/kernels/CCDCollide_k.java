package com.controllerface.bvge.cl.kernels;

public class CCDCollide_k extends GPUKernel
{
    public enum Args
    {
        edges,
        points,
        bounds,
        bounds_bank_data,
        hull_entity_ids,
        hull_flags,
        point_hull_indices,
        candidates,
        match_offsets,
        key_map,
        key_bank,
        key_counts,
        key_offsets,
        matches,
        used,
        counter,
        x_subdivisions,
        key_count_length,
        max_index,
    }

    public CCDCollide_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
