package com.controllerface.bvge.cl.programs;

import com.controllerface.bvge.cl.CLUtils;
import com.controllerface.bvge.cl.GPUProgram;
import com.controllerface.bvge.cl.Kernel;

public class ScanDeletes extends GPUProgram
{
    @Override
    public void init()
    {
        src.add(const_entity_flags);
        src.add(const_hull_flags);
        src.add(func_exclusive_scan);
        src.add(CLUtils.read_src("programs/scan_deletes.cl"));

        make_program();

        load_kernel(Kernel.locate_out_of_bounds);
        load_kernel(Kernel.scan_deletes_single_block_out);
        load_kernel(Kernel.scan_deletes_multi_block_out);
        load_kernel(Kernel.complete_deletes_multi_block_out);
        load_kernel(Kernel.compact_entities);
        load_kernel(Kernel.compact_hulls);
        load_kernel(Kernel.compact_edges);
        load_kernel(Kernel.compact_points);
        load_kernel(Kernel.compact_hull_bones);
        load_kernel(Kernel.compact_armature_bones);
    }
}
