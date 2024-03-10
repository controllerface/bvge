package com.controllerface.bvge.cl.programs;

import com.controllerface.bvge.cl.CLUtils;
import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUProgram;

public class ScanDeletes extends GPUProgram
{
    @Override
    public void init()
    {
        src.add(const_hull_flags);
        src.add(func_exclusive_scan);
        src.add(CLUtils.read_src("programs/scan_deletes.cl"));

        make_program();

        load_kernel(GPU.Kernel.locate_out_of_bounds);
        load_kernel(GPU.Kernel.scan_deletes_single_block_out);
        load_kernel(GPU.Kernel.scan_deletes_multi_block_out);
        load_kernel(GPU.Kernel.complete_deletes_multi_block_out);
        load_kernel(GPU.Kernel.compact_armatures);
        load_kernel(GPU.Kernel.compact_hulls);
        load_kernel(GPU.Kernel.compact_edges);
        load_kernel(GPU.Kernel.compact_points);
        load_kernel(GPU.Kernel.compact_bones);
        load_kernel(GPU.Kernel.compact_armature_bones);
    }
}
