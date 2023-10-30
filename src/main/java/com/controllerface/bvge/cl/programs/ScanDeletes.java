package com.controllerface.bvge.cl.programs;

import com.controllerface.bvge.cl.GPUProgram;

import static com.controllerface.bvge.cl.CLUtils.read_src;
import static com.controllerface.bvge.cl.GPU.Kernel;

public class ScanDeletes extends GPUProgram
{
    @Override
    protected void init()
    {
        src.add(const_hull_flags);
        src.add(func_exclusive_scan);
        src.add(read_src("kernels/scan_deletes.cl"));

        make_program();

        load_kernel(Kernel.locate_out_of_bounds);
        load_kernel(Kernel.scan_deletes_single_block_out);
        load_kernel(Kernel.scan_deletes_multi_block_out);
        load_kernel(Kernel.complete_deletes_multi_block_out);
    }
}
