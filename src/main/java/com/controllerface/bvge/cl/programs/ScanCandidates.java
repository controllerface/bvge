package com.controllerface.bvge.cl.programs;

import com.controllerface.bvge.cl.GPUProgram;

import static com.controllerface.bvge.cl.GPU.*;
import static com.controllerface.bvge.cl.CLUtils.read_src;

public class ScanCandidates extends GPUProgram
{
    @Override
    protected void init()
    {
        src.add(func_exclusive_scan);
        src.add(read_src("kernels/scan_key_candidates.cl"));

        make_program();

        load_kernel(Kernel.scan_candidates_single_block_out);
        load_kernel(Kernel.scan_candidates_multi_block_out);
        load_kernel(Kernel.complete_candidates_multi_block_out);
    }
}
