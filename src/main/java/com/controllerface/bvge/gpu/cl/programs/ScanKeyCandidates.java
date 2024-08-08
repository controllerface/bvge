package com.controllerface.bvge.gpu.cl.programs;

import com.controllerface.bvge.gpu.cl.kernels.Kernel;
import com.controllerface.bvge.gpu.cl.CLUtils;

public class ScanKeyCandidates extends GPUProgram
{
    @Override
    public GPUProgram init()
    {
        src.add(func_exclusive_scan);
        src.add(CLUtils.read_src("programs/scan_key_candidates.cl"));

        make_program();

        load_kernel(Kernel.scan_candidates_single_block_out);
        load_kernel(Kernel.scan_candidates_multi_block_out);
        load_kernel(Kernel.complete_candidates_multi_block_out);

        return this;
    }
}
