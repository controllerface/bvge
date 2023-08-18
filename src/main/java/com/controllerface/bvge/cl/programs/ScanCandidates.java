package com.controllerface.bvge.cl.programs;

import com.controllerface.bvge.cl.GpuKernel;

import static com.controllerface.bvge.cl.OpenCL.*;
import static com.controllerface.bvge.cl.OpenCLUtils.read_src;

public class ScanCandidates extends GpuKernel
{
    @Override
    protected void init()
    {
        add_src(func_exclusive_scan);
        add_src(read_src("kernels/scan_key_candidates.cl"));
        make_program();
        make_kernel(kn_scan_candidates_single_block);
        make_kernel(kn_scan_candidates_multi_block);
        make_kernel(kn_complete_candidates_multi_block);
    }
}
