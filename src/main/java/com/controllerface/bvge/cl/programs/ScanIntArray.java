package com.controllerface.bvge.cl.programs;

import com.controllerface.bvge.cl.GpuKernel;

import static com.controllerface.bvge.cl.OpenCL.*;
import static com.controllerface.bvge.cl.OpenCLUtils.read_src;

public class ScanIntArray extends GpuKernel
{
    @Override
    protected void init()
    {
        add_src(func_exclusive_scan);
        add_src(read_src("kernels/scan_int_array.cl"));
        make_program();
        make_kernel(kn_scan_int_single_block);
        make_kernel(kn_scan_int_multi_block);
        make_kernel(kn_complete_int_multi_block);
    }
}
