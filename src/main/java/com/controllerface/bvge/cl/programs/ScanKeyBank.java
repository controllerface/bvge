package com.controllerface.bvge.cl.programs;

import com.controllerface.bvge.cl.GpuKernel;

import static com.controllerface.bvge.cl.OpenCL.*;
import static com.controllerface.bvge.cl.OpenCLUtils.read_src;

public class ScanKeyBank extends GpuKernel
{
    @Override
    protected void init()
    {
        var source = read_src("kernels/scan_key_bank.cl");
        this.program = cl_p(func_exclusive_scan, source);
        this.kernels.put(kn_scan_bounds_single_block,    cl_k(program, kn_scan_bounds_single_block));
        this.kernels.put(kn_scan_bounds_multi_block,     cl_k(program, kn_scan_bounds_multi_block));
        this.kernels.put(kn_complete_bounds_multi_block, cl_k(program, kn_complete_bounds_multi_block));
    }
}
