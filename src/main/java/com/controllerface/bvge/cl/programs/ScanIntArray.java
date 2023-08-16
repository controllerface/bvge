package com.controllerface.bvge.cl.programs;

import com.controllerface.bvge.cl.GpuKernel;

import static com.controllerface.bvge.cl.OpenCL.*;
import static com.controllerface.bvge.cl.OpenCLUtils.read_src;

public class ScanIntArray extends GpuKernel
{
    @Override
    protected void init()
    {
        var source = read_src("kernels/scan_int_array.cl");
        this.program = cl_p(func_exclusive_scan, source);
        this.kernels.put(kn_scan_int_single_block,    cl_k(program, kn_scan_int_single_block));
        this.kernels.put(kn_scan_int_multi_block,     cl_k(program, kn_scan_int_multi_block));
        this.kernels.put(kn_complete_int_multi_block, cl_k(program, kn_complete_int_multi_block));
    }
}
