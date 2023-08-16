package com.controllerface.bvge.cl.programs;

import com.controllerface.bvge.cl.GpuKernel;

import static com.controllerface.bvge.cl.OpenCL.*;
import static com.controllerface.bvge.cl.OpenCLUtils.read_src;

public class ScanIntArrayOut extends GpuKernel
{
    @Override
    protected void init()
    {
        var source = read_src("kernels/scan_int_array_out.cl");
        this.program = cl_p(func_exclusive_scan, source);
        this.kernels.put(kn_scan_int_single_block_out,    cl_k(program, kn_scan_int_single_block_out));
        this.kernels.put(kn_scan_int_multi_block_out,     cl_k(program, kn_scan_int_multi_block_out));
        this.kernels.put(kn_complete_int_multi_block_out, cl_k(program, kn_complete_int_multi_block_out));
    }
}
