package com.controllerface.bvge.cl.programs;

import com.controllerface.bvge.cl.CLUtils;
import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUProgram;

public class ScanInt2Array extends GPUProgram
{
    @Override
    public void init()
    {
        src.add(func_exclusive_scan);
        src.add(CLUtils.read_src("programs/scan_int2_array.cl"));

        make_program();

        load_kernel(GPU.Kernel.scan_int2_single_block);
        load_kernel(GPU.Kernel.scan_int2_multi_block);
        load_kernel(GPU.Kernel.complete_int2_multi_block);
    }
}
