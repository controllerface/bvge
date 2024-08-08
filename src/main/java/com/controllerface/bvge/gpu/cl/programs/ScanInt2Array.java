package com.controllerface.bvge.gpu.cl.programs;

import com.controllerface.bvge.gpu.GPU;
import com.controllerface.bvge.gpu.cl.CLUtils;
import com.controllerface.bvge.gpu.cl.kernels.KernelType;

public class ScanInt2Array extends GPUProgram
{
    @Override
    public GPUProgram init()
    {
        src.add(func_exclusive_scan);
        src.add(GPU.CL.read_src("programs/scan_int2_array.cl"));

        make_program();

        load_kernel(KernelType.scan_int2_single_block);
        load_kernel(KernelType.scan_int2_multi_block);
        load_kernel(KernelType.complete_int2_multi_block);

        return this;
    }
}
