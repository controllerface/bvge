package com.controllerface.bvge.gpu.cl.programs.scan;

import com.controllerface.bvge.gpu.GPU;
import com.controllerface.bvge.gpu.cl.kernels.KernelType;
import com.controllerface.bvge.gpu.cl.programs.GPUProgram;

public class ScanInt4Array extends GPUProgram
{
    @Override
    public GPUProgram init()
    {
        src.add(func_exclusive_scan);
        src.add(GPU.CL.read_src("programs/scan_int4_array.cl"));

        make_program();

        load_kernel(KernelType.scan_int4_single_block);
        load_kernel(KernelType.scan_int4_multi_block);
        load_kernel(KernelType.complete_int4_multi_block);

        return this;
    }
}
