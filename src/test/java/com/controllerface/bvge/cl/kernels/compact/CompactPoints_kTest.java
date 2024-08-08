package com.controllerface.bvge.cl.kernels.compact;

import com.controllerface.bvge.gpu.cl.kernels.compact.CompactPoints_k;
import org.junit.jupiter.api.Test;

class CompactPoints_kTest
{
    @Test
    public void generate_kernel()
    {
        System.out.println(CompactPoints_k.kernel_source);
    }
}