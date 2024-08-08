package com.controllerface.bvge.cl.kernels.compact;

import com.controllerface.bvge.gpu.cl.kernels.compact.CompactEdges_k;
import org.junit.jupiter.api.Test;

class CompactEdges_kTest
{
    @Test
    public void generate_kernel()
    {
        System.out.println(CompactEdges_k.kernel_source);
    }
}