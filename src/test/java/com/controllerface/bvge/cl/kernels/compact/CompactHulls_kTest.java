package com.controllerface.bvge.cl.kernels.compact;

import com.controllerface.bvge.gpu.cl.kernels.compact.CompactHulls_k;
import org.junit.jupiter.api.Test;

class CompactHulls_kTest
{
    @Test
    public void generate_kernel()
    {
        System.out.println(CompactHulls_k.kernel_source);
    }
}