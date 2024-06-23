package com.controllerface.bvge.cl.kernels.compact;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class CompactHulls_kTest
{
    @Test
    public void generate_kernel()
    {
        System.out.println(CompactHulls_k.kernel_source);
    }
}