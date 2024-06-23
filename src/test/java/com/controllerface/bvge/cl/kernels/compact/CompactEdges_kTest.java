package com.controllerface.bvge.cl.kernels.compact;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class CompactEdges_kTest
{
    @Test
    public void generate_kernel()
    {
        System.out.println(CompactEdges_k.kernel_source);
    }
}