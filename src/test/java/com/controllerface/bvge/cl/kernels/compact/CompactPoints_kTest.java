package com.controllerface.bvge.cl.kernels.compact;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class CompactPoints_kTest
{
    @Test
    public void generate_kernel()
    {
        System.out.println(CompactPoints_k.kernel_source);
    }
}