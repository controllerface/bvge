package com.controllerface.bvge.cl.kernels.compact;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class CompactHullBones_kTest
{
    @Test
    public void generate_kernel()
    {
        System.out.println(CompactHullBones_k.kernel_source);
    }
}