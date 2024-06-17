package com.controllerface.bvge.cl.kernels.crud;

import org.junit.jupiter.api.Test;

class CreatePoint_kTest
{
    @Test
    public void generate_kernel()
    {
        System.out.println(CreatePoint_k.cl_kernel());
    }
}