package com.controllerface.bvge.cl.kernels.crud;

import org.junit.jupiter.api.Test;


class CreateEntity_kTest
{
    @Test
    public void generate_kernel()
    {
        System.out.println(CreateEntity_k.cl_kernel());
    }
}