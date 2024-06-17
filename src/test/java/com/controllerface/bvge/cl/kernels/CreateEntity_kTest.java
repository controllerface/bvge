package com.controllerface.bvge.cl.kernels;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class CreateEntity_kTest
{
    @Test
    public void generate_kernel()
    {
        System.out.println(CreateEntity_k.cl_kernel());
    }
}