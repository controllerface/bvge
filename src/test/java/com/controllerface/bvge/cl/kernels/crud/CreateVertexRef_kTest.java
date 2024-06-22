package com.controllerface.bvge.cl.kernels.crud;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class CreateVertexRef_kTest
{
    @Test
    public void generate_kernel()
    {
        System.out.println(CreateVertexRef_k.kernel_source);
    }
}