package com.controllerface.bvge.cl.kernels.crud;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class CreateEntityBone_kTest
{
    @Test
    public void generate_kernel()
    {
        System.out.println(CreateEntityBone_k.cl_kernel());
    }
}