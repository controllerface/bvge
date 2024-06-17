package com.controllerface.bvge.cl.kernels.crud;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class CreateBoneChannel_kTest
{
    @Test
    public void generate_kernel()
    {
        System.out.println(CreateBoneChannel_k.cl_kernel());
    }
}