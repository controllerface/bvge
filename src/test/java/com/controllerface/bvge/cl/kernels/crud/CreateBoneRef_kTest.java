package com.controllerface.bvge.cl.kernels.crud;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class CreateBoneRef_kTest
{
    @Test
    public void generate_kernel()
    {
        System.out.println(CreateBoneRef_k.cl_kernel());
    }
}