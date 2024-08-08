package com.controllerface.bvge.cl.kernels.crud;

import com.controllerface.bvge.gpu.cl.kernels.crud.CreateHullBone_k;
import org.junit.jupiter.api.Test;

class CreateHullBone_kTest
{
    @Test
    public void generate_kernel()
    {
        System.out.println(CreateHullBone_k.kernel_source);
    }
}