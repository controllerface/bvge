package com.controllerface.bvge.cl.kernels.crud;

import com.controllerface.bvge.gpu.cl.kernels.crud.CreateBoneRef_k;
import org.junit.jupiter.api.Test;

class CreateBoneRef_kTest
{
    @Test
    public void generate_kernel()
    {
        System.out.println(CreateBoneRef_k.kernel_source);
    }
}