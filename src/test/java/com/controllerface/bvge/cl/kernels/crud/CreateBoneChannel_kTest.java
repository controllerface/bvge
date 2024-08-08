package com.controllerface.bvge.cl.kernels.crud;

import com.controllerface.bvge.gpu.cl.kernels.crud.CreateBoneChannel_k;
import org.junit.jupiter.api.Test;

class CreateBoneChannel_kTest
{
    @Test
    public void generate_kernel()
    {
        System.out.println(CreateBoneChannel_k.kernel_source);
    }
}