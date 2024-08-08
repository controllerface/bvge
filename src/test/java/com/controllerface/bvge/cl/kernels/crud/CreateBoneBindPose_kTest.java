package com.controllerface.bvge.cl.kernels.crud;

import com.controllerface.bvge.gpu.cl.kernels.crud.CreateBoneBindPose_k;
import org.junit.jupiter.api.Test;

class CreateBoneBindPose_kTest
{
    @Test
    public void generate_kernel()
    {
        System.out.println(CreateBoneBindPose_k.kernel_source);
    }
}