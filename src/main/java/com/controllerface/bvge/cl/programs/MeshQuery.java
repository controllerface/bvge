package com.controllerface.bvge.cl.programs;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUProgram;

import static com.controllerface.bvge.cl.CLUtils.read_src;

public class MeshQuery extends GPUProgram
{
    @Override
    public void init()
    {
        src.add(read_src("programs/mesh_query.cl"));

        make_program();

        load_kernel(GPU.Kernel.count_mesh_batches);
        load_kernel(GPU.Kernel.count_mesh_instances);
        load_kernel(GPU.Kernel.write_mesh_details);
        load_kernel(GPU.Kernel.calculate_batch_offsets);
        load_kernel(GPU.Kernel.transfer_detail_data);
        load_kernel(GPU.Kernel.transfer_render_data);
    }
}
