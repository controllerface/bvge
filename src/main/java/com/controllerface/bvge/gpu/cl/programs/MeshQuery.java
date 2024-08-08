package com.controllerface.bvge.gpu.cl.programs;

import com.controllerface.bvge.gpu.GPU;
import com.controllerface.bvge.gpu.cl.CLUtils;
import com.controllerface.bvge.gpu.cl.kernels.KernelType;

public class MeshQuery extends GPUProgram
{
    @Override
    public GPUProgram init()
    {
        src.add(const_entity_flags);
        src.add(const_hull_flags);
        src.add(const_hit_thresholds);
        src.add(const_point_flags);
        src.add(GPU.CL.read_src("programs/mesh_query.cl"));

        make_program();

        load_kernel(KernelType.count_mesh_batches);
        load_kernel(KernelType.count_mesh_instances);
        load_kernel(KernelType.write_mesh_details);
        load_kernel(KernelType.calculate_batch_offsets);
        load_kernel(KernelType.transfer_detail_data);
        load_kernel(KernelType.transfer_render_data);

        return this;
    }
}
