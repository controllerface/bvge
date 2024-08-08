package com.controllerface.bvge.gpu.cl.programs;

import com.controllerface.bvge.gpu.cl.kernels.Kernel;
import com.controllerface.bvge.gpu.cl.CLUtils;

public class MeshQuery extends GPUProgram
{
    @Override
    public GPUProgram init()
    {
        src.add(const_entity_flags);
        src.add(const_hull_flags);
        src.add(const_hit_thresholds);
        src.add(const_point_flags);
        src.add(CLUtils.read_src("programs/mesh_query.cl"));

        make_program();

        load_kernel(Kernel.count_mesh_batches);
        load_kernel(Kernel.count_mesh_instances);
        load_kernel(Kernel.write_mesh_details);
        load_kernel(Kernel.calculate_batch_offsets);
        load_kernel(Kernel.transfer_detail_data);
        load_kernel(Kernel.transfer_render_data);

        return this;
    }
}
