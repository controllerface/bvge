package com.controllerface.bvge.gpu.cl;

import com.controllerface.bvge.gpu.GPU;
import com.controllerface.bvge.gpu.GPUResource;
import com.controllerface.bvge.gpu.cl.contexts.CL_CommandQueue;
import com.controllerface.bvge.gpu.cl.contexts.CL_Context;
import com.controllerface.bvge.gpu.cl.devices.CL_Device;

import java.util.ArrayList;
import java.util.List;

public class CL_ComputeController implements GPUResource
{
    public final long max_work_group_size;
    public final long max_scan_block_size;
    public final long[] preferred_work_size;
    public final int preferred_work_size_int;
    public final long[] local_work_default;
    public final long[] global_single_size = GPU.CL.arg_long(1);

    public final CL_Device device;
    public final CL_Context context;
    public final CL_CommandQueue physics_queue;
    public final CL_CommandQueue render_queue;
    public final CL_CommandQueue sector_queue;

    private final List<GPUResource> resources = new ArrayList<>();

    public CL_ComputeController(long max_work_group_size,
                                long max_scan_block_size,
                                long[] preferred_work_size,
                                int preferred_work_size_int,
                                long[] local_work_default,
                                CL_Device device,
                                CL_Context context,
                                CL_CommandQueue physics_queue,
                                CL_CommandQueue render_queue,
                                CL_CommandQueue sector_queue)
    {
        this.max_work_group_size = max_work_group_size;
        this.max_scan_block_size = max_scan_block_size;
        this.preferred_work_size = preferred_work_size;
        this.preferred_work_size_int = preferred_work_size_int;
        this.local_work_default = local_work_default;
        this.device = device;
        this.context = context;
        this.physics_queue = physics_queue;
        this.render_queue = render_queue;
        this.sector_queue = sector_queue;

        resources.add(device);
        resources.add(context);
        resources.add(physics_queue);
        resources.add(render_queue);
        resources.add(sector_queue);
    }

    public int calculate_preferred_global_size(int globalWorkSize)
    {
        int remainder = globalWorkSize % preferred_work_size_int;
        if (remainder != 0)
        {
            globalWorkSize += (preferred_work_size_int - remainder);
        }
        return globalWorkSize;
    }

    public int work_group_count(int n)
    {
        return (int) Math.ceil((float) n / (float) max_scan_block_size);
    }

    @Override
    public void release()
    {
        for (var resource : resources)
        {
            resource.release();
        }
    }
}
