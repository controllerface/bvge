package com.controllerface.bvge.gpu.cl.programs.scan;

import com.controllerface.bvge.editor.Editor;
import com.controllerface.bvge.gpu.GPU;
import com.controllerface.bvge.gpu.GPUResource;
import com.controllerface.bvge.gpu.cl.buffers.CL_DataTypes;
import com.controllerface.bvge.gpu.cl.GPGPU;
import com.controllerface.bvge.gpu.cl.contexts.CL_CommandQueue;
import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;
import com.controllerface.bvge.gpu.cl.kernels.KernelType;
import com.controllerface.bvge.gpu.cl.kernels.scan.CompleteIntMultiBlock_k;
import com.controllerface.bvge.gpu.cl.kernels.scan.ScanIntMultiBlock_k;
import com.controllerface.bvge.gpu.cl.kernels.scan.ScanIntSingleBlock_k;
import com.controllerface.bvge.gpu.cl.programs.GPUProgram;

import static com.controllerface.bvge.gpu.GPU.CL.arg_long;

public class GPUScanScalarInt implements GPUResource
{
    private final GPUProgram p_scan_int_array = new ScanIntArray();

    private final GPUKernel k_scan_int_single_block;
    private final GPUKernel k_scan_int_multi_block;
    private final GPUKernel k_complete_int_multi_block;

    public GPUScanScalarInt(CL_CommandQueue cmd_queue)
    {
        p_scan_int_array.init();

        k_scan_int_single_block          = new ScanIntSingleBlock_k(cmd_queue, p_scan_int_array);
        k_scan_int_multi_block           = new ScanIntMultiBlock_k(cmd_queue, p_scan_int_array);
        k_complete_int_multi_block       = new CompleteIntMultiBlock_k(cmd_queue, p_scan_int_array);
    }

    public void scan_int(long data_ptr, int n)
    {
        long s = Editor.ACTIVE ? System.nanoTime() : 0;

        int k = GPGPU.compute.work_group_count(n);
        if (k == 1)
        {
            scan_single_block_int(data_ptr, n);
        }
        else
        {
            scan_multi_block_int(data_ptr, n, k);
        }

        if (Editor.ACTIVE)
        {
            long e = System.nanoTime() - s;
            Editor.queue_event("render_model_scan_int", String.valueOf(e));
        }
    }

    private void scan_single_block_int(long data_ptr, int n)
    {
        long local_buffer_size = CL_DataTypes.cl_int.size() * GPGPU.compute.max_scan_block_size;

        k_scan_int_single_block
            .ptr_arg(ScanIntSingleBlock_k.Args.data, data_ptr)
            .loc_arg(ScanIntSingleBlock_k.Args.buffer, local_buffer_size)
            .set_arg(ScanIntSingleBlock_k.Args.n, n)
            .call(GPGPU.compute.local_work_default, GPGPU.compute.local_work_default);
    }

    private void scan_multi_block_int(long data_ptr, int n, int k)
    {
        long local_buffer_size = CL_DataTypes.cl_int.size() * GPGPU.compute.max_scan_block_size;
        long gx = k * GPGPU.compute.max_scan_block_size;
        long[] global_work_size = arg_long(gx);
        int part_size = k * 2;
        long part_buf_size = ((long) CL_DataTypes.cl_int.size() * ((long) part_size));

        var part_data = GPU.CL.new_buffer(GPGPU.compute.context, part_buf_size);

        k_scan_int_multi_block
            .ptr_arg(ScanIntMultiBlock_k.Args.data, data_ptr)
            .loc_arg(ScanIntMultiBlock_k.Args.buffer, local_buffer_size)
            .buf_arg(ScanIntMultiBlock_k.Args.part, part_data)
            .set_arg(ScanIntMultiBlock_k.Args.n, n)
            .call(global_work_size, GPGPU.compute.local_work_default);

        scan_int(part_data.ptr(), part_size);

        k_complete_int_multi_block
            .ptr_arg(CompleteIntMultiBlock_k.Args.data, data_ptr)
            .loc_arg(CompleteIntMultiBlock_k.Args.buffer, local_buffer_size)
            .buf_arg(CompleteIntMultiBlock_k.Args.part, part_data)
            .set_arg(CompleteIntMultiBlock_k.Args.n, n)
            .call(global_work_size, GPGPU.compute.local_work_default);

        part_data.release();
    }

    public void release()
    {
        p_scan_int_array.release();
    }
}
