package com.controllerface.bvge.cl;

import com.controllerface.bvge.cl.buffers.Destoryable;
import com.controllerface.bvge.cl.kernels.*;
import com.controllerface.bvge.cl.programs.GPUProgram;
import com.controllerface.bvge.cl.programs.ScanIntArray;
import com.controllerface.bvge.editor.Editor;

import static com.controllerface.bvge.cl.CLUtils.arg_long;

public class GPUScanScalarInt implements Destoryable
{
    private final GPUProgram p_scan_int_array = new ScanIntArray();

    private final GPUKernel k_scan_int_single_block;
    private final GPUKernel k_scan_int_multi_block;
    private final GPUKernel k_complete_int_multi_block;

    public GPUScanScalarInt(long ptr_queue)
    {
        p_scan_int_array.init();

        long k_ptr_scan_int_array_single = p_scan_int_array.kernel_ptr(Kernel.scan_int_single_block);
        long k_ptr_scan_int_array_multi  = p_scan_int_array.kernel_ptr(Kernel.scan_int_multi_block);
        long k_ptr_scan_int_array_comp   = p_scan_int_array.kernel_ptr(Kernel.complete_int_multi_block);
        k_scan_int_single_block          = new ScanIntSingleBlock_k(ptr_queue, k_ptr_scan_int_array_single);
        k_scan_int_multi_block           = new ScanIntMultiBlock_k(ptr_queue, k_ptr_scan_int_array_multi);
        k_complete_int_multi_block       = new CompleteIntMultiBlock_k(ptr_queue, k_ptr_scan_int_array_comp);
    }

    public void scan_int(long data_ptr, int n)
    {
        long s = Editor.ACTIVE ? System.nanoTime() : 0;

        int k = GPGPU.work_group_count(n);
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
        long local_buffer_size = CLData.cl_int.size() * GPGPU.max_scan_block_size;

        k_scan_int_single_block
            .ptr_arg(ScanIntSingleBlock_k.Args.data, data_ptr)
            .loc_arg(ScanIntSingleBlock_k.Args.buffer, local_buffer_size)
            .set_arg(ScanIntSingleBlock_k.Args.n, n)
            .call(GPGPU.local_work_default, GPGPU.local_work_default);
    }

    private void scan_multi_block_int(long data_ptr, int n, int k)
    {
        long local_buffer_size = CLData.cl_int.size() * GPGPU.max_scan_block_size;
        long gx = k * GPGPU.max_scan_block_size;
        long[] global_work_size = arg_long(gx);
        int part_size = k * 2;
        long part_buf_size = ((long) CLData.cl_int.size() * ((long) part_size));

        var part_data = GPGPU.cl_new_buffer(part_buf_size);

        k_scan_int_multi_block
            .ptr_arg(ScanIntMultiBlock_k.Args.data, data_ptr)
            .loc_arg(ScanIntMultiBlock_k.Args.buffer, local_buffer_size)
            .ptr_arg(ScanIntMultiBlock_k.Args.part, part_data)
            .set_arg(ScanIntMultiBlock_k.Args.n, n)
            .call(global_work_size, GPGPU.local_work_default);

        scan_int(part_data, part_size);

        k_complete_int_multi_block
            .ptr_arg(CompleteIntMultiBlock_k.Args.data, data_ptr)
            .loc_arg(CompleteIntMultiBlock_k.Args.buffer, local_buffer_size)
            .ptr_arg(CompleteIntMultiBlock_k.Args.part, part_data)
            .set_arg(CompleteIntMultiBlock_k.Args.n, n)
            .call(global_work_size, GPGPU.local_work_default);

        GPGPU.cl_release_buffer(part_data);
    }

    public void destroy()
    {
        p_scan_int_array.destroy();
    }
}
