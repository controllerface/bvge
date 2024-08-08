package com.controllerface.bvge.gpu.cl;

import com.controllerface.bvge.editor.Editor;
import com.controllerface.bvge.gpu.GPUResource;
import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;
import com.controllerface.bvge.gpu.cl.kernels.KernelType;
import com.controllerface.bvge.gpu.cl.kernels.scan.CompleteInt2MultiBlock_k;
import com.controllerface.bvge.gpu.cl.kernels.scan.ScanInt2MultiBlock_k;
import com.controllerface.bvge.gpu.cl.kernels.scan.ScanInt2SingleBlock_k;
import com.controllerface.bvge.gpu.cl.programs.GPUProgram;
import com.controllerface.bvge.gpu.cl.programs.ScanInt2Array;

import static com.controllerface.bvge.gpu.GPU.CL.arg_long;

public class GPUScanVectorInt2 implements GPUResource
{
    private final GPUProgram p_scan_int2_array = new ScanInt2Array();

    private final GPUKernel k_scan_int2_single_block;
    private final GPUKernel k_scan_int2_multi_block;
    private final GPUKernel k_complete_int2_multi_block;

    public GPUScanVectorInt2(long ptr_queue)
    {
        p_scan_int2_array.init();

        long k_ptr_scan_int2_array_single = p_scan_int2_array.kernel_ptr(KernelType.scan_int2_single_block);
        long k_ptr_scan_int2_array_multi = p_scan_int2_array.kernel_ptr(KernelType.scan_int2_multi_block);
        long k_ptr_scan_int2_array_comp = p_scan_int2_array.kernel_ptr(KernelType.complete_int2_multi_block);
        k_scan_int2_single_block = new ScanInt2SingleBlock_k(ptr_queue, k_ptr_scan_int2_array_single);
        k_scan_int2_multi_block = new ScanInt2MultiBlock_k(ptr_queue, k_ptr_scan_int2_array_multi);
        k_complete_int2_multi_block = new CompleteInt2MultiBlock_k(ptr_queue, k_ptr_scan_int2_array_comp);
    }

    public void scan_int2(long data_ptr, int n)
    {
        long s = Editor.ACTIVE ? System.nanoTime() : 0;

        int k = GPGPU.work_group_count(n);
        if (k == 1)
        {
            scan_single_block_int2(data_ptr, n);
        }
        else
        {
            scan_multi_block_int2(data_ptr, n, k);
        }

        if (Editor.ACTIVE)
        {
            long e = System.nanoTime() - s;
            Editor.queue_event("render_model_scan_int2", String.valueOf(e));
        }
    }

    private void scan_single_block_int2(long data_ptr, int n)
    {
        long local_buffer_size = CL_DataTypes.cl_int2.size() * GPGPU.max_scan_block_size;

        k_scan_int2_single_block
            .ptr_arg(ScanInt2SingleBlock_k.Args.data, data_ptr)
            .loc_arg(ScanInt2SingleBlock_k.Args.buffer, local_buffer_size)
            .set_arg(ScanInt2SingleBlock_k.Args.n, n)
            .call(GPGPU.local_work_default, GPGPU.local_work_default);
    }

    private void scan_multi_block_int2(long data_ptr, int n, int k)
    {
        long local_buffer_size = CL_DataTypes.cl_int2.size() * GPGPU.max_scan_block_size;
        long gx = k * GPGPU.max_scan_block_size;
        long[] global_work_size = arg_long(gx);
        int part_size = k * 2;
        long part_buf_size = ((long) CL_DataTypes.cl_int2.size() * ((long) part_size));

        var part_data = GPGPU.cl_new_buffer(part_buf_size);

        k_scan_int2_multi_block
            .ptr_arg(ScanInt2MultiBlock_k.Args.data, data_ptr)
            .loc_arg(ScanInt2MultiBlock_k.Args.buffer, local_buffer_size)
            .ptr_arg(ScanInt2MultiBlock_k.Args.part, part_data)
            .set_arg(ScanInt2MultiBlock_k.Args.n, n)
            .call(global_work_size, GPGPU.local_work_default);

        scan_int2(part_data, part_size);

        k_complete_int2_multi_block
            .ptr_arg(CompleteInt2MultiBlock_k.Args.data, data_ptr)
            .loc_arg(CompleteInt2MultiBlock_k.Args.buffer, local_buffer_size)
            .ptr_arg(CompleteInt2MultiBlock_k.Args.part, part_data)
            .set_arg(CompleteInt2MultiBlock_k.Args.n, n)
            .call(global_work_size, GPGPU.local_work_default);

        GPGPU.cl_release_buffer(part_data);
    }

    public void release()
    {
        p_scan_int2_array.release();
    }
}
