package com.controllerface.bvge.gpu.cl.programs.scan;

import com.controllerface.bvge.gpu.GPU;
import com.controllerface.bvge.gpu.GPUResource;
import com.controllerface.bvge.gpu.cl.buffers.CL_DataTypes;
import com.controllerface.bvge.gpu.cl.GPGPU;
import com.controllerface.bvge.gpu.cl.contexts.CL_CommandQueue;
import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;
import com.controllerface.bvge.gpu.cl.kernels.KernelType;
import com.controllerface.bvge.gpu.cl.kernels.scan.CompleteInt4MultiBlock_k;
import com.controllerface.bvge.gpu.cl.kernels.scan.ScanInt4MultiBlock_k;
import com.controllerface.bvge.gpu.cl.kernels.scan.ScanInt4SingleBlock_k;
import com.controllerface.bvge.gpu.cl.programs.GPUProgram;

import static com.controllerface.bvge.gpu.GPU.CL.arg_long;

public class GPUScanVectorInt4 implements GPUResource
{
    private final GPUProgram p_scan_int4_array = new ScanInt4Array();

    private final GPUKernel k_scan_int4_single_block;
    private final GPUKernel k_scan_int4_multi_block;
    private final GPUKernel k_complete_int4_multi_block;

    public GPUScanVectorInt4(CL_CommandQueue cmd_queue)
    {
        p_scan_int4_array.init();

        long k_ptr_scan_int4_array_single = p_scan_int4_array.kernel_ptr(KernelType.scan_int4_single_block);
        long k_ptr_scan_int4_array_multi = p_scan_int4_array.kernel_ptr(KernelType.scan_int4_multi_block);
        long k_ptr_scan_int4_array_comp = p_scan_int4_array.kernel_ptr(KernelType.complete_int4_multi_block);
        k_scan_int4_single_block = new ScanInt4SingleBlock_k(cmd_queue, k_ptr_scan_int4_array_single);
        k_scan_int4_multi_block = new ScanInt4MultiBlock_k(cmd_queue, k_ptr_scan_int4_array_multi);
        k_complete_int4_multi_block = new CompleteInt4MultiBlock_k(cmd_queue, k_ptr_scan_int4_array_comp);
    }

    public void scan_int4(long data_ptr, int n)
    {
        int k = GPGPU.compute.work_group_count(n);
        if (k == 1)
        {
            scan_single_block_int4(data_ptr, n);
        }
        else
        {
            scan_multi_block_int4(data_ptr, n, k);
        }
    }

    private void scan_single_block_int4(long data_ptr, int n)
    {
        long local_buffer_size = CL_DataTypes.cl_int4.size() * GPGPU.compute.max_scan_block_size;

        k_scan_int4_single_block
            .ptr_arg(ScanInt4SingleBlock_k.Args.data, data_ptr)
            .loc_arg(ScanInt4SingleBlock_k.Args.buffer, local_buffer_size)
            .set_arg(ScanInt4SingleBlock_k.Args.n, n)
            .call(GPGPU.compute.local_work_default, GPGPU.compute.local_work_default);
    }

    private void scan_multi_block_int4(long data_ptr, int n, int k)
    {
        long local_buffer_size = CL_DataTypes.cl_int4.size() * GPGPU.compute.max_scan_block_size;
        long gx = k * GPGPU.compute.max_scan_block_size;
        long[] global_work_size = arg_long(gx);
        int part_size = k * 2;
        long part_buf_size = ((long) CL_DataTypes.cl_int4.size() * ((long) part_size));

        var part_data = GPU.CL.new_buffer(GPGPU.compute.context, part_buf_size);

        k_scan_int4_multi_block
            .ptr_arg(ScanInt4MultiBlock_k.Args.data, data_ptr)
            .loc_arg(ScanInt4MultiBlock_k.Args.buffer, local_buffer_size)
            .buf_arg(ScanInt4MultiBlock_k.Args.part, part_data)
            .set_arg(ScanInt4MultiBlock_k.Args.n, n)
            .call(global_work_size, GPGPU.compute.local_work_default);

        scan_int4(part_data.ptr(), part_size);

        k_complete_int4_multi_block
            .ptr_arg(CompleteInt4MultiBlock_k.Args.data, data_ptr)
            .loc_arg(CompleteInt4MultiBlock_k.Args.buffer, local_buffer_size)
            .buf_arg(CompleteInt4MultiBlock_k.Args.part, part_data)
            .set_arg(CompleteInt4MultiBlock_k.Args.n, n)
            .call(global_work_size, GPGPU.compute.local_work_default);

        part_data.release();
    }

    public void release()
    {
        p_scan_int4_array.release();
    }
}
