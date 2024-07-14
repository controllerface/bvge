package com.controllerface.bvge.cl;

import com.controllerface.bvge.cl.buffers.Destoryable;
import com.controllerface.bvge.cl.kernels.*;
import com.controllerface.bvge.cl.programs.GPUProgram;
import com.controllerface.bvge.cl.programs.ScanIntArrayOut;
import com.controllerface.bvge.editor.Editor;

import static com.controllerface.bvge.cl.CLUtils.arg_long;

public class GPUScanScalarIntOut implements Destoryable
{
    private final GPUProgram p_scan_int_array_out = new ScanIntArrayOut();

    private final GPUKernel k_scan_int_single_block_out;
    private final GPUKernel k_scan_int_multi_block_out;
    private final GPUKernel k_complete_int_multi_block_out;

    private final GPUScanScalarInt gpu_int_scan;

    private final boolean own_int_scan;

    private GPUScanScalarIntOut(boolean own_int_scan, long ptr_queue, GPUScanScalarInt gpu_int_scan)
    {
        this.own_int_scan = own_int_scan;
        this.gpu_int_scan = gpu_int_scan;
        p_scan_int_array_out.init();

        long k_ptr_scan_int_array_out_single = p_scan_int_array_out.kernel_ptr(Kernel.scan_int_single_block_out);
        long k_ptr_scan_int_array_out_multi = p_scan_int_array_out.kernel_ptr(Kernel.scan_int_multi_block_out);
        long k_ptr_scan_int_array_out_comp = p_scan_int_array_out.kernel_ptr(Kernel.complete_int_multi_block_out);
        k_scan_int_single_block_out = new ScanIntSingleBlockOut_k(ptr_queue, k_ptr_scan_int_array_out_single);
        k_scan_int_multi_block_out = new ScanIntMultiBlockOut_k(ptr_queue, k_ptr_scan_int_array_out_multi);
        k_complete_int_multi_block_out = new CompleteIntMultiBlockOut_k(ptr_queue, k_ptr_scan_int_array_out_comp);
    }

    public GPUScanScalarIntOut(long ptr_queue, GPUScanScalarInt gpu_int_scan)
    {
        this(false, ptr_queue, gpu_int_scan);
    }

    public GPUScanScalarIntOut(long ptr_queue)
    {
        this(true, ptr_queue, new GPUScanScalarInt(ptr_queue));
    }

    public void scan_int_out(long data_ptr, long o_data_ptr, int n)
    {
        long s = Editor.ACTIVE ? System.nanoTime() : 0;

        int k = GPGPU.work_group_count(n);
        if (k == 1)
        {
            scan_single_block_int_out(data_ptr, o_data_ptr, n);
        }
        else
        {
            scan_multi_block_int_out(data_ptr, o_data_ptr, n, k);
        }

        if (Editor.ACTIVE)
        {
            long e = System.nanoTime() - s;
            Editor.queue_event("render_model_scan_int_out", String.valueOf(e));
        }
    }

    private void scan_single_block_int_out(long data_ptr, long o_data_ptr, int n)
    {
        long local_buffer_size = CLData.cl_int.size() * GPGPU.max_scan_block_size;

        k_scan_int_single_block_out
            .ptr_arg(ScanIntSingleBlockOut_k.Args.input, data_ptr)
            .ptr_arg(ScanIntSingleBlockOut_k.Args.output, o_data_ptr)
            .loc_arg(ScanIntSingleBlockOut_k.Args.buffer, local_buffer_size)
            .set_arg(ScanIntSingleBlockOut_k.Args.n, n)
            .call(GPGPU.local_work_default, GPGPU.local_work_default);
    }

    private void scan_multi_block_int_out(long data_ptr, long o_data_ptr, int n, int k)
    {
        long local_buffer_size = CLData.cl_int.size() * GPGPU.max_scan_block_size;
        long gx = k * GPGPU.max_scan_block_size;
        long[] global_work_size = arg_long(gx);
        int part_size = k * 2;
        long part_buf_size = ((long) CLData.cl_int.size() * ((long) part_size));
        var part_data = GPGPU.cl_new_buffer(part_buf_size);

        k_scan_int_multi_block_out
            .ptr_arg(ScanIntMultiBlockOut_k.Args.input, data_ptr)
            .ptr_arg(ScanIntMultiBlockOut_k.Args.output, o_data_ptr)
            .loc_arg(ScanIntMultiBlockOut_k.Args.buffer, local_buffer_size)
            .ptr_arg(ScanIntMultiBlockOut_k.Args.part, part_data)
            .set_arg(ScanIntMultiBlockOut_k.Args.n, n)
            .call(global_work_size, GPGPU.local_work_default);

        gpu_int_scan.scan_int(part_data, part_size);

        k_complete_int_multi_block_out
            .ptr_arg(CompleteIntMultiBlockOut_k.Args.output, o_data_ptr)
            .loc_arg(CompleteIntMultiBlockOut_k.Args.buffer, local_buffer_size)
            .ptr_arg(CompleteIntMultiBlockOut_k.Args.part, part_data)
            .set_arg(CompleteIntMultiBlockOut_k.Args.n, n)
            .call(global_work_size, GPGPU.local_work_default);

        GPGPU.cl_release_buffer(part_data);
    }

    public void destroy()
    {
        p_scan_int_array_out.destroy();
        if (own_int_scan) gpu_int_scan.destroy();
    }
}
