package com.controllerface.bvge.gpu.cl.programs.scan;

import com.controllerface.bvge.editor.Editor;
import com.controllerface.bvge.gpu.GPU;
import com.controllerface.bvge.gpu.GPUResource;
import com.controllerface.bvge.gpu.cl.buffers.CL_DataTypes;
import com.controllerface.bvge.gpu.cl.contexts.CL_CommandQueue;
import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;
import com.controllerface.bvge.gpu.cl.kernels.scan.CompleteIntMultiBlockOut_k;
import com.controllerface.bvge.gpu.cl.kernels.scan.ScanIntMultiBlockOut_k;
import com.controllerface.bvge.gpu.cl.kernels.scan.ScanIntSingleBlockOut_k;
import com.controllerface.bvge.gpu.cl.programs.GPUProgram;

import static com.controllerface.bvge.gpu.GPU.CL.arg_long;

public class GPUScanScalarIntOut implements GPUResource
{
    private final GPUProgram p_scan_int_array_out = new ScanIntArrayOut();

    private final GPUKernel k_scan_int_single_block_out;
    private final GPUKernel k_scan_int_multi_block_out;
    private final GPUKernel k_complete_int_multi_block_out;

    private final GPUScanScalarInt gpu_int_scan;

    private final boolean own_int_scan;

    private GPUScanScalarIntOut(boolean own_int_scan, CL_CommandQueue ptr_queue, GPUScanScalarInt gpu_int_scan)
    {
        this.own_int_scan = own_int_scan;
        this.gpu_int_scan = gpu_int_scan;
        p_scan_int_array_out.init();

        k_scan_int_single_block_out = new ScanIntSingleBlockOut_k(ptr_queue, p_scan_int_array_out);
        k_scan_int_multi_block_out = new ScanIntMultiBlockOut_k(ptr_queue, p_scan_int_array_out);
        k_complete_int_multi_block_out = new CompleteIntMultiBlockOut_k(ptr_queue, p_scan_int_array_out);
    }

    public GPUScanScalarIntOut(CL_CommandQueue cmd_queue, GPUScanScalarInt gpu_int_scan)
    {
        this(false, cmd_queue, gpu_int_scan);
    }

    public GPUScanScalarIntOut(CL_CommandQueue cmd_queue)
    {
        this(true, cmd_queue, new GPUScanScalarInt(cmd_queue));
    }

    public void scan_int_out(long data_ptr, long o_data_ptr, int n)
    {
        long s = Editor.ACTIVE ? System.nanoTime() : 0;

        int k = GPU.compute.work_group_count(n);
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
        long local_buffer_size = CL_DataTypes.cl_int.size() * GPU.compute.max_scan_block_size;

        k_scan_int_single_block_out
            .ptr_arg(ScanIntSingleBlockOut_k.Args.input, data_ptr)
            .ptr_arg(ScanIntSingleBlockOut_k.Args.output, o_data_ptr)
            .loc_arg(ScanIntSingleBlockOut_k.Args.buffer, local_buffer_size)
            .set_arg(ScanIntSingleBlockOut_k.Args.n, n)
            .call(GPU.compute.local_work_default, GPU.compute.local_work_default);
    }

    private void scan_multi_block_int_out(long data_ptr, long o_data_ptr, int n, int k)
    {
        long local_buffer_size = CL_DataTypes.cl_int.size() * GPU.compute.max_scan_block_size;
        long gx = k * GPU.compute.max_scan_block_size;
        long[] global_work_size = arg_long(gx);
        int part_size = k * 2;
        long part_buf_size = ((long) CL_DataTypes.cl_int.size() * ((long) part_size));
        var part_data = GPU.CL.new_buffer(GPU.compute.context, part_buf_size);

        k_scan_int_multi_block_out
            .ptr_arg(ScanIntMultiBlockOut_k.Args.input, data_ptr)
            .ptr_arg(ScanIntMultiBlockOut_k.Args.output, o_data_ptr)
            .loc_arg(ScanIntMultiBlockOut_k.Args.buffer, local_buffer_size)
            .buf_arg(ScanIntMultiBlockOut_k.Args.part, part_data)
            .set_arg(ScanIntMultiBlockOut_k.Args.n, n)
            .call(global_work_size, GPU.compute.local_work_default);

        gpu_int_scan.scan_int(part_data.ptr(), part_size);

        k_complete_int_multi_block_out
            .ptr_arg(CompleteIntMultiBlockOut_k.Args.output, o_data_ptr)
            .loc_arg(CompleteIntMultiBlockOut_k.Args.buffer, local_buffer_size)
            .buf_arg(CompleteIntMultiBlockOut_k.Args.part, part_data)
            .set_arg(CompleteIntMultiBlockOut_k.Args.n, n)
            .call(global_work_size, GPU.compute.local_work_default);

        part_data.release();
    }

    public void release()
    {
        p_scan_int_array_out.release();
        if (own_int_scan) gpu_int_scan.release();
    }
}
