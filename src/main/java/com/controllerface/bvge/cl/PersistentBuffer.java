package com.controllerface.bvge.cl;

import com.controllerface.bvge.cl.kernels.BufferTransfer_k;
import com.controllerface.bvge.cl.kernels.VerifyBufferTransfer_k;
import com.controllerface.bvge.cl.programs.BufferTransfer;

public class PersistentBuffer extends ResizableBuffer
{
    private final GPUProgram buffer_transfer = new BufferTransfer();
    private final GPUKernel buffer_transfer_k;
    private final GPUKernel verify_buffer_transfer_k;


    public PersistentBuffer(int item_size, long item_capacity)
    {
        super(item_size, item_capacity);
        buffer_transfer.init();
        buffer_transfer_k = new BufferTransfer_k(GPGPU.command_queue_ptr, buffer_transfer.kernel_ptr(Kernel.buffer_transfer));
        verify_buffer_transfer_k = new VerifyBufferTransfer_k(GPGPU.command_queue_ptr, buffer_transfer.kernel_ptr(Kernel.verify_buffer_transfer));
        clear();
    }

    public PersistentBuffer(int item_size)
    {
        super(item_size);
        buffer_transfer.init();
        buffer_transfer_k = new BufferTransfer_k(GPGPU.command_queue_ptr, buffer_transfer.kernel_ptr(Kernel.buffer_transfer));
        verify_buffer_transfer_k = new VerifyBufferTransfer_k(GPGPU.command_queue_ptr, buffer_transfer.kernel_ptr(Kernel.verify_buffer_transfer));
        clear();
    }

    public void ensure_total_capacity(long total_item_capacity)
    {
        if (total_item_capacity > 64415 && total_item_capacity < 64420)
        {
            System.out.println("ensuring: " + total_item_capacity);
        }
        var required_capacity = item_size * total_item_capacity;
        if (required_capacity <= this.byte_capacity) return;

        long previous_capacity = this.byte_capacity;

        while (this.byte_capacity < required_capacity)
        {
            this.byte_capacity += (long)this.item_size * 256L;
        }

        System.out.println("Resizing to: " + byte_capacity + " from: " + previous_capacity + " size: " + item_size);

        long new_pointer = GPGPU.cl_new_buffer(this.byte_capacity);
        GPGPU.cl_zero_buffer(new_pointer, this.byte_capacity);

        long x = previous_capacity / this.item_size;
        System.out.println("global size: " + x);

        buffer_transfer_k
            .ptr_arg(BufferTransfer_k.Args.input, this.pointer)
            .ptr_arg(BufferTransfer_k.Args.output, new_pointer)
            .call(CLUtils.arg_long(x));

        verify_buffer_transfer_k
            .ptr_arg(BufferTransfer_k.Args.input, this.pointer)
            .ptr_arg(BufferTransfer_k.Args.output, new_pointer)
            .call(CLUtils.arg_long(x));

        release();
        this.pointer = new_pointer;
        update_registered_kernels();
    }

//    @Override
//    long resize(long previous_capacity)
//    {
//        System.out.println("Resizing to: " + this.byte_capacity + " from: " + previous_capacity);
//        long new_pointer = GPGPU.cl_new_buffer(this.byte_capacity);
//        GPGPU.cl_transfer_buffer(this.pointer, new_pointer, previous_capacity);
//        release();
//        return new_pointer;
//    }
}
