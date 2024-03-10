package com.controllerface.bvge.cl.kernels;

import com.controllerface.bvge.cl.GPUKernel;

public class CreateKeyFrame_k extends GPUKernel
{
    public enum Args
    {
        key_frames,
        frame_times,
        target,
        new_keyframe,
        new_frame_time;
    }

    public CreateKeyFrame_k(long command_queue_ptr, long kernel_ptr)
    {
        super(command_queue_ptr, kernel_ptr);
    }
}
