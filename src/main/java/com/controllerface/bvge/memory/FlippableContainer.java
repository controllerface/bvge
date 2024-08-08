package com.controllerface.bvge.memory;

import com.controllerface.bvge.gpu.GPUResource;

public class FlippableContainer<T extends GPUResource> implements GPUResource
{
    private final T front;
    private final T back;

    private boolean flipped = false;

    FlippableContainer(T front, T back)
    {
        this.front = front;
        this.back = back;
    }

    public void flip()
    {
        flipped = !flipped;
    }

    public T front()
    {
        return flipped
            ? back
            : front;
    }

    public T back()
    {
        return flipped
            ? front
            : back;
    }

    @Override
    public void release()
    {
        front.release();
        back.release();
    }
}
