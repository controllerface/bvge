package com.controllerface.bvge.cl;

import com.controllerface.bvge.cl.buffers.Destoryable;

class FlippableContainer<T extends Destoryable> implements Destoryable
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
    public void destroy()
    {
        front.destroy();
        back.destroy();
    }
}
