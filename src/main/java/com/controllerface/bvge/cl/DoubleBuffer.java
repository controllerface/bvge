package com.controllerface.bvge.cl;

class DoubleBuffer<T>
{
    private final T front;
    private final T back;

    private boolean flipped = false;

    DoubleBuffer(T front, T back)
    {
        this.front = front;
        this.back = back;
    }

    public static <T> DoubleBuffer<T> from(T front, T back)
    {
        return new DoubleBuffer<>(front, back);
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
}
