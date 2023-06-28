package com.controllerface.bvge.ecs.systems.physics;

public class BoxKey
{
    public final int x;
    public final int y;

    public BoxKey(int x, int y)
    {
        this.x = x;
        this.y = y;
    }

    @Override
    public String toString()
    {
        return "[x:" + x + ", y:" + y + "]";
    }

    @Override
    public boolean equals(Object o)
    {
        if (this == o)
        {
            return true;
        }
        if (o == null || getClass() != o.getClass())
        {
            return false;
        }

        BoxKey boxKey = (BoxKey) o;

        if (x != boxKey.x)
        {
            return false;
        }
        return y == boxKey.y;
    }

    @Override
    public int hashCode()
    {
        int result = x;
        result = 31 * result + y;
        return result;
    }
}
