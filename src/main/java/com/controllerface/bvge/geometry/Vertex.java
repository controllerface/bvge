package com.controllerface.bvge.geometry;

public record Vertex(float x, float y)
{
    public Vertex uniform_scale(float s)
    {
        return new Vertex(x * s, y * s);
    }

    public Vertex translate(float tx, float ty)
    {
        return new Vertex(x + tx, y + ty);
    }

    public double angle(Vertex point)
    {
        return angle(point.x(), point.y());
    }

    public double angle(double x, double y)
    {
        final double ax = x();
        final double ay = y();

        final double delta = (ax * x + ay * y) / Math.sqrt((ax * ax + ay * ay) * (x * x + y * y));

        if (delta > 1.0)
        {
            return 0.0;
        }
        if (delta < -1.0)
        {
            return 180.0;
        }

        return Math.toDegrees(Math.acos(delta));
    }

    public double angle(Vertex p1, Vertex p2)
    {
        final double x = x();
        final double y = y();

        final double ax = p1.x() - x;
        final double ay = p1.y() - y;
        final double bx = p2.x() - x;
        final double by = p2.y() - y;

        final double delta = (ax * bx + ay * by) / Math.sqrt((ax * ax + ay * ay) * (bx * bx + by * by));

        if (delta > 1.0)
        {
            return 0.0;
        }
        if (delta < -1.0)
        {
            return 180.0;
        }

        return Math.toDegrees(Math.acos(delta));
    }
}
