package com.controllerface.bvge.util;

import com.controllerface.bvge.geometry.Vertex;
import org.joml.Vector2f;

public class MathEX
{
    /**
     * Calculates the centroid, i.e. center point, of a polygon and stores it in the target vector.
     *
     * @param target vector buffer to store the result
     * @param vertices the vertices to use for the calculation
     */
    public static void centroid(Vector2f target, float[] ... vertices)
    {
        var points = vertices.length;
        float x_sum = 0;
        float y_sum = 0;
        for (float[] point : vertices)
        {
            x_sum += point[0];
            y_sum += point[1];
        }
        float x = x_sum / points;
        float y = y_sum / points;
        target.x = x;
        target.y = y;
    }

    public static void centroid(Vector2f target, Vertex ... vertices)
    {
        var points = vertices.length;
        float x_sum = 0;
        float y_sum = 0;
        for (Vertex point : vertices)
        {
            x_sum += point.x();
            y_sum += point.y();
        }
        float x = x_sum / points;
        float y = y_sum / points;
        target.x = x;
        target.y = y;
    }

    public static void rotate(Vector2f vec, float angleDeg, Vector2f origin) {
        float x = vec.x - origin.x;
        float y = vec.y - origin.y;

        float cos = (float)Math.cos(Math.toRadians(angleDeg));
        float sin = (float)Math.sin(Math.toRadians(angleDeg));

        float xPrime = (x * cos) - (y * sin);
        float yPrime = (x * sin) + (y * cos);

        xPrime += origin.x;
        yPrime += origin.y;

        vec.x = xPrime;
        vec.y = yPrime;
    }

    public static float degreesToRadians(float degrees)
    {
        return (float)(degrees * (Math.PI / 180));
    }

    public static float radiansToDegrees(float radians)
    {
        return (float)(Math.PI / radians);
    }

    public static float angleBetween2Lines(float[] line1, float[] line2)
    {
        float l1x1 = line1[0];
        float l1y1 = line1[1];
        float l1x2 = line1[2];
        float l1y2 = line1[3];

        float l2x1 = line2[0];
        float l2y1 = line2[1];
        float l2x2 = line2[2];
        float l2y2 = line2[3];

        float angle1 = (float)Math.atan2(l1y1 - l1y2, l1x1 - l1x2);
        float angle2 = (float)Math.atan2(l2y1 - l2y2, l2x1 - l2x2);
        return angle1 - angle2;
    }

    public static boolean compare(float x, float y, float epsilon) {
        return Math.abs(x - y) <= epsilon * Math.max(1.0f, Math.max(Math.abs(x), Math.abs(y)));
    }

    public static boolean compare(Vector2f vec1, Vector2f vec2, float epsilon) {
        return compare(vec1.x, vec2.x, epsilon) && compare(vec1.y, vec2.y, epsilon);
    }

    public static boolean compare(float x, float y) {
        return Math.abs(x - y) <= Float.MIN_VALUE * Math.max(1.0f, Math.max(Math.abs(x), Math.abs(y)));
    }

    public static boolean compare(Vector2f vec1, Vector2f vec2) {
        return compare(vec1.x, vec2.x) && compare(vec1.y, vec2.y);
    }




    public static Vertex uniform_scale(Vertex target, float s)
    {
        return new Vertex(target.x() * s, target.y() * s, target.bone_name(), target.bone_weight());
    }

    public static Vertex translate(Vertex target, float tx, float ty)
    {
        return new Vertex(target.x() + tx, target.y() + ty, target.bone_name(), target.bone_weight());
    }

    public static double angle(Vertex target, Vertex point)
    {
        return angle(target, point.x(), point.y());
    }

    public static double angle(Vertex target, double x, double y)
    {
        final double ax = target.x();
        final double ay = target.y();

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

    public static double angle(Vertex target, Vertex p1, Vertex p2)
    {
        final double x = target.x();
        final double y = target.y();

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
