package com.controllerface.bvge.util;

import com.controllerface.bvge.data.FPoint2D;
import com.controllerface.bvge.ecs.Point2D;
import org.joml.Vector2f;

import java.util.List;

public class MathEX
{
    /**
     * Calculates the centroid, i.e. center point, of a polygon and stores it in the target vector.
     * This method is only tested with convex polygons.
     * @param verts
     * @param target
     */
    public static void centroid(FPoint2D[] verts, Vector2f target)
    {
        var points = verts.length;
        float x_sum = 0;
        float y_sum = 0;
        for (FPoint2D point : verts)
        {
            x_sum += point.pos_x();
            y_sum += point.pos_y();
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
}
