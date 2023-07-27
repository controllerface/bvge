package com.controllerface.bvge.util;

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
