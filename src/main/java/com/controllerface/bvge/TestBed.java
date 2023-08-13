package com.controllerface.bvge;

import org.jocl.*;
import org.joml.Matrix4f;
import org.joml.Vector4f;

import java.util.Arrays;

import static org.jocl.CL.*;

public class TestBed
{
    private static final float[] indices1 =
        {
            22.4f, 28.3f, 27.6f, 61.7f,
            58.3f, 31.5f, 48.2f, 47.6f,
            63.6f, 75.1f, 21.5f, 34.5f,
            29.1f, 36.2f, 64.7f, 28.2f,
        };

    private static final float[] indices2 =
        {
            63.6f, 75.1f, 21.5f, 74.5f,
            58.3f, 61.5f, 34.7f, 47.6f,
            39.1f, 36.2f, 35.8f, 28.2f,
            22.4f, 78.3f, 27.6f, 51.7f,
        };

    private static final float[] vector =
        {
            74.2f, 22.0f, 0f, 0f
        };

    public static void main(String args[])
    {
        Matrix4f test1 = new Matrix4f();
        Matrix4f test2 = new Matrix4f();
        Vector4f test_vector =  new Vector4f();

        test_vector.set(vector);
        test1.set(indices1);
        test2.set(indices2);

        System.out.println(test_vector);
        test1.transform(test_vector);
        System.out.println(test_vector);

        System.out.println(Arrays.toString(vector));
        var r = matrix_transform(indices1, vector);
        System.out.println(Arrays.toString(r));
    }


    private static float[] matrix_transform(float[] matrix, float[] vector)
    {
        float[] result = new float[4];
        result[0] = matrix[0] * vector[0] + matrix[4] * vector[1] + matrix[8] * vector[2] + matrix[12] * vector[3];
        result[1] = matrix[1] * vector[0] + matrix[5] * vector[1] + matrix[9] * vector[2] + matrix[13] * vector[3];
        result[2] = matrix[2] * vector[0] + matrix[6] * vector[1] + matrix[10] * vector[2] + matrix[14] * vector[3];
        result[3] = matrix[3] * vector[0] + matrix[7] * vector[1] + matrix[11] * vector[2] + matrix[15] * vector[3];
        return result;
    }
}
