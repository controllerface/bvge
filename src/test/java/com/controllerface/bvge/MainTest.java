package com.controllerface.bvge;

import com.controllerface.bvge.cl.OCLFunctions;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.util.Arrays;

import static com.controllerface.bvge.Main.Memory.*;

class MainTest
{
    private static final int body_width = 16;
    private static final int point_width = 4;
    private static final int bounds_width = 8;
//    private static float[] body_buffer = new float[2 * body_width]; // 2 bodies, 16 floats each
//    private static float[] point_buffer = new float[8 * point_width]; // 8 vertices, 4 floats each
//    private static float[] bounds_buffer = new float[2 * bounds_width]; // 2 bounding boxes, 8 floats each

    @BeforeAll
    public static void setup()
    {
        // point 1
        point_buffer[0] = 45f;               // pos x
        point_buffer[1] = 45f;               // pos y
        point_buffer[2] = point_buffer[0];   // prv x
        point_buffer[3] = point_buffer[1];   // prv y
        // point 2
        point_buffer[4] = 55f;               // pos x
        point_buffer[5] = 45f;               // pos y
        point_buffer[6] = point_buffer[4];   // prv x
        point_buffer[7] = point_buffer[5];   // prv y
        // point 3
        point_buffer[8] = 55f;               // pos x
        point_buffer[9] = 55f;               // pos y
        point_buffer[10] = point_buffer[8];  // prv x
        point_buffer[11] = point_buffer[9];  // prv y
        // point 4
        point_buffer[12] = 45f;              // pos x
        point_buffer[13] = 55f;              // pos y
        point_buffer[14] = point_buffer[12]; // prv x
        point_buffer[15] = point_buffer[13]; // prv y


        // point 5
        point_buffer[16] = 145f;             // pos x
        point_buffer[17] = 145f;             // pos y
        point_buffer[18] = point_buffer[16]; // prv x
        point_buffer[19] = point_buffer[17]; // prv y

        // point 6
        point_buffer[20] = 155f;             // pos x
        point_buffer[21] = 145f;             // pos y
        point_buffer[22] = point_buffer[20]; // prv x
        point_buffer[23] = point_buffer[21]; // prv y

        // point 7
        point_buffer[24] = 155f;             // pos x
        point_buffer[25] = 155f;             // pos y
        point_buffer[26] = point_buffer[24]; // prv x
        point_buffer[27] = point_buffer[25]; // prv y

        // point 8
        point_buffer[28] = 145f;             // pos x
        point_buffer[29] = 155f;             // pos y
        point_buffer[30] = point_buffer[28]; // prv x
        point_buffer[31] = point_buffer[29]; // prv y


        // body 1
        body_buffer[0]  = 0f;        // pos x
        body_buffer[1]  = 0f;        // pos y
        body_buffer[2]  = 10f;       // scale w
        body_buffer[3]  = 10f;       // scale h
        body_buffer[4]  = 1f;        // acc x
        body_buffer[5]  = 0f;        // acc y
        body_buffer[6]  = (float) 0; // bounds x
        body_buffer[7]  = (float) 0; // point index start (int cast);
        body_buffer[8]  = (float) 3; // point index end (int cast);
        body_buffer[9]  = (float) 0; // edge index start (int cast)
        body_buffer[10] = (float) 0; // edge index end (int cast)
        // empty for now, but needed p2 pad out float16 data type
        body_buffer[11] = 0f;
        body_buffer[12] = 0f;
        body_buffer[13] = 0f;
        body_buffer[14] = 0f;
        body_buffer[15] = 0f;


        // body 2
        body_buffer[16] = 0f;        // pos x
        body_buffer[17] = 0f;        // pos y
        body_buffer[18] = 10f;       // scale w
        body_buffer[19] = 10f;       // scale h
        body_buffer[20] = 0f;        // acc x
        body_buffer[21] = 0f;        // acc y
        body_buffer[22] = (float) 1; // bounds index (int cast)
        body_buffer[23] = (float) 4; // point index start (int cast);
        body_buffer[24] = (float) 7; // point index end (int cast);
        body_buffer[25] = (float) 0; // edge index start (int cast)
        body_buffer[26] = (float) 0; // edge index end (int cast)
        // empty for now, but needed p2 pad out float16 data type
        body_buffer[27] = 0f;
        body_buffer[28] = 0f;
        body_buffer[29] = 0f;
        body_buffer[30] = 0f;
        body_buffer[31] = 0f;


        OCLFunctions.init();
    }

    @AfterAll
    public static void tearDown()
    {
        OCLFunctions.destroy();
    }

    @Test
    public void test1()
    {
        System.out.println("Before:");
        System.out.println(Arrays.toString(body_buffer));
        System.out.println(Arrays.toString(point_buffer));
        System.out.println(Arrays.toString(bounds_buffer));
        //OCLFunctions.integrate(1f/60f, 100, 100);
        System.out.println("After:");
        System.out.println(Arrays.toString(body_buffer));
        System.out.println(Arrays.toString(point_buffer));
        System.out.println(Arrays.toString(bounds_buffer));
    }
}