package com.controllerface.bvge;

import com.controllerface.bvge.cl.OpenCL;
import com.controllerface.bvge.ecs.systems.physics.SpatialMap;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

class MainTest
{
    // all the vertices that are stored
    private static float[] vertex_pos_buffer = new float[16];

    // all the previous positions for each vertex stored, aligned to the pos buffer
    private static float[] vertex_prv_buffer = new float[16];

    // holds the start and end index of the vertices that define a body, 2 entries per body
    private static int[] body_index_buffer = new int[4];

    // holds the x,y of the body buffer, aligns with the body index buffer
    private static float[] body_pos_buffer = new float[4];

    // holds the acceleration vector of the object, aligned with the body pos buffer
    private static float[] body_acc_buffer = new float[4];

    // holds bounding boxes for bodies. Aligned with the body index buffer, but double wide (4 floats each)
    private static float[] body_bounds_buffer = new float[8];

    private static int body_count = 2;

    @BeforeAll
    public static void setup()
    {
        // body 1, 4 vertices (1 quad, 2 tris)
        vertex_pos_buffer[0] = 0.0f; // vertex 1 x
        vertex_pos_buffer[1] = 0.0f; // vertex 1 y
        vertex_pos_buffer[2] = 10.0f; // .. x
        vertex_pos_buffer[3] = 0.0f; // .. y
        vertex_pos_buffer[4] = 10.0f;
        vertex_pos_buffer[5] = 10.0f;
        vertex_pos_buffer[6] = 0.0f;
        vertex_pos_buffer[7] = 10.0f;

        // body 2
        vertex_pos_buffer[8] = 20.0f;
        vertex_pos_buffer[9] = 20.0f;
        vertex_pos_buffer[10] = 30.0f;
        vertex_pos_buffer[11] = 20.0f;
        vertex_pos_buffer[12] = 30.0f;
        vertex_pos_buffer[13] = 30.0f;
        vertex_pos_buffer[14] = 20.0f;
        vertex_pos_buffer[15] = 30.0f;


        // set all previous positions to the current position for 0 movement
        vertex_prv_buffer[0] = vertex_pos_buffer[0]; // vertex 1 x
        vertex_prv_buffer[1] = vertex_pos_buffer[1]; // vertex 1 y
        vertex_prv_buffer[2] = vertex_pos_buffer[2]; // .. x
        vertex_prv_buffer[3] = vertex_pos_buffer[3]; // .. y
        vertex_prv_buffer[4] = vertex_pos_buffer[4];
        vertex_prv_buffer[5] = vertex_pos_buffer[5];
        vertex_prv_buffer[6] = vertex_pos_buffer[6];
        vertex_prv_buffer[7] = vertex_pos_buffer[7];
        vertex_prv_buffer[8]  = vertex_pos_buffer[8];
        vertex_prv_buffer[9]  = vertex_pos_buffer[9];
        vertex_prv_buffer[10] = vertex_pos_buffer[10];
        vertex_prv_buffer[11] = vertex_pos_buffer[11];
        vertex_prv_buffer[12] = vertex_pos_buffer[12];
        vertex_prv_buffer[13] = vertex_pos_buffer[13];
        vertex_prv_buffer[14] = vertex_pos_buffer[14];
        vertex_prv_buffer[15] = vertex_pos_buffer[15];


        // leave these empty, they should be recalculated during execution
        body_pos_buffer[0] = 0.0f; // x body 1
        body_pos_buffer[1] = 0.0f; // y
        body_pos_buffer[2] = 0.0f; // x body 2
        body_pos_buffer[3] = 0.0f; // y

        body_acc_buffer[0] = 0.0f; // x body 1
        body_acc_buffer[1] = 0.0f; // y
        body_acc_buffer[2] = 0.0f; // x body 2
        body_acc_buffer[3] = 0.0f; // y

        // bounding box for 1st body
        body_bounds_buffer[0] = 0.0f; // x
        body_bounds_buffer[1] = 0.0f; // y
        body_bounds_buffer[2] = 0.0f; // width
        body_bounds_buffer[3] = 0.0f; // height

        // bounding box for 2nd body
        body_bounds_buffer[4] = 0.0f;
        body_bounds_buffer[5] = 0.0f;
        body_bounds_buffer[6] = 0.0f;
        body_bounds_buffer[7] = 0.0f;

        // start/end indices for bodies
        body_index_buffer[0] = 0;  // body 1 start
        body_index_buffer[1] = 7;  // body 1 end
        body_index_buffer[2] = 8;  // body 2 start
        body_index_buffer[3] = 15; // body 2 end

        OpenCL.init();
    }

    @AfterAll
    public static void tearDown()
    {
        OpenCL.destroy();
    }

    @Test
    public void test1()
    {
        SpatialMap spatialMap = new SpatialMap();
        float[] displacement = new float[8];

        float dt = 1.0f / 60.0f;

        // in the game, there would be one pre-execution step to add forces for the player, but that is omitted here
        // as it is a test, it would not make sense to parallelize that part

        // first get the displacement from the acc vector (gravity, etc.)
//        OpenCL.displacement(body_acc_buffer, dt, displacement);
//
//        // next, loop through the vertices of each body, and add the displacement
//        // as well as update the pos/prv vectors as needed. the centroid for each body is returned
//        OpenCL.integrate(displacement, vertex_pos_buffer, vertex_prv_buffer, body_pos_buffer);
//
//        // update the bounding boxes, store the result directly in the bounds buffer
//        OpenCL.findBoundingBox(body_index_buffer, vertex_pos_buffer, body_bounds_buffer);

        // update the spatial map with the bounding boxes
        // todo: this

        int[] candidate_pairs = new int[1000]; // todo: use a large buffer and take a subset for the next stage
        int pairCount = 0;
        int pairVertCount = 0;

        // find collisions
        // do broad-phase in cpu to find potential matches, store in candidate_pairs
        int nextCandidateIndex = 0;
        for (int i = 0; i < body_count; i++)
        {
            // update the backing box object while we're already looping, and map the candidate list into
            // the candidate pair array
            int[] matches = new int[10]; // todo: determine length inside detection logic
            // todo: if matches is empty, bail on this one
            for (int j = 0; j < matches.length; j++)
            {
                candidate_pairs[nextCandidateIndex] = i;
                candidate_pairs[nextCandidateIndex + 1] = j;
                nextCandidateIndex += 2;
            }
        }

        float[] candidate_normals;
        // from candidate pairs, calculate matching normals, there is one for each vertex of each object pair



    }
}