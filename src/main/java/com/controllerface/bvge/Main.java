package com.controllerface.bvge;

import com.controllerface.bvge.cl.OpenCL;
import com.controllerface.bvge.data.*;
import com.controllerface.bvge.window.Window;


public class Main
{
    // TODO: (large)
    //  The data that is stored here can instead be allocated and stored on the GPU. Before
    //  this can be done, a few things need to happen. First, GPU characteristics need to be
    //  determined programmatically, so max RAM, etc. can be known before the buffers are
    //  created. Once the sizes are better known, kernel code needs to be written that acts
    //  as an interface, similar to a database, where objects can be written/read using the
    //  GPU-side buffer. The end goal is to keep the data resident on the GPU as much as
    //  possible, and only pull values down to the host if needed. The entire physics loop
    //  will then be implemented in GPU code.
    public static class Memory
    {
        // todo: actually add the cleaner/tracking mechanism for keeping track of
        //  all points, bodies, etc. This will be needed when the arrays need to compact
        //  because of this, those objects won't be able to be records.

        // Memory layout notes:
        //---------------------
        // Objects managed by this class are laid out as 1 dimensional arrays for interoperability
        // with OpenGL and OpenCL. Each object type is mapped to a corresponding OpenCL vector type,
        // and when the elements are used within cl kernel code, these arrays are indexed based on
        // their "vector width" within the kernel environment. For example, in the Java environment,
        // an integer array declared like this:
        //     int[] test_array= new int[10];
        // may be accessed in the kernel code as an int2 (i.e. a 2D vector of ints) as an *int2 pointer.
        // This in this scenario, the int[] of size 10, is treated as an int2[] of size 5. I.e., indexing
        // into the array from within kernel code, the maximum index would be 4. The Width class below will
        // contain comments describing the kernel side structure used for each.
        public static class Width
        {
            // float16
            public static final int BODY = 16;
            // Individual objects, including the player are a single body

            // float4
            public static final int POINT = 4;
            // Bodies are composed of one or more points

            // float4
            public static final int EDGE = 4;
            // Edges define constraints that are set between two vertices

            // float8
            public static final int BOUNDS = 16;
            // Bounding boxes are used for proximity checks on bodies

            // int2
            public static final int COLLISION = 2;
            // Collisions are represented as a pair of indices, one for each object
        }

        private static final int MAX_BODIES  = 100_000;
        private static final int MAX_POINTS  = 1_000_000;

        private static final int BODY_BUFFER_SIZE    = Width.BODY    * MAX_BODIES;
        private static final int POINT_BUFFER_SIZE   = Width.POINT   * MAX_POINTS;
        private static final int EDGE_BUFFER_SIZE    = Width.EDGE    * MAX_POINTS;
        private static final int BOUNDS_BUFFER_SIZE  = Width.BOUNDS  * MAX_BODIES;

        public static float[] body_buffer   = new float[BODY_BUFFER_SIZE];
        public static float[] point_buffer  = new float[POINT_BUFFER_SIZE];
        public static float[] edge_buffer   = new float[EDGE_BUFFER_SIZE];
        public static float[] bounds_buffer = new float[BOUNDS_BUFFER_SIZE];

        private static int body_index    = 0;
        private static int point_index   = 0;
        private static int edge_index    = 0;
        private static int bounds_index  = 0;

        public static int bodyCount()
        {
            return body_index / Width.BODY;
        }

        public static int bodyLength()
        {
            return body_index;
        }

        public static int pointLength()
        {
            return point_index + 1;
        }

        public static int boundsLength()
        {
            return bounds_index;
        }

        public static int edgesLength()
        {
            return edge_index;
        }

        public static int edgesCount()
        {
            return edge_index / Width.EDGE;
        }

        public static int boundsCount()
        {
            return bounds_index / Width.BOUNDS;
        }

        public static FBounds2D newBounds()
        {
            bounds_buffer[bounds_index++] = 0f;
            bounds_buffer[bounds_index++] = 0f;
            bounds_buffer[bounds_index++] = 0f;
            bounds_buffer[bounds_index++] = 0f;
            bounds_buffer[bounds_index++] = 0f;
            bounds_buffer[bounds_index++] = 0f;
            bounds_buffer[bounds_index++] = 0f;
            bounds_buffer[bounds_index++] = 0f;
            bounds_buffer[bounds_index++] = 0f;
            bounds_buffer[bounds_index++] = 0f;
            bounds_buffer[bounds_index++] = 0f;
            bounds_buffer[bounds_index++] = 0f;
            bounds_buffer[bounds_index++] = 0f;
            bounds_buffer[bounds_index++] = 0f;
            bounds_buffer[bounds_index++] = 0f;
            bounds_buffer[bounds_index++] = 0f;
            return new FBounds2D(bounds_index - Width.BOUNDS);
        }

        public static FEdge2D newEdge(int p1, int p2, float l, FPoint2D from, FPoint2D to)
        {
            edge_buffer[edge_index++] = (float) p1;
            edge_buffer[edge_index++] = (float) p2;
            edge_buffer[edge_index++] = l;
            edge_buffer[edge_index++] = 0f;
            return new FEdge2D(edge_index - Width.EDGE, from, to);
        }

        public static FPoint2D newPoint(float x, float y)
        {
            return newPoint(x, y, x, y);
        }

        public static FPoint2D newPoint(float x, float y, float px, float py)
        {
            point_buffer[point_index++] = x;
            point_buffer[point_index++] = y;
            point_buffer[point_index++] = px;
            point_buffer[point_index++] = py;
            return new FPoint2D(point_index - Width.POINT);
        }

        public static FBody2D newBody(float x, float y,
                                      float sx, float sy,
                                      float ax, float ay,
                                      float ps, float pe,
                                      float es, float ee,
                                      int c_flags,
                                      FPoint2D[] points,
                                      FEdge2D[] edges,
                                      FBounds2D bounds,
                                      float force, String entity)
        {
            body_buffer[body_index++] = x;
            body_buffer[body_index++] = y;
            body_buffer[body_index++] = sx;
            body_buffer[body_index++] = sy;
            body_buffer[body_index++] = ax;
            body_buffer[body_index++] = ay;
            body_buffer[body_index++] = c_flags;
            body_buffer[body_index++] = ps;
            body_buffer[body_index++] = pe;
            body_buffer[body_index++] = es;
            body_buffer[body_index++] = ee;
            body_buffer[body_index++] = 0f;
            body_buffer[body_index++] = 0f;
            body_buffer[body_index++] = 0f;
            body_buffer[body_index++] = 0f;
            body_buffer[body_index++] = 0f;
            var idx = body_index - Width.BODY;
            var transform = new FTransform(idx);
            return new FBody2D(idx, force, points, edges, bounds, transform, entity);
        }

    }

    public static void main(String[] args)
    {
        Window window = Window.get();
        OpenCL.init();
        window.run();
        OpenCL.destroy();
    }
}

