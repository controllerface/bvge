package com.controllerface.bvge;

import com.controllerface.bvge.cl.OpenCL;
import com.controllerface.bvge.gl.OpenGL;
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
        }

        private static final int MAX_BODIES = 100_000;
        private static final int MAX_POINTS = 1_000_000;

        private static final int BODY_BUFFER_SIZE = Width.BODY * MAX_BODIES;
        private static final int POINT_BUFFER_SIZE = Width.POINT * MAX_POINTS;
        private static final int EDGE_BUFFER_SIZE = Width.EDGE * MAX_POINTS;

        public static float[] body_buffer = new float[BODY_BUFFER_SIZE];
        public static float[] point_buffer = new float[POINT_BUFFER_SIZE];
        public static float[] edge_buffer = new float[EDGE_BUFFER_SIZE];

        static
        {
            int body_bytes = body_buffer.length * Float.BYTES;
            int point_bytes = point_buffer.length * Float.BYTES;
            int edge_bytes = edge_buffer.length * Float.BYTES;
            int bounds_bytes = body_bytes;
            int total = body_bytes + point_bytes + edge_bytes + bounds_bytes;
            System.out.println("body_buffer   : " + body_bytes   + " Bytes");
            System.out.println("point_buffer  : " + point_bytes  + " Bytes");
            System.out.println("edge_buffer   : " + edge_bytes   + " Bytes");
            System.out.println("bounds_buffer : " + bounds_bytes + " Bytes");
            System.out.println("-Total-       : " + total        + " Bytes");
            System.out.println("-Total-       : " + total / 1024 / 1024 + " MB");
        }

        private static int body_index  = 0;
        private static int point_index = 0;
        private static int edge_index  = 0;

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
            return point_index;
        }

        public static int pointsCount()
        {
            return point_index / Width.POINT;
        }

        public static int edgesLength()
        {
            return edge_index;
        }

        public static int edgesCount()
        {
            return edge_index / Width.EDGE;
        }

        public static int newEdge(int p1, int p2, float l)
        {
            OpenCL.create_edge(edgesCount(), p1, p2, l);
            edge_index += Width.EDGE;
            return edge_index - Width.EDGE;
        }

        public static int newPoint(float[] p)
        {
            OpenCL.create_point(pointsCount(), p[0], p[1], p[0], p[1]);
            point_index += Width.POINT;
            return point_index - Width.POINT;
        }

        public static int newBody(float x, float y,
                                  float sx, float sy,
                                  float ax, float ay,
                                  float ps, float pe,
                                  float es, float ee,
                                  int c_flags)
        {
            OpenCL.create_body(bodyCount(), new float[]
                {
                    x,
                    y,
                    sx,
                    sy,
                    ax,
                    ay,
                    c_flags,
                    ps,
                    pe,
                    es,
                    ee,
                    0f,
                    0f,
                    0f,
                    0f,
                    0f
                });
//            body_buffer[body_index++] = x;
//            body_buffer[body_index++] = y;
//            body_buffer[body_index++] = sx;
//            body_buffer[body_index++] = sy;
//            body_buffer[body_index++] = ax;
//            body_buffer[body_index++] = ay;
//            body_buffer[body_index++] = c_flags;
//            body_buffer[body_index++] = ps;
//            body_buffer[body_index++] = pe;
//            body_buffer[body_index++] = es;
//            body_buffer[body_index++] = ee;
//            body_buffer[body_index++] = 0f;
//            body_buffer[body_index++] = 0f;
//            body_buffer[body_index++] = 0f;
//            body_buffer[body_index++] = 0f;
//            body_buffer[body_index++] = 0f;
            body_index += Width.BODY;
            var idx = body_index - Width.BODY;

            return idx / Width.BODY;
        }
    }

    public static void main(String[] args)
    {
        Window window = Window.get();

        // todo: pre-generate Open CL/GL buffers
        //  CL should be done first, offloading object data to the GPU
        //  GL can then be added by making a single VBO out of the point buffer
        //  draw calls will need to be done with ebo's in batches
        //  experiment with batch sizes on different systems
        var b_buf = Memory.BODY_BUFFER_SIZE * Float.BYTES;
        var e_buf = Memory.EDGE_BUFFER_SIZE * Float.BYTES;
        var p_buf = Memory.POINT_BUFFER_SIZE * Float.BYTES;

        window.initOpenGL();
        OpenCL.init(b_buf, e_buf, p_buf);
        //OpenGL.init(p_buf);
        window.initGameMode();
        window.run();
        OpenCL.destroy();
    }
}

