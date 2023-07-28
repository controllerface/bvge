package com.controllerface.bvge;

import com.controllerface.bvge.cl.OpenCL;
import com.controllerface.bvge.window.Window;


public class Main
{
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

        static
        {
            int body_bytes = BODY_BUFFER_SIZE * Float.BYTES;
            int point_bytes = POINT_BUFFER_SIZE * Float.BYTES;
            int edge_bytes = EDGE_BUFFER_SIZE * Float.BYTES;
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

        public static int pointsCount()
        {
            return point_index / Width.POINT;
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

        public static int newBody(float[] arg)
        {
            OpenCL.create_body(bodyCount(), arg);
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
        window.initGameMode();
        window.run();
        OpenCL.destroy();
    }
}

