package com.controllerface.bvge;

import com.controllerface.bvge.cl.OpenCL;
import com.controllerface.bvge.gl.Models;
import com.controllerface.bvge.window.Window;


public class Main
{
    public static class Memory
    {
        public static class Width
        {
            // float16
            public static final int BODY = 16;

            // float4
            public static final int POINT = 4;

            // float4
            public static final int EDGE = 4;
        }

        private static final int MAX_BODIES = 100_000;
        private static final int MAX_POINTS = 1_000_000;

        private static final int BODY_BUFFER_SIZE = Width.BODY * MAX_BODIES;
        private static final int POINT_BUFFER_SIZE = Width.POINT * MAX_POINTS;
        private static final int EDGE_BUFFER_SIZE = Width.EDGE * MAX_POINTS;
        private static final int BODY_BUFFER_LENGTH = BODY_BUFFER_SIZE * Float.BYTES;
        private static final int POINT_BUFFER_LENGTH = EDGE_BUFFER_SIZE * Float.BYTES;
        private static final int EDGE_BUFFER_LENGTH = POINT_BUFFER_SIZE * Float.BYTES;
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
        Models.init();
        Window window = Window.get();
        window.initOpenGL();
        OpenCL.init(Memory.BODY_BUFFER_LENGTH,
            Memory.EDGE_BUFFER_LENGTH,
            Memory.POINT_BUFFER_LENGTH);
        window.initGameMode();
        window.run();
        OpenCL.destroy();
    }
}

