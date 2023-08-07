package com.controllerface.bvge;

import com.controllerface.bvge.cl.OpenCL;
import com.controllerface.bvge.gl.Meshes;
import com.controllerface.bvge.window.Window;


public class Main
{
    public static class Memory
    {
        public static class Width
        {
            public static final int BODY = 4;
            public static final int POINT = 4;
            public static final int EDGE = 4;
        }

        private static final int MAX_BODIES = 100_000;
        private static final int MAX_POINTS = 1_000_000;

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
            return newEdge(p1, p2, l, 0);
        }

        public static int newEdge(int p1, int p2, float l, int flags)
        {
            OpenCL.create_edge(edgesCount(), p1, p2, l, flags);
            edge_index += Width.EDGE;
            var idx = edge_index - Width.EDGE;
            return idx / Width.EDGE;
        }

        public static int newPoint(float[] p)
        {
            OpenCL.create_point(pointsCount(), p[0], p[1], p[0], p[1]);
            point_index += Width.POINT;
            var idx = point_index - Width.POINT;
            return idx / Width.POINT;
        }

        public static int newBody(float[] transform, float[] rotation, int[] table, int flags)
        {
            OpenCL.create_body(bodyCount(), transform, rotation, table, flags);
            body_index += Width.BODY;
            var idx = body_index - Width.BODY;
            return idx / Width.BODY;
        }
    }

    public static void main(String[] args)
    {
        Meshes.init();
        Window window = Window.get();
        window.initOpenGL();
        OpenCL.init(Memory.MAX_BODIES, Memory.MAX_POINTS);
        window.initGameMode();
        window.run();
        OpenCL.destroy();
    }
}

