package com.controllerface.bvge;

import com.controllerface.bvge.cl.OpenCL;
import com.controllerface.bvge.data.FBody2D;
import com.controllerface.bvge.data.FEdge2D;
import com.controllerface.bvge.data.FPoint2D;
import com.controllerface.bvge.window.Window;


public class Main
{
    public static class Memory
    {
        public static class Width
        {
            public static final int BODY = 16;
            public static final int POINT = 4;
            public static final int EDGE = 3;
        }

        private static final int body_buffer_size = Width.BODY * 10000;
        private static final int point_buffer_size = Width.POINT * 10000;
        private static final int edge_buffer_size = Width.EDGE * 10000;

        public static float[] body_buffer = new float[body_buffer_size];
        public static float[] point_buffer = new float[point_buffer_size];
        public static float[] edge_buffer = new float[edge_buffer_size];

        private static int body_index = 0;
        private static int point_index = 0;
        private static int edge_index = 0;

        public static FEdge2D newEdge(int p1, int p2, float l, FPoint2D from, FPoint2D to)
        {
            edge_buffer[edge_index++] = (float) p1;
            edge_buffer[edge_index++] = (float) p2;
            edge_buffer[edge_index++] = l;
            return new FEdge2D(edge_index - Width.EDGE, from, to);
            // todo: add cleaner and log/or possibly compact, if these go out of scope
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
            // todo: add cleaner and log/or possibly compact, if these go out of scope
        }

        public static FBody2D newBody(float x, float y,
                                      float sx, float sy,
                                      float ax, float ay,
                                      float bx, float by,
                                      float bw, float bh,
                                      float ps, float pe,
                                      float es, float ee,
                                      FPoint2D[] points, FEdge2D edges[],
                                      float force, String entity)
        {
            body_buffer[body_index++] = x;
            body_buffer[body_index++] = y;
            body_buffer[body_index++] = sx;
            body_buffer[body_index++] = sy;
            body_buffer[body_index++] = ax;
            body_buffer[body_index++] = ay;
            body_buffer[body_index++] = bx;
            body_buffer[body_index++] = by;
            body_buffer[body_index++] = bw;
            body_buffer[body_index++] = bh;
            body_buffer[body_index++] = ps;
            body_buffer[body_index++] = pe;
            body_buffer[body_index++] = es;
            body_buffer[body_index++] = ee;
            body_buffer[body_index++] = 0f;
            body_buffer[body_index++] = 0f;
            return new FBody2D(body_index - Width.BODY, force, points, edges, entity);
            // todo: add cleaner and log/or possibly compact, if these go out of scope
        }

    }

    public static void main(String[] args)
    {
        OpenCL.init();
        Window window = Window.get();
        window.run();
        OpenCL.destroy();
    }
}

