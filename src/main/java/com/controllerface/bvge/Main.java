package com.controllerface.bvge;

import com.controllerface.bvge.cl.OpenCL_EX;
import com.controllerface.bvge.data.*;
import com.controllerface.bvge.window.Window;

import java.util.*;


public class Main
{
    public static class Memory
    {
        // todo: actually add the cleaner/tracking mechanism for keeping track of
        //  all points, bodies, etc. This will be needed when the arrays need to compact
        //  because of this, those objects won't be able to be records.
        public static class Width
        {
            public static final int BODY = 16;
            public static final int POINT = 4;
            public static final int EDGE = 3;
            public static final int BOUNDS = 8;
        }

        private static final Map<Integer, FBody2D> bodies = new HashMap<>();

        private static final int MAX_BODIES = 100_000;
        private static final int MAX_POINTS = 1_000_000;

        private static final int body_buffer_size   = Width.BODY   * MAX_BODIES;
        private static final int point_buffer_size  = Width.POINT  * MAX_POINTS;
        private static final int edge_buffer_size   = Width.EDGE   * MAX_POINTS;
        private static final int bounds_buffer_size = Width.BOUNDS * MAX_BODIES;

        public static float[] body_buffer   = new float[body_buffer_size];
        public static float[] point_buffer  = new float[point_buffer_size];
        public static float[] edge_buffer   = new float[edge_buffer_size];
        public static float[] bounds_buffer = new float[bounds_buffer_size];

        private static int body_index   = 0;
        private static int point_index  = 0;
        private static int edge_index   = 0;
        private static int bounds_index = 0;

        public static FBody2D bodyByIndex(int index)
        {
            return bodies.get(index);
        }

        public static int bodyCount()
        {
            return body_index / Width.BODY;
        }

        public static int bodyLength()
        {
            return body_index + 1;
        }

        public static int pointCount()
        {
            return point_index / Width.POINT;
        }

        public static int pointLength()
        {
            return point_index + 1;
        }

        public static int boundsLength()
        {
            return bounds_index + 1;
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
            return new FBounds2D(bounds_index - Width.BOUNDS);
        }

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
                                      float ps, float pe,
                                      float es, float ee,
                                      float bi,
                                      FPoint2D[] points,
                                      FEdge2D edges[],
                                      FBounds2D bounds,
                                      float force, String entity)
        {
            body_buffer[body_index++] = x;
            body_buffer[body_index++] = y;
            body_buffer[body_index++] = sx;
            body_buffer[body_index++] = sy;
            body_buffer[body_index++] = ax;
            body_buffer[body_index++] = ay;
            body_buffer[body_index++] = bi;
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
            var newBody = new FBody2D(idx, force, points, edges, bounds, transform, entity);
            bodies.put(idx / Width.BODY, newBody);
            return newBody;
            // todo: add cleaner and log/or possibly compact, if these go out of scope
        }

    }

    public static void main(String[] args)
    {
        int x2 = 0;
        int z = 0;
        Integer y = 0;

        if (x2 == z)
        {
            System.out.println("woudl fire");
        }

        if (y == x2)
        {
            System.out.println("fire?");
        }

        Integer[] x = new Integer[10];
        x[0] = 1;
        List<Integer> Allowed = new ArrayList<>();
        OpenCL_EX.init();
        Window window = Window.get();
        window.run();
        OpenCL_EX.destroy();
    }
}

