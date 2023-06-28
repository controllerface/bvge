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

            // float3
            public static final int EDGE = 3;
            // Edges define constraints that are set between two vertices

            // float8
            public static final int BOUNDS = 8;
            // Bounding boxes are used for proximity checks on bodies

            // int2
            public static final int KEY = 2;
            // A spatial index key is calculated for each body to aid in collision checks

            // int
            public static final int POINTER = 1;
            // A pointer index used for referencing spatial key matches
        }

        private static final Map<Integer, FBody2D> bodies = new HashMap<>();

        private static final int MAX_BODIES  = 100_000;
        private static final int MAX_POINTS  = 1_000_000;
        private static final int MAX_KEYS    = MAX_BODIES * 50;

        private static final int BODY_BUFFER_SIZE    = Width.BODY    * MAX_BODIES;
        private static final int POINT_BUFFER_SIZE   = Width.POINT   * MAX_POINTS;
        private static final int EDGE_BUFFER_SIZE    = Width.EDGE    * MAX_POINTS;
        private static final int BOUNDS_BUFFER_SIZE  = Width.BOUNDS  * MAX_BODIES;
        private static final int KEY_BUFFER_SIZE     = Width.KEY     * MAX_KEYS;
        private static final int POINTER_BUFFER_SIZE = Width.POINTER * MAX_KEYS;

        public static float[] body_buffer   = new float[BODY_BUFFER_SIZE];
        public static float[] point_buffer  = new float[POINT_BUFFER_SIZE];
        public static float[] edge_buffer   = new float[EDGE_BUFFER_SIZE];
        public static float[] bounds_buffer = new float[BOUNDS_BUFFER_SIZE];
        public static int[] key_buffer      = new int[KEY_BUFFER_SIZE];
        public static int[] pointer_buffer  = new int[POINTER_BUFFER_SIZE];


        private static int body_index    = 0;
        private static int point_index   = 0;
        private static int edge_index    = 0;
        private static int bounds_index  = 0;
        private static int key_index     = 0;
        private static int pointer_index = 0;

        public static void startKeyRebuild()
        {
            key_index = 0;
            pointer_index = 0;
        }

        public static int[] storeKeyBank(int[] key_bank)
        {
            int[] out = new int[2];
            out[0] = key_index;
            out[1] = key_bank.length;
            System.arraycopy(key_bank, 0, key_buffer, key_index, key_bank.length);
            key_index += key_bank.length;
            return out;
        }

        public static int storeKeyPointer(int[] key_data)
        {
            int out = pointer_index;
            pointer_buffer[pointer_index++] = key_data.length; // the first value is the length of data
            System.arraycopy(key_data, 0, pointer_buffer, pointer_index, key_data.length);
            pointer_index += key_data.length;
            return out;
        }

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
        OpenCL_EX.init();
        Window window = Window.get();
        window.run();
        OpenCL_EX.destroy();
    }
}

