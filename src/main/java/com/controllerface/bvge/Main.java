package com.controllerface.bvge;

import com.controllerface.bvge.cl.OpenCL;
import com.controllerface.bvge.geometry.Meshes;
import com.controllerface.bvge.geometry.Models;
import com.controllerface.bvge.window.Window;


public class Main
{
    public static class Memory
    {
        /**
         * Memory Mapping Table:
         * =====================
         * This table defines the layout of all tracked structures used to calculate and render the simulation.
         * Each object is broken down into its constituent raw data types, which are primarily a mixture of float
         * and int vectors, and some scalars as well.
         *
         * Point:
         * -----
         * - float4: vertex
         *      x: x position
         *      y: y position
         *      z: x previous
         *      w: y previous
         *
         * Edge:
         * ----
         * - int2: edge points
         *      x: point 1 index
         *      y: point 2 index
         *
         * Hull: (formerly Body) todo: actually rename everything so it is consistent
         * ----
         * - float4: transform
         *      x: position x
         *      y: position y
         *      z: scale x
         *      w: scale y
         *
         * - float4: bounding box
         *      x: start corner x
         *      y: start corner y
         *      z: width
         *      w: height
         *
         * - float2: rotation
         *      x: reference angle
         *      y: current angle
         *
         * - float2: acceleration
         *      x: acceleration x component
         *      y: acceleration y component
         *
         * - int4: element table
         *      x: start point
         *      y: end point
         *      z: start edge
         *      w: end edge
         *
         * - int4: spatial index
         *      x: minimum x key
         *      y: maximum x key
         *      z: minimum y key
         *      w: maximum y key
         *
         * - int2: spatial key bank
         *      x: bank index
         *      y: bank size
         *
         * - int: flags
         *      x: flags for this hull (32-bit bit-field)
         *
         *
         *  todo: add the following
         *
         * Bone:
         * ----
         * - ???: transform
         *      ?: transform data?
         *
         */

        public static class Width
        {
            public static final int ARMATURE = 4;
            public static final int VERTEX =2;
            public static final int HULL = 4;
            public static final int POINT = 4;
            public static final int EDGE = 4;
            public static final int BONE = 16;
        }

        private static final int MAX_HULLS = 100_000;
        private static final int MAX_POINTS = 1_000_000;

        private static int hull_index = 0;
        private static int point_index = 0;
        private static int edge_index  = 0;
        private static int vertex_ref_index = 0;
        private static int bone_ref_index = 0;
        private static int bone_index = 0;
        private static int armature_index = 0;

        public static int armature_count()
        {
            return armature_index / Width.ARMATURE;
        }

        public static int hull_count()
        {
            return hull_index / Width.HULL;
        }

        public static int point_count()
        {
            return point_index / Width.POINT;
        }

        public static int edge_count()
        {
            return edge_index / Width.EDGE;
        }

        public static int vertex_ref_count()
        {
            return vertex_ref_index / Width.VERTEX;
        }

        public static int bone_ref_count()
        {
            return bone_ref_index / Width.BONE;
        }

        public static int bone_count()
        {
            return bone_index / Width.BONE;
        }




        public static int new_edge(int p1, int p2, float l)
        {
            return new_edge(p1, p2, l, 0);
        }

        public static int new_edge(int p1, int p2, float l, int flags)
        {
            OpenCL.create_edge(edge_count(), p1, p2, l, flags);
            edge_index += Width.EDGE;
            var idx = edge_index - Width.EDGE;
            return idx / Width.EDGE;
        }

        public static int new_point(float[] p, int[] t)
        {
            OpenCL.create_point(point_count(), p[0], p[1], p[0], p[1], t[0], t[1]);
            point_index += Width.POINT;
            var idx = point_index - Width.POINT;
            return idx / Width.POINT;
        }

        public static int new_hull(float[] transform, float[] rotation, int[] table, int[] flags)
        {
            OpenCL.create_hull(hull_count(), transform, rotation, table, flags);
            hull_index += Width.HULL;
            var idx = hull_index - Width.HULL;
            return idx / Width.HULL;
        }

        public static int next_armature_id()
        {
            var current_armature_index = armature_index;
            current_armature_index += Width.ARMATURE;
            var idx = current_armature_index - Width.ARMATURE;
            return idx / Width.ARMATURE;
        }

        public static int new_armature(float x, float y, int flags)
        {
            OpenCL.create_armature(armature_count(), x, y, flags);
            armature_index += Width.ARMATURE;
            var idx = armature_index - Width.ARMATURE;
            return idx / Width.ARMATURE;
        }

        public static int new_vertex_reference(float x, float y)
        {
            OpenCL.create_vertex_reference(vertex_ref_count(), x, y);
            vertex_ref_index += Width.VERTEX;
            var idx = vertex_ref_index - Width.VERTEX;
            return idx / Width.VERTEX;
        }

        public static int new_bone_reference(float[] bone_data)
        {
            OpenCL.create_bone_reference(bone_ref_count(), bone_data);
            bone_ref_index += Width.BONE;
            var idx = bone_ref_index - Width.BONE;
            return idx / Width.BONE;
        }

        public static int new_bone(int id, float[] bone_data)
        {
            OpenCL.create_bone(bone_count(), id, bone_data);
            bone_index += Width.BONE;
            var idx = bone_index - Width.BONE;
            return idx / Width.BONE;
        }
    }

    public static void main(String[] args)
    {
        Window window = Window.get();
        window.init();

        OpenCL.init(Memory.MAX_HULLS, Memory.MAX_POINTS);

        Meshes.init();
        Models.init();

        window.initGameMode();
        window.run();

        OpenCL.destroy();
    }
}

