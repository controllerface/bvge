package com.controllerface.bvge;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.geometry.Meshes;
import com.controllerface.bvge.geometry.Models;
import com.controllerface.bvge.window.Window;


public class Main
{
    public static class Memory
    {
        public static class Width
        {
            public static final int ARMATURE = 4;
            public static final int VERTEX   = 2;
            public static final int HULL     = 4;
            public static final int POINT    = 4;
            public static final int EDGE     = 4;
            public static final int BONE     = 16;
        }

        private static final int MAX_HULLS  = 100_000;
        private static final int MAX_POINTS = 1_000_000;

        private static int hull_index       = 0;
        private static int point_index      = 0;
        private static int edge_index       = 0;
        private static int vertex_ref_index = 0;
        private static int bone_ref_index   = 0;
        private static int bone_index       = 0;
        private static int armature_index   = 0;

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
            GPU.create_edge(edge_count(), p1, p2, l, flags);
            edge_index += Width.EDGE;
            var idx = edge_index - Width.EDGE;
            return idx / Width.EDGE;
        }

        public static int new_point(float[] p, int[] t)
        {
            GPU.create_point(point_count(), p[0], p[1], p[0], p[1], t[0], t[1]);
            point_index += Width.POINT;
            var idx = point_index - Width.POINT;
            return idx / Width.POINT;
        }

        public static int new_hull(float[] transform, float[] rotation, int[] table, int[] flags)
        {
            GPU.create_hull(hull_count(), transform, rotation, table, flags);
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
            GPU.create_armature(armature_count(), x, y, flags);
            armature_index += Width.ARMATURE;
            var idx = armature_index - Width.ARMATURE;
            return idx / Width.ARMATURE;
        }

        public static int new_vertex_reference(float x, float y)
        {
            GPU.create_vertex_reference(vertex_ref_count(), x, y);
            vertex_ref_index += Width.VERTEX;
            var idx = vertex_ref_index - Width.VERTEX;
            return idx / Width.VERTEX;
        }

        public static int new_bone_reference(float[] bone_data)
        {
            GPU.create_bone_reference(bone_ref_count(), bone_data);
            bone_ref_index += Width.BONE;
            var idx = bone_ref_index - Width.BONE;
            return idx / Width.BONE;
        }

        public static int new_bone(int id, float[] bone_data)
        {
            GPU.create_bone(bone_count(), id, bone_data);
            bone_index += Width.BONE;
            var idx = bone_index - Width.BONE;
            return idx / Width.BONE;
        }
    }

    public static void main(String[] args)
    {
        Window window = Window.get();
        window.init();

        GPU.init(Memory.MAX_HULLS, Memory.MAX_POINTS);



        window.initGameMode();
        window.run();

        GPU.destroy();
    }
}

