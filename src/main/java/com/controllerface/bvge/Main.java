package com.controllerface.bvge;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.window.Window;


public class Main
{
    /**
     * The "Main Memory" interface
     * -
     * Calling code uses this object as single access point to the core memory that
     * is involved in the game engine's operation. The memory being used is resident
     * on the GPU, so this class is essentially a direct interface into GPU memory.
     * This class keeps track of the number of "top-level" memory objects, which are
     * generally defined as any "logical object" that is composed of various primitive
     * values.
     * Typically, a method is exposed, that accepts as parameters, the various
     * components that describe an object. These methods in turn delegate the memory
     * access calls to the GPU class. In practice, this means that the actual buffers
     * that store the components of created objects may reside in separate memory
     * regions depending on the layout and type of the data being stored.
     * For example, consider an object that has two float components and one integer
     * component, a method to create this object may have a signature like this:
     * -
     *    foo(float x, float y, int i)
     * -
     * The GPU implementation may store all three of these values in one continuous
     * block of memory, or it may store the x and y components together, and the int
     * value in a separate memory segment, or even store all three in completely
     * different sections of memory. As such, this class cannot make too many
     * assumptions about the memory layout of all the various components. However,
     * each top-level object does have one known "memory width" that is used to keep
     * track of the number of objects of that type that are currently stored in
     * memory.
     * For these base objects, they will always be laid out in a continuous
     * manner. Just note that this width generally applies to one or a small number
     * of key properties of the object, and any extra properties or meta-data related
     * to the object will cause take up more memory space. The best existing
     * example of this concept are the HULL object type. The "main" value tracked for
     * these objects is represented on the GPU as a float4, with the first two values
     * (x,y) designating the world-space position of the hull center, and the second
     * two values (z,w) defining the width and height scaling of the hull. This base
     * set of values has a width of 4, so the hull index increments by 4 when each new
     * hull is created. Hulls however also have other data, for example an indexing
     * table that defines the start/end indices in the point and edge buffers of the
     * points and edges that are part of the hull. These values are stored in buffers
     * that align with the hull buffer, so that an index into one buffer can be used
     * interchangeably with the buffers that store all the other components of the
     * object, making access to a single "object" possible by indexing into all the
     * aligned arrays with the same index.
     */
    public static class Memory
    {
        public static class Width
        {
            public static final int ARMATURE = 4;
            public static final int MESH     = 4;
            public static final int FACE     = 4;
            public static final int VERTEX   = 2;
            public static final int UV       = 2;
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
        private static int bone_bind_index  = 0;
        private static int bone_ref_index   = 0;
        private static int bone_index       = 0;
        private static int armature_index   = 0;
        private static int mesh_index       = 0;
        private static int face_index       = 0;
        private static int uv_index         = 0;

        public static int next_uv()
        {
            return uv_index / Width.UV;
        }

        public static int next_face()
        {
            return face_index / Width.FACE;
        }

        public static int next_mesh()
        {
            return mesh_index / Width.MESH;
        }

        public static int next_armature()
        {
            return armature_index / Width.ARMATURE;
        }

        public static int next_hull()
        {
            return hull_index / Width.HULL;
        }

        public static int next_point()
        {
            return point_index / Width.POINT;
        }

        public static int next_edge()
        {
            return edge_index / Width.EDGE;
        }

        public static int next_vertex_ref()
        {
            return vertex_ref_index / Width.VERTEX;
        }

        public static int next_bone_bind()
        {
            return bone_bind_index / Width.BONE;
        }

        public static int next_bone_ref()
        {
            return bone_ref_index / Width.BONE;
        }

        public static int next_bone()
        {
            return bone_index / Width.BONE;
        }

        public static int new_edge(int p1, int p2, float l)
        {
            return new_edge(p1, p2, l, 0);
        }

        public static int new_texture_uv(float u, float v)
        {
            GPU.create_texture_uv(next_uv(), u, v);
            var idx = uv_index;
            uv_index += Width.UV;
            return idx / Width.UV;
        }

        public static int new_edge(int p1, int p2, float l, int flags)
        {
            GPU.create_edge(next_edge(), p1, p2, l, flags);
            var idx = edge_index;
            edge_index += Width.EDGE;
            return idx / Width.EDGE;
        }

        public static int new_point(float[] position, int[] vertex_table, int[] bone_ids)
        {
            var init_vert = new float[]{position[0], position[1], position[0], position[1]};
            GPU.create_point(next_point(), init_vert, vertex_table, bone_ids);
            var idx = point_index;
            point_index += Width.POINT;
            return idx / Width.POINT;
        }

        public static int new_hull(int mesh_id, float[] transform, float[] rotation, int[] table, int[] flags)
        {
            GPU.create_hull(next_hull(), mesh_id, transform, rotation, table, flags);
            var idx = hull_index;
            hull_index += Width.HULL;
            return idx / Width.HULL;
        }

        public static int new_mesh_reference(int[] mesh_ref_table)
        {
            GPU.create_mesh_reference(next_mesh(), mesh_ref_table);
            var idx = mesh_index;
            mesh_index += Width.MESH;
            return idx / Width.MESH;
        }

        public static int new_mesh_face(int[] face)
        {
            GPU.create_mesh_face(next_face(), face);
            var idx = face_index;
            face_index += Width.FACE;
            return idx / Width.FACE;
        }

        public static int new_armature(float x, float y, int[] table, int[] flags, float mass)
        {
            GPU.create_armature(next_armature(), x, y, table, flags, mass);
            var idx = armature_index;
            armature_index += Width.ARMATURE;
            return idx / Width.ARMATURE;
        }

        public static int new_vertex_reference(float x, float y, float[] weights, int[] uv_table)
        {
            GPU.create_vertex_reference(next_vertex_ref(), x, y, weights, uv_table);
            var idx = vertex_ref_index;
            vertex_ref_index += Width.VERTEX;
            return idx / Width.VERTEX;
        }

        public static int new_bone_bind_pose(int bind_parent, float[] bone_data)
        {
            GPU.create_bone_bind_pose(next_bone_bind(), bind_parent, bone_data);
            var idx = bone_bind_index;
            bone_bind_index += Width.BONE;
            return idx / Width.BONE;
        }

        public static int new_bone_reference(float[] bone_data)
        {
            GPU.create_bone_reference(next_bone_ref(), bone_data);
            var idx = bone_ref_index;
            bone_ref_index += Width.BONE;
            return idx / Width.BONE;
        }

        public static int new_bone(int[] offset_id, float[] bone_data)
        {
            GPU.create_bone(next_bone(), offset_id, bone_data);
            var idx = bone_index;
            bone_index += Width.BONE;
            return idx / Width.BONE;
        }

        public static void compact_buffers(int edge_shift,
                                           int bone_shift,
                                           int point_shift,
                                           int hull_shift,
                                           int armature_shift)
        {
            edge_index     -= (edge_shift * Width.EDGE);
            bone_index     -= (bone_shift * Width.BONE);
            point_index    -= (point_shift * Width.POINT);
            hull_index     -= (hull_shift * Width.HULL);
            armature_index -= (armature_shift * Width.ARMATURE);
        }
    }

    public static void main(String[] args)
    {
        //Configuration.DISABLE_CHECKS.set(true);
        Window window = Window.get();
        window.init();

        // todo: the maximum number of certain objects should be collapsed into a single "limits"
        //  object and passed in, this will make it cleaner to add more limits, which is needed
        GPU.init(Memory.MAX_HULLS, Memory.MAX_POINTS);

        window.initGameMode();
        window.run();

        GPU.destroy();
    }
}

