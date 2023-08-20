package com.controllerface.bvge;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.geometry.Meshes;
import com.controllerface.bvge.geometry.Models;
import com.controllerface.bvge.window.Window;


public class Main
{
    /**
     * The "Main Memory" interface
     * -
     * Calling code uses this object as single access point to the core memory that
     * is involved in the game engine's operation. The memory being used is resident
     * on the GPU, so this class is essentially a direct interface into GPU memory.
     * This class keeps track of the number of each "top-level" memory object, which
     * are generally categorized as any object that is composed of various primitive
     * types.
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

        // The current count of the various types of memory objects are available
        // through the following accessor methods.

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

        // Pre-calculating the next ID that will be used is useful for setting up
        // data structures, for example when loading models, as it allows you to
        // know the index of an object you will create later. This is helpful when
        // a child object needs to have a reference to a parent, but the parent
        // object can't be created until all children have been created.

        public static int next_armature_id()
        {
            return armature_index / Width.ARMATURE;
        }

        // Creation of new memory objects is available using the following methods

        public static int new_edge(int p1, int p2, float l)
        {
            return new_edge(p1, p2, l, 0);
        }

        public static int new_edge(int p1, int p2, float l, int flags)
        {
            GPU.create_edge(edge_count(), p1, p2, l, flags);
            var idx = edge_index;
            edge_index += Width.EDGE;
            return idx / Width.EDGE;
        }

        public static int new_point(float[] p, int[] t)
        {
            GPU.create_point(point_count(), p[0], p[1], p[0], p[1], t[0], t[1]);
            var idx = point_index;
            point_index += Width.POINT;
            return idx / Width.POINT;
        }

        public static int new_hull(float[] transform, float[] rotation, int[] table, int[] flags)
        {
            GPU.create_hull(hull_count(), transform, rotation, table, flags);
            var idx = hull_index;
            hull_index += Width.HULL;
            return idx / Width.HULL;
        }

        public static int new_armature(float x, float y, int flags)
        {
            GPU.create_armature(armature_count(), x, y, flags);
            var idx = armature_index;
            armature_index += Width.ARMATURE;
            return idx / Width.ARMATURE;
        }

        public static int new_vertex_reference(float x, float y)
        {
            GPU.create_vertex_reference(vertex_ref_count(), x, y);
            var idx = vertex_ref_index;
            vertex_ref_index += Width.VERTEX;
            return idx / Width.VERTEX;
        }

        public static int new_bone_reference(float[] bone_data)
        {
            GPU.create_bone_reference(bone_ref_count(), bone_data);
            var idx = bone_ref_index;
            bone_ref_index += Width.BONE;
            return idx / Width.BONE;
        }

        public static int new_bone(int id, float[] bone_data)
        {
            GPU.create_bone(bone_count(), id, bone_data);
            var idx = bone_index;
            bone_index += Width.BONE;
            return idx / Width.BONE;
        }
    }

    public static void main(String[] args)
    {
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

