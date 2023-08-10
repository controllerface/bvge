package com.controllerface.bvge.geometry;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;

public class Meshes
{
    private static final AtomicInteger next_mesh_index = new AtomicInteger(0);

    public static final int CIRCLE_MESH = next_mesh_index.getAndIncrement();
    public static final int BOX_MESH = next_mesh_index.getAndIncrement();
    public static final int POLYGON1_MESH = next_mesh_index.getAndIncrement();

    private static final Map<Integer, Mesh> loaded_meshes = new HashMap<>();
    private static final Map<Integer, Boolean> dirty_meshes = new HashMap<>();
    private static final Map<Integer, Set<Integer>> mesh_instances = new HashMap<>();

    private static final Map<String, Integer> mesh_index_map = new HashMap<>();


    public static Mesh get_mesh_by_index(int index)
    {
        return loaded_meshes.get(index);
    }


    public static void register_mesh_instance(int model_id, int hull_id)
    {
        mesh_instances.computeIfAbsent(model_id, _k -> new HashSet<>()).add(hull_id);
        dirty_meshes.put(model_id, true);
    }

    public static Set<Integer> get_mesh_instances(int model_id)
    {
        return mesh_instances.get(model_id);
    }

    public static boolean is_mesh_dirty(int model_id)
    {
        var r = dirty_meshes.get(model_id);
        return r != null && r;
    }

    public static int get_instance_count(int model_id)
    {
        return mesh_instances.get(model_id).size();
    }

    public static void set_mesh_clean(int model_id)
    {
        dirty_meshes.put(model_id, false);
    }

    /**
     * A simple unit square; 4 vertices defining a square of size 1.
     */
    private static Mesh load_box_mesh()
    {
        Vertex[] vertices = new Vertex[4];
        Face[] faces = new Face[2];

        float halfSize = 1f / 2f;

        vertices[0] = new Vertex(-halfSize, -halfSize);
        vertices[1] = new Vertex(halfSize, -halfSize);
        vertices[2] = new Vertex(halfSize, halfSize);
        vertices[3] = new Vertex(-halfSize, halfSize);

        faces[0] = new Face(0, 1, 2);
        faces[1] = new Face(0, 2, 3);
        return new Mesh(vertices, faces);

//        float[] model = new float[8];
//        model[0] = -halfSize;
//        model[1] = -halfSize;
//
//        model[2] = halfSize;
//        model[3] = -halfSize;
//
//        model[4] = halfSize;
//        model[5] = halfSize;
//
//        model[6] = -halfSize;
//        model[7] = halfSize;
//        return model;
    }

    /**
     * A simple polygon with 5 vertices
     */
    private static Mesh load_poly1_mesh()
    {
        Vertex[] vertices = new Vertex[5];
        Face[] faces = new Face[3];

        float halfSize = 1f / 2f;

        vertices[0] = new Vertex(-halfSize, -halfSize);
        vertices[1] = new Vertex(halfSize, -halfSize);
        vertices[2] = new Vertex(halfSize, halfSize);
        vertices[3] = new Vertex(-halfSize, halfSize);
        vertices[4] = new Vertex(0, halfSize * 2);

        faces[0] = new Face(0, 1, 2);
        faces[1] = new Face(0, 2, 3);
        faces[2] = new Face(3, 2, 4);
        return new Mesh(vertices, faces);

//        float[] model = new float[10];
//        model[0] = -halfSize;
//        model[1] = -halfSize;
//
//        model[2] = halfSize;
//        model[3] = -halfSize;
//
//        model[4] = halfSize;
//        model[5] = halfSize;
//
//        model[6] = -halfSize;
//        model[7] = halfSize;
//
//        model[8] = 0;
//        model[9] = halfSize * 2;
//
//        return model;
    }

    public static void init()
    {
        var circle = new Mesh(new Vertex[]{ new Vertex(0,0) }, new Face[]{ new Face(0, 0, 0) });
        loaded_meshes.put(CIRCLE_MESH, circle);

        loaded_meshes.put(BOX_MESH, load_box_mesh());
        loaded_meshes.put(POLYGON1_MESH, load_poly1_mesh());
    }
}
