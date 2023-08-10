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
    private static final Map<String, Integer> mesh_index_map = new HashMap<>();

    public static Mesh get_mesh_by_index(int index)
    {
        return loaded_meshes.get(index);
    }

    private static Mesh generate_circle_mesh()
    {
        var vertices = new Vertex[]{ new Vertex(0,0) };
        var faces = new Face[]{ new Face(0, 0, 0) };
        return new Mesh(vertices, faces, Bone.identity(), Models.SceneNode.empty());
    }

    /**
     * A simple unit square; 4 vertices defining a square of size 1.
     */
    private static Mesh generate_box_mesh()
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
        return new Mesh(vertices, faces, Bone.identity(), Models.SceneNode.empty());
    }

    /**
     * A simple polygon with 5 vertices
     */
    private static Mesh generate_poly1_mesh()
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
        return new Mesh(vertices, faces, Bone.identity(), Models.SceneNode.empty());

    }

    public static int register_mesh(String mesh_name, Mesh mesh)
    {
        if (mesh_index_map.containsKey(mesh_name))
        {
            throw new IllegalStateException("mesh: " + mesh_name + "already registered.");
        }

        var mesh_id = next_mesh_index.getAndIncrement();
        mesh_index_map.put(mesh_name, mesh_id);
        register_mesh(mesh_id, mesh);
        return mesh_id;
    }

    private static void register_mesh(int id, Mesh mesh)
    {
        loaded_meshes.put(id, mesh);
    }

    public static void init()
    {
        register_mesh(CIRCLE_MESH, generate_circle_mesh());
        register_mesh(BOX_MESH, generate_box_mesh());
        register_mesh(POLYGON1_MESH, generate_poly1_mesh());
    }
}
