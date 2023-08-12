package com.controllerface.bvge.geometry;

import com.controllerface.bvge.Main;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;

import static com.controllerface.bvge.geometry.Bone.IDENTITY_BONE_NAME;

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
        var vert_ref_id = Main.Memory.new_vertex_reference(0,0);
        var vertices = new Vertex[]{ new Vertex(vert_ref_id, 0,0, IDENTITY_BONE_NAME, 1.0f) };
        var faces = new Face[]{ new Face(0, 0, 0) };
        var hull = new int[]{ 0 };
        return new Mesh(vertices, faces, Bone.identity(), Models.SceneNode.empty(), hull);
    }

    /**
     * A simple unit square; 4 vertices defining a square of size 1.
     */
    private static Mesh generate_box_mesh()
    {
        Vertex[] vertices = new Vertex[4];
        Face[] faces = new Face[2];

        float halfSize = 1f / 2f;

        float x1 = -halfSize;
        float y1 = -halfSize;
        int v1 = Main.Memory.new_vertex_reference(x1, y1);
        float x2 = halfSize;
        float y2 = -halfSize;
        int v2 = Main.Memory.new_vertex_reference(x2, y2);
        float x3 = halfSize;
        float y3 = halfSize;
        int v3 = Main.Memory.new_vertex_reference(x3, y3);
        float x4 = -halfSize;
        float y4 = halfSize;
        int v4 = Main.Memory.new_vertex_reference(x4, y4);

        vertices[0] = new Vertex(v1, x1, y1, IDENTITY_BONE_NAME, 1.0f);
        vertices[1] = new Vertex(v2, x2, y2, IDENTITY_BONE_NAME, 1.0f);
        vertices[2] = new Vertex(v3, x3, y3, IDENTITY_BONE_NAME, 1.0f);
        vertices[3] = new Vertex(v4, x4, y4, IDENTITY_BONE_NAME, 1.0f);

        faces[0] = new Face(0, 1, 2);
        faces[1] = new Face(0, 2, 3);

        var hull = new int[]{ 0, 1, 2, 3 };

        return new Mesh(vertices, faces, Bone.identity(), Models.SceneNode.empty(), hull);
    }

    /**
     * A simple polygon with 5 vertices
     */
    private static Mesh generate_poly1_mesh()
    {
        Vertex[] vertices = new Vertex[5];
        Face[] faces = new Face[3];

        float halfSize = 1f / 2f;

        float x1 = -halfSize;
        float y1 = -halfSize;
        int v1 = Main.Memory.new_vertex_reference(x1, y1);
        float x2 = halfSize;
        float y2 = -halfSize;
        int v2 = Main.Memory.new_vertex_reference(x1, y1);
        float x3 = halfSize;
        float y3 = halfSize;
        int v3 = Main.Memory.new_vertex_reference(x1, y1);
        float x4 = 0;
        float y4 = halfSize * 2;
        int v4 = Main.Memory.new_vertex_reference(x1, y1);
        float x5 = -halfSize;
        float y5 = halfSize;
        int v5 = Main.Memory.new_vertex_reference(x1, y1);

        vertices[0] = new Vertex(v1, x1, y1, IDENTITY_BONE_NAME, 1.0f);
        vertices[1] = new Vertex(v2, x2, y2, IDENTITY_BONE_NAME, 1.0f);
        vertices[2] = new Vertex(v3, x3, y3, IDENTITY_BONE_NAME, 1.0f);
        vertices[3] = new Vertex(v4, x4, y4, IDENTITY_BONE_NAME, 1.0f);
        vertices[4] = new Vertex(v5, x5, y5, IDENTITY_BONE_NAME, 1.0f);

        faces[0] = new Face(0, 1, 2);
        faces[1] = new Face(0, 2, 4);
        faces[2] = new Face(4, 2, 3);

        var hull = new int[]{ 0, 1, 2, 3, 4 };

        return new Mesh(vertices, faces, Bone.identity(), Models.SceneNode.empty(), hull);

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
        System.out.printf("registered mesh [%s] with id [%d]\n", mesh_name, mesh_id);
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
