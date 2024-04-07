package com.controllerface.bvge.geometry;

import com.controllerface.bvge.animation.BoneOffset;
import com.controllerface.bvge.cl.GPGPU;

import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;

public class Meshes
{
    private static final AtomicInteger next_mesh_index = new AtomicInteger(0);

    public static final int CIRCLE_MESH = next_mesh_index.getAndIncrement();
    public static final int TRIANGLE_MESH = next_mesh_index.getAndIncrement();
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
        var vert_ref_id = GPGPU.core_memory.new_vertex_reference(0,0, new float[4], new int[2]);
        var vertices = new Vertex[]{ new Vertex(vert_ref_id, 0,0, Collections.emptyList(), new String[0], new float[0]) };
        var faces = new Face[]{ new Face(-1,0, 0, 0) };
        var hull = new int[]{ 0 };
        return new Mesh("int_particle",-1, vertices, faces, List.of(BoneOffset.identity()), Models.SceneNode.empty(), hull);
    }

    /**
     * A simple triangle; 3 vertices defining a triangle with a base width of 1
     */
    private static Mesh generate_tri_mesh()
    {
        Vertex[] vertices = new Vertex[3];
        Face[] faces = new Face[1];

        float halfSize = 1f / 2f;

        float x1 = -halfSize;
        float y1 = 0;
        int v1 = GPGPU.core_memory.new_vertex_reference(x1, y1, new float[4], new int[2]);
        float x2 = halfSize;
        float y2 = 0;
        int v2 = GPGPU.core_memory.new_vertex_reference(x2, y2, new float[4], new int[2]);
        float x3 = 0f;
        float y3 = 0.866f;
        int v3 = GPGPU.core_memory.new_vertex_reference(x3, y3, new float[4], new int[2]);

        vertices[0] = new Vertex(v1, x1, y1, Collections.emptyList(), new String[0], new float[0]);
        vertices[1] = new Vertex(v2, x2, y2, Collections.emptyList(), new String[0], new float[0]);
        vertices[2] = new Vertex(v3, x3, y3, Collections.emptyList(), new String[0], new float[0]);

        faces[0] = new Face(-1,0, 1, 2);

        var hull = new int[]{ 0, 1, 2 };

        return new Mesh("int_triangle",-1, vertices, faces, List.of(BoneOffset.identity()), Models.SceneNode.empty(), hull);
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
        int v1 = GPGPU.core_memory.new_vertex_reference(x1, y1, new float[4], new int[2]);
        float x2 = halfSize;
        float y2 = -halfSize;
        int v2 = GPGPU.core_memory.new_vertex_reference(x2, y2, new float[4], new int[2]);
        float x3 = halfSize;
        float y3 = halfSize;
        int v3 = GPGPU.core_memory.new_vertex_reference(x3, y3, new float[4], new int[2]);
        float x4 = -halfSize;
        float y4 = halfSize;
        int v4 = GPGPU.core_memory.new_vertex_reference(x4, y4, new float[4], new int[2]);

        vertices[0] = new Vertex(v1, x1, y1, Collections.emptyList(), new String[0], new float[0]);
        vertices[1] = new Vertex(v2, x2, y2, Collections.emptyList(), new String[0], new float[0]);
        vertices[2] = new Vertex(v3, x3, y3, Collections.emptyList(), new String[0], new float[0]);
        vertices[3] = new Vertex(v4, x4, y4, Collections.emptyList(), new String[0], new float[0]);

        faces[0] = new Face(-1,0, 1, 2);
        faces[1] = new Face(-1,0, 2, 3);

        var hull = new int[]{ 0, 1, 2, 3 };

        return new Mesh("int_square",-1, vertices, faces, List.of(BoneOffset.identity()), Models.SceneNode.empty(), hull);
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
        int v1 = GPGPU.core_memory.new_vertex_reference(x1, y1, new float[4], new int[2]);
        float x2 = halfSize;
        float y2 = -halfSize;
        int v2 = GPGPU.core_memory.new_vertex_reference(x1, y1, new float[4], new int[2]);
        float x3 = halfSize;
        float y3 = halfSize;
        int v3 = GPGPU.core_memory.new_vertex_reference(x1, y1, new float[4], new int[2]);
        float x4 = 0;
        float y4 = halfSize * 2;
        int v4 = GPGPU.core_memory.new_vertex_reference(x1, y1, new float[4], new int[2]);
        float x5 = -halfSize;
        float y5 = halfSize;
        int v5 = GPGPU.core_memory.new_vertex_reference(x1, y1, new float[4], new int[2]);

        vertices[0] = new Vertex(v1, x1, y1, Collections.emptyList(), new String[0], new float[0]);
        vertices[1] = new Vertex(v2, x2, y2, Collections.emptyList(), new String[0], new float[0]);
        vertices[2] = new Vertex(v3, x3, y3, Collections.emptyList(), new String[0], new float[0]);
        vertices[3] = new Vertex(v4, x4, y4, Collections.emptyList(), new String[0], new float[0]);
        vertices[4] = new Vertex(v5, x5, y5, Collections.emptyList(), new String[0], new float[0]);

        faces[0] = new Face(-1,0, 1, 2);
        faces[1] = new Face(-1,0, 2, 4);
        faces[2] = new Face(-1,4, 2, 3);

        var hull = new int[]{ 0, 1, 2, 3, 4 };

        return new Mesh("int_polygon",-1, vertices, faces, List.of(BoneOffset.identity()), Models.SceneNode.empty(), hull);

    }

    public static int register_mesh(String model_name, String mesh_name, Mesh mesh)
    {
        var mesh_key =  model_name + ":" + mesh_name;
        if (mesh_index_map.containsKey(mesh_key))
        {
            throw new IllegalStateException("mesh: " + mesh_key + "already registered.");
        }

        var mesh_id = next_mesh_index.getAndIncrement();
        mesh_index_map.put(mesh_key, mesh_id);
        register_mesh(mesh_id, mesh);
        //System.out.printf("registered mesh [%s] with id [%d]\n", mesh_key, mesh_id);
        return mesh_id;
    }

    private static void register_mesh(int id, Mesh mesh)
    {
        loaded_meshes.put(id, mesh);
    }

    public static void init()
    {
        register_mesh(CIRCLE_MESH, generate_circle_mesh());
        register_mesh(TRIANGLE_MESH, generate_tri_mesh());
        register_mesh(BOX_MESH, generate_box_mesh());
        register_mesh(POLYGON1_MESH, generate_poly1_mesh());
    }
}
