package com.controllerface.bvge.geometry;

import com.controllerface.bvge.animation.BoneOffset;
import com.controllerface.bvge.cl.GPGPU;
import com.controllerface.bvge.physics.PhysicsObjects;
import org.joml.Vector2f;

import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;

public class MeshRegistry
{
    private static final AtomicInteger next_mesh_index = new AtomicInteger(0);

    public static final int CIRCLE_MESH = next_mesh_index.getAndIncrement();
    public static final int TRIANGLE_MESH = next_mesh_index.getAndIncrement();
    public static final int BOX_MESH = next_mesh_index.getAndIncrement();

    private static final Map<Integer, Mesh> loaded_meshes = new HashMap<>();
    private static final Map<String, Integer> mesh_index_map = new HashMap<>();

    private static final BlockAtlas BLOCK_ATLAS = new BlockAtlas();

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
        return new Mesh("int_circle",-1, vertices, faces, List.of(BoneOffset.IDENTITY), SceneNode.empty(), hull);
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

        return new Mesh("int_triangle",-1, vertices, faces, List.of(BoneOffset.IDENTITY), SceneNode.empty(), hull);
    }

    /**
     * A simple unit square; 4 vertices defining a square of size 1.
     */
    private static Mesh generate_square_mesh()
    {
        Vertex[] vertices = new Vertex[4];
        Face[] faces = new Face[2];

        float halfSize = 1f / 2f;

        int[] uv_table_1 = new int[2];
        int[] uv_table_2 = new int[2];
        int[] uv_table_3 = new int[2];
        int[] uv_table_4 = new int[2];
        uv_table_1[0] = -1;
        uv_table_2[0] = -1;
        uv_table_3[0] = -1;
        uv_table_4[0] = -1;
        var uv_list_1 = new ArrayList<Vector2f>();
        var uv_list_2 = new ArrayList<Vector2f>();
        var uv_list_3 = new ArrayList<Vector2f>();
        var uv_list_4 = new ArrayList<Vector2f>();


        for (var channel : BLOCK_ATLAS.uv_channels())
        {
            var chan_1 = channel.get(0);
            uv_list_1.add(chan_1);
            var uv_ref_1 = GPGPU.core_memory.new_texture_uv(chan_1.x, chan_1.y);
            if (uv_table_1[0] == -1) { uv_table_1[0] = uv_ref_1; }
            uv_table_1[1] = uv_ref_1;
        }

        for (var channel : BLOCK_ATLAS.uv_channels())
        {
            var chan_2 = channel.get(1);
            uv_list_2.add(chan_2);
            var uv_ref_2 = GPGPU.core_memory.new_texture_uv(chan_2.x, chan_2.y);
            if (uv_table_2[0] == -1) { uv_table_2[0] = uv_ref_2; }
            uv_table_2[1] = uv_ref_2;
        }

        for (var channel : BLOCK_ATLAS.uv_channels())
        {
            var chan_3 = channel.get(2);
            uv_list_3.add(chan_3);
            var uv_ref_3 = GPGPU.core_memory.new_texture_uv(chan_3.x, chan_3.y);
            if (uv_table_3[0] == -1) { uv_table_3[0] = uv_ref_3; }
            uv_table_3[1] = uv_ref_3;
        }

        for (var channel : BLOCK_ATLAS.uv_channels())
        {
            var chan_4 = channel.get(3);
            uv_list_4.add(chan_4);
            var uv_ref_4 = GPGPU.core_memory.new_texture_uv(chan_4.x, chan_4.y);
            if (uv_table_4[0] == -1) { uv_table_4[0] = uv_ref_4; }
            uv_table_4[1] = uv_ref_4;
        }

        float x1 = -halfSize;
        float y1 = -halfSize;
        int v1 = GPGPU.core_memory.new_vertex_reference(x1, y1, new float[4], uv_table_1);

        float x2 = halfSize;
        float y2 = -halfSize;
        int v2 = GPGPU.core_memory.new_vertex_reference(x2, y2, new float[4], uv_table_2);

        float x3 = halfSize;
        float y3 = halfSize;
        int v3 = GPGPU.core_memory.new_vertex_reference(x3, y3, new float[4], uv_table_3);

        float x4 = -halfSize;
        float y4 = halfSize;
        int v4 = GPGPU.core_memory.new_vertex_reference(x4, y4, new float[4], uv_table_4);

        vertices[0] = new Vertex(v1, x1, y1, uv_list_1, new String[0], new float[0]);
        vertices[1] = new Vertex(v2, x2, y2, uv_list_2, new String[0], new float[0]);
        vertices[2] = new Vertex(v3, x3, y3, uv_list_3, new String[0], new float[0]);
        vertices[3] = new Vertex(v4, x4, y4, uv_list_4, new String[0], new float[0]);


        int next_mesh = GPGPU.core_memory.next_mesh();

        int[] raw_face_1 = new int[4];
        raw_face_1[0] = 0;
        raw_face_1[1] = 1;
        raw_face_1[2] = 2;
        raw_face_1[3] = next_mesh;

        int[] raw_face_2 = new int[4];
        raw_face_2[0] = 0;
        raw_face_2[1] = 2;
        raw_face_2[2] = 3;
        raw_face_2[3] = next_mesh;

        int face_id_1 = GPGPU.core_memory.new_mesh_face(raw_face_1);
        int face_id_2 = GPGPU.core_memory.new_mesh_face(raw_face_2);

        // face need to be created in memory
        faces[0] = new Face(face_id_1,0, 1, 2);
        faces[1] = new Face(face_id_2,0, 2, 3);

        var hull_table = PhysicsObjects.calculate_convex_hull_table(vertices);

        int[] vertex_table = new int[2];
        int[] face_table = new int[2];
        vertex_table[0] = v1;
        vertex_table[1] = v4;
        face_table[0] = face_id_1;
        face_table[1] = face_id_2;
        var mesh_id = GPGPU.core_memory.new_mesh_reference(vertex_table, face_table);

        return new Mesh("int_square", mesh_id, vertices, faces, List.of(BoneOffset.IDENTITY), SceneNode.empty(), hull_table);
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
        register_mesh(BOX_MESH, generate_square_mesh());
    }
}
