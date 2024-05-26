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
    public static final int BLOCK_MESH = next_mesh_index.getAndIncrement();

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

    private static Vertex block_vertex(float x, float y, int uv_index)
    {
        int[] uv_table = new int[2];
        uv_table[0] = -1;
        var uv_list = new ArrayList<Vector2f>();
        for (var uv_channel : BLOCK_ATLAS.uv_channels())
        {
            var channel = uv_channel.get(uv_index);
            uv_list.add(channel);
            var uv_ref_1 = GPGPU.core_memory.new_texture_uv(channel.x, channel.y);
            if (uv_table[0] == -1) { uv_table[0] = uv_ref_1; }
            uv_table[1] = uv_ref_1;
        }
        int v1 = GPGPU.core_memory.new_vertex_reference(x, y, new float[4], uv_table);
        return new Vertex(v1, x, y, uv_list, new String[0], new float[0]);
    }

    private static Face face(int p0, int p1, int p2)
    {
        int[] raw_face_1 = new int[]{ p0, p1, p2, GPGPU.core_memory.next_mesh() };
        int face_id_1 = GPGPU.core_memory.new_mesh_face(raw_face_1);
        return new Face(face_id_1, p0, p1, p2);
    }

    private static final float[] BLOCK =  new float[]
        {
            -0.5f, -0.5f, // bottom left
             0.5f, -0.5f, // bottom right
             0.5f,  0.5f, // top right
            -0.5f,  0.5f, // top left
        };

    /**
     * A simple unit square; 4 vertices defining a square of size 1.
     */
    private static Mesh generate_block_mesh()
    {
        var verts = new Vertex[4];
        verts[0] = block_vertex(BLOCK[0], BLOCK[1], 0);
        verts[1] = block_vertex(BLOCK[2], BLOCK[3], 1);
        verts[2] = block_vertex(BLOCK[4], BLOCK[5], 2);
        verts[3] = block_vertex(BLOCK[6], BLOCK[7], 3);

        var faces        = new Face[]{ face(0, 1, 2), face(0, 2, 3) };
        var vert_table   = new int[]{ verts[0].index(), verts[3].index() };
        var face_table   = new int[]{ faces[0].index(), faces[1].index() };
        var hull         = PhysicsObjects.calculate_convex_hull_table(verts);
        int mesh_id      = GPGPU.core_memory.new_mesh_reference(vert_table, face_table);

        return new Mesh("block", mesh_id, verts, faces, List.of(BoneOffset.IDENTITY), SceneNode.empty(), hull);
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
        register_mesh(BLOCK_MESH, generate_block_mesh());
    }
}
