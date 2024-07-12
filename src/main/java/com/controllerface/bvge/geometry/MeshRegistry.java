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
    public static final int R_SHARD_MESH = next_mesh_index.getAndIncrement();
    public static final int L_SHARD_MESH = next_mesh_index.getAndIncrement();
    public static final int SPIKE_MESH = next_mesh_index.getAndIncrement();
    public static final int LINE_MESH = next_mesh_index.getAndIncrement();
    public static final int BLOCK_MESH = next_mesh_index.getAndIncrement();

    private static final Map<Integer, Mesh> loaded_meshes = new HashMap<>();
    private static final Map<String, Integer> mesh_index_map = new HashMap<>();

    private static final BlockAtlas BLOCK_ATLAS = new BlockAtlas();

    private static final float[] BLOCK =  new float[]
        {
            -0.5f, -0.5f, // bottom left
             0.5f, -0.5f, // bottom right
             0.5f,  0.5f, // top right
            -0.5f,  0.5f, // top left
        };

    private static final float[] R_SHARD =  new float[]
        {
            -0.5f, -0.5f, // bottom left
             0.5f, -0.5f, // bottom right
             0.5f,  0.5f, // top
        };

    private static final float[] L_SHARD =  new float[]
        {
            -0.5f, -0.5f, // bottom left
             0.5f, -0.5f, // bottom right
            -0.5f,  0.5f, // top
        };

    private static final float[] SPIKE =  new float[]
        {
            0.5f, 0.5f, // bottom left
            -0.5f, 0.5f, // bottom right
            0.0f,  -0.366f, // top
        };

    private static final float[] LINE =  new float[]
        {
            -0.5f, -0.5f, // bottom left
            0.5f,  0.5f, // top
        };

    public static void init()
    {
        register_mesh(CIRCLE_MESH, generate_circle_mesh());
        register_mesh(R_SHARD_MESH, generate_shard_mesh(R_SHARD, "r_shard"));
        register_mesh(L_SHARD_MESH, generate_shard_mesh(L_SHARD, "l_shard"));
        register_mesh(SPIKE_MESH, generate_spike_mesh());
        register_mesh(LINE_MESH, generate_line_mesh());
        register_mesh(BLOCK_MESH, generate_block_mesh());
    }

    public static Mesh get_mesh_by_index(int index)
    {
        return loaded_meshes.get(index);
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
            var uv_ref_1 = GPGPU.core_memory.reference_container().new_texture_uv(channel.x, channel.y);
            if (uv_table[0] == -1) { uv_table[0] = uv_ref_1; }
            uv_table[1] = uv_ref_1;
        }
        int v1 = GPGPU.core_memory.reference_container().new_vertex_reference(x, y, new float[4], uv_table);
        return new Vertex(v1, x, y, uv_list, new String[0], new float[0]);
    }

    private static Face face(int p0, int p1, int p2)
    {
        int[] raw_face_1 = new int[]{ p0, p1, p2, GPGPU.core_memory.reference_container().next_mesh() };
        int face_id_1 = GPGPU.core_memory.reference_container().new_mesh_face(raw_face_1);
        return new Face(face_id_1, p0, p1, p2);
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

    private static Mesh generate_circle_mesh()
    {
        var vert_ref_id = GPGPU.core_memory.reference_container().new_vertex_reference(0,0, new float[4], new int[2]);
        var vertices = new Vertex[]{ new Vertex(vert_ref_id, 0,0, Collections.emptyList(), new String[0], new float[0]) };
        var faces = new Face[]{ new Face(-1,0, 0, 0) };
        var hull = new int[]{ 0 };
        return new Mesh("int_circle",-1, vertices, faces, List.of(BoneOffset.IDENTITY), SceneNode.empty(), hull);
    }

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
        int mesh_id      = GPGPU.core_memory.reference_container().new_mesh_reference(vert_table, face_table);

        return new Mesh("block", mesh_id, verts, faces, List.of(BoneOffset.IDENTITY), SceneNode.empty(), hull);
    }

    /**
     * A simple triangle; 3 vertices defining a triangle with a base width of 1
     */
    private static Mesh generate_shard_mesh(float[] SHARD, String name)
    {
        var verts = new Vertex[3];
        verts[0] = block_vertex(SHARD[0], SHARD[1], 0);
        verts[1] = block_vertex(SHARD[2], SHARD[3], 1);
        verts[2] = block_vertex(SHARD[4], SHARD[5], 2);

        var faces        = new Face[]{ face(0, 1, 2) };
        var vert_table   = new int[]{ verts[0].index(), verts[2].index() };
        var face_table   = new int[]{ faces[0].index(), faces[0].index() };
        var hull         = PhysicsObjects.calculate_convex_hull_table(verts);
        int mesh_id      = GPGPU.core_memory.reference_container().new_mesh_reference(vert_table, face_table);

        return new Mesh(name, mesh_id, verts, faces, List.of(BoneOffset.IDENTITY), SceneNode.empty(), hull);
    }

    private static Mesh generate_spike_mesh()
    {
        var verts = new Vertex[3];
        verts[0] = block_vertex(SPIKE[0], SPIKE[1], 0);
        verts[1] = block_vertex(SPIKE[2], SPIKE[3], 1);
        verts[2] = block_vertex(SPIKE[4], SPIKE[5], 2);

        var faces        = new Face[]{ face(0, 1, 2) };
        var vert_table   = new int[]{ verts[0].index(), verts[2].index() };
        var face_table   = new int[]{ faces[0].index(), faces[0].index() };
        var hull         = PhysicsObjects.calculate_convex_hull_table(verts);
        int mesh_id      = GPGPU.core_memory.reference_container().new_mesh_reference(vert_table, face_table);

        return new Mesh("spike", mesh_id, verts, faces, List.of(BoneOffset.IDENTITY), SceneNode.empty(), hull);
    }

    private static Mesh generate_line_mesh()
    {
        var vert_ref_id1 = GPGPU.core_memory.reference_container().new_vertex_reference(LINE[0], LINE[1], new float[4], new int[2]);
        var vert_ref_id2 = GPGPU.core_memory.reference_container().new_vertex_reference(LINE[2], LINE[3], new float[4], new int[2]);
        var vertices = new Vertex[]
            {
                new Vertex(vert_ref_id1, LINE[0], LINE[1], Collections.emptyList(), new String[0], new float[0]),
                new Vertex(vert_ref_id2, LINE[2], LINE[3], Collections.emptyList(), new String[0], new float[0])
            };
        var faces = new Face[]{ new Face(-1,0, 0, 0) };
        var hull = new int[]{ 0 };

        return new Mesh("line", -1, vertices, faces, List.of(BoneOffset.IDENTITY), SceneNode.empty(), hull);
    }
}
