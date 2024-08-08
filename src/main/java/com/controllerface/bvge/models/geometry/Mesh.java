package com.controllerface.bvge.models.geometry;

import com.controllerface.bvge.models.bones.BoneOffset;
import com.controllerface.bvge.models.SceneNode;

import java.util.List;

public record Mesh(String name,
                   int mesh_id,
                   Vertex[] vertices,
                   Face[] faces,
                   List<BoneOffset> bone_offsets,
                   SceneNode sceneNode,
                   int[] hull)
{

    public RawMesh raw_copy()
    {
        float[] raw_vertices = new float[vertices.length * 2];
        float[] raw_uvs = new float[vertices.length * 2];
        int[] raw_faces = new int[faces.length * 3];

        for (int i = 0; i < vertices.length; i++)
        {
            int next = i * 2;
            var vert = vertices[i];
            var uvc = vert.uv_data().getFirst();
            raw_vertices[next] = vert.x();
            raw_vertices[next+1] = vert.y();
            raw_uvs[next] = uvc.x;
            raw_uvs[next+1] = uvc.x;
        }

        for (int i = 0; i < faces.length; i++)
        {
            int next = i * 3;
            var face = faces[i];
            raw_faces[next] = face.p0();
            raw_faces[next+1] = face.p1();
            raw_faces[next+2] = face.p2();
        }

        return new RawMesh(raw_vertices, raw_uvs, raw_faces);
    }
}
