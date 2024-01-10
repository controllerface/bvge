package com.controllerface.bvge.geometry;

import java.util.List;

public record Mesh(int mesh_id,
                   Vertex[] vertices,
                   Face[] faces,
                   List<BoneOffset> bone_offsets,
                   Models.SceneNode sceneNode,
                   int[] hull)
{
    public record Raw (float[] r_vertices, float[] r_uv_coords, int[] r_faces)
    {}

    public Raw raw_copy()
    {
        float[] r_vertices = new float[vertices.length * 2];
        float[] r_uv_coords = new float[vertices.length * 2];
        int[] r_faces = new int[faces.length * 3];

        for (int i = 0; i < vertices.length; i++)
        {
            int next = i * 2;
            var vert = vertices[i];
            var uvc = vert.uv_data().get(0);
            r_vertices[next] = vert.x();
            r_vertices[next+1] = vert.y();
            r_uv_coords[next] = uvc.x;
            r_uv_coords[next+1] = uvc.x;
        }

        for (int i = 0; i < faces.length; i++)
        {
            int next = i * 3;
            var face = faces[i];
            r_faces[next] = face.p0();
            r_faces[next+1] = face.p1();
            r_faces[next+2] = face.p2();
        }

        return new Raw(r_vertices, r_uv_coords, r_faces);
    }
}
