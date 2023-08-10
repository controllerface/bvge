package com.controllerface.bvge.geometry;

public record Model(Mesh[] meshes, int root_index)
{
    public static Model fromBasicMesh(Mesh mesh)
    {
        return new Model(new Mesh[]{ mesh }, 0);
    }
}