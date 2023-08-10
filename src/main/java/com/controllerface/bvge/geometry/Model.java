package com.controllerface.bvge.geometry;

public record Model(Mesh[] meshes)
{
    public static Model fromBasicMesh(Mesh mesh)
    {
        return new Model(new Mesh[]{ mesh });
    }
}