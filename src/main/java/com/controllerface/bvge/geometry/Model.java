package com.controllerface.bvge.geometry;

import java.util.ArrayList;
import java.util.List;

public record Model(Mesh[] meshes)
{
    public static Model fromMesh(Mesh mesh)
    {
        return new Model(new Mesh[]{ mesh });
    }
}