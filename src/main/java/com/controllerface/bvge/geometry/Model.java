package com.controllerface.bvge.geometry;

import java.util.HashMap;
import java.util.Map;

public record Model(Mesh[] meshes, Map<String, Bone> boneMap, int root_index)
{
    public static Model fromBasicMesh(Mesh mesh)
    {
        return new Model(new Mesh[]{ mesh }, new HashMap<>(),0);
    }
}