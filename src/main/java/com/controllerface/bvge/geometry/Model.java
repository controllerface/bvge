package com.controllerface.bvge.geometry;

import org.joml.Matrix4f;

import java.util.Map;

public record Model(Mesh[] meshes, Map<String, Bone> boneMap, Map<String, Matrix4f> bone_transforms, int root_index)
{
    public static Model fromBasicMesh(Mesh mesh)
    {
        return new Model(new Mesh[]{ mesh },
            Map.of(Bone.IDENTITY_BONE_NAME, mesh.bone()),
            Map.of(Bone.IDENTITY_BONE_NAME, new Matrix4f()),
            0);
    }
}