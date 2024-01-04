package com.controllerface.bvge.geometry;

import org.joml.Matrix4f;

public record MeshBone(int bone_ref_id, String name, Matrix4f offset, Models.SceneNode sceneNode)
{
    public static String IDENTITY_BONE_NAME = "<identity>";

    public static MeshBone identity()
    {
        return new MeshBone(-1, IDENTITY_BONE_NAME, new Matrix4f(), Models.SceneNode.empty());
    }
}
