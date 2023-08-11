package com.controllerface.bvge.geometry;

import org.joml.Matrix4f;

public record Bone(String name, Matrix4f offset, BoneWeight[] weights, Models.SceneNode sceneNode)
{
    public static String IDENTITY_BONE_NAME = "<identity>";

    public static Bone identity()
    {
        return new Bone(IDENTITY_BONE_NAME, new Matrix4f(), new BoneWeight[0], Models.SceneNode.empty());
    }
}
