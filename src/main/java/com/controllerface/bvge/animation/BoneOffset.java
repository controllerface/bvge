package com.controllerface.bvge.animation;

import com.controllerface.bvge.geometry.SceneNode;
import org.joml.Matrix4f;

public record BoneOffset(int offset_ref_id, String name, Matrix4f transform, SceneNode sceneNode)
{
    public static String IDENTITY_BONE_NAME = "<identity>";
    public static BoneOffset IDENTITY =
            new BoneOffset(-1, IDENTITY_BONE_NAME, new Matrix4f(), SceneNode.empty());
}
