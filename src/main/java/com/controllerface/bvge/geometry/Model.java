package com.controllerface.bvge.geometry;

import com.controllerface.bvge.gl.Texture;
import org.joml.Matrix4f;

import java.util.Collections;
import java.util.List;
import java.util.Map;

public record Model(Mesh[] meshes,
                    Map<String, Matrix4f> bone_transforms,
                    Map<String, Integer> bone_indices,
                    Map<Integer, BoneBindPose> bind_poses,
                    List<Texture> textures,
                    int root_index)
{
    public static Model fromBasicMesh(Mesh mesh)
    {
        return new Model(new Mesh[]{ mesh },
            Map.of(BoneOffset.IDENTITY_BONE_NAME, new Matrix4f()),
            Collections.emptyMap(),
            Collections.emptyMap(),
            Collections.emptyList(),
            0);
    }
}