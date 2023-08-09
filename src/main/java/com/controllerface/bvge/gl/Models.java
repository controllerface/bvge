package com.controllerface.bvge.gl;

import org.lwjgl.assimp.AIScene;

import static org.lwjgl.assimp.Assimp.*;

public class Models
{
    public static AIScene test_load()
    {
        var path = "C:/Users/Stephen/mdl/test_humanoid.fbx";
        int flags = aiProcess_JoinIdenticalVertices | aiProcess_Triangulate | aiProcess_FixInfacingNormals;
        return aiImportFile(path, flags);
    }

}
