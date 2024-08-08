package com.controllerface.bvge.util;

import com.controllerface.bvge.gpu.gl.shaders.GL_Shader;
import com.controllerface.bvge.gpu.gl.textures.Texture;
import com.controllerface.bvge.gpu.gl.shaders.ThreeStageShader;
import com.controllerface.bvge.gpu.gl.shaders.TwoStageShader;
import org.lwjgl.assimp.AITexture;

import java.util.HashMap;
import java.util.Map;

public class Assets
{
    private static Map<String, Texture> textures = new HashMap<>();

    public static Texture load_texture(AITexture textureData)
    {
        var resourceName = textureData.mFilename().dataString();
        if (textures.containsKey(resourceName))
        {
            return textures.get(resourceName);
        }
        else
        {
            Texture texture = new Texture();
            texture.init(textureData);
            Assets.textures.put(resourceName, texture);
            return texture;
        }
    }

    public static Texture load_texture(String resourceName)
    {
        if (textures.containsKey(resourceName))
        {
            return textures.get(resourceName);
        }
        else
        {
            Texture texture = new Texture();
            texture.init(resourceName);
            Assets.textures.put(resourceName, texture);
            return texture;
        }
    }
}
