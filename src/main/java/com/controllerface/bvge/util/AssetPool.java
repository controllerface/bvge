package com.controllerface.bvge.util;

import com.controllerface.bvge.gl.Shader;
import com.controllerface.bvge.gl.Texture;

import java.util.HashMap;
import java.util.Map;

public class AssetPool
{
    private static Map<String, Shader> shaders = new HashMap<>();
    private static Map<String, Texture> textures = new HashMap<>();

    public static Shader getShader(String resourceName)
    {
        if (shaders.containsKey(resourceName))
        {
            return shaders.get(resourceName);
        }
        else
        {
            Shader shader = new Shader(resourceName);
            shader.compile();
            AssetPool.shaders.put(resourceName, shader);
            return shader;
        }
    }

    public static Texture getTexture(String resourceName)
    {
        if (textures.containsKey(resourceName))
        {
            return textures.get(resourceName);
        }
        else
        {
            Texture texture = new Texture();
            texture.init(resourceName);
            AssetPool.textures.put(resourceName, texture);
            return texture;
        }
    }
}
