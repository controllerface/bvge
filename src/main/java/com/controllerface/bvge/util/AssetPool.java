package com.controllerface.bvge.util;

import com.controllerface.bvge.gl.Shader;
import com.controllerface.bvge.old.SpriteSheet;
import com.controllerface.bvge.gl.Texture;

import java.io.File;
import java.util.HashMap;
import java.util.Map;

public class AssetPool
{
    private static Map<String, Shader> shaders = new HashMap<>();
    private static Map<String, Texture> textures = new HashMap<>();
    private static Map<String, SpriteSheet> spriteSheets = new HashMap<>();

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

    public static void addSpriteSheet(String resourceName, SpriteSheet spriteSheet)
    {
        File file = new File(resourceName);
        if (!AssetPool.spriteSheets.containsKey(file.getAbsolutePath()))
        {
            AssetPool.spriteSheets.put(file.getAbsolutePath(), spriteSheet);
        }
    }

    public static SpriteSheet getSpriteSheet(String resourceName)
    {
        File file = new File(resourceName);
        if (!AssetPool.spriteSheets.containsKey(file.getAbsolutePath()))
        {
            assert false : "Error, tried p2 access non-existent sprite sheet";
        }
        return AssetPool.spriteSheets.getOrDefault(file.getAbsolutePath(), null);
    }
}
