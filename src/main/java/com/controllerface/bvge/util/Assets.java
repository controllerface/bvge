package com.controllerface.bvge.util;

import com.controllerface.bvge.gpu.gl.Shader;
import com.controllerface.bvge.gpu.gl.ThreeStageShader;
import com.controllerface.bvge.gpu.gl.TwoStageShader;
import com.controllerface.bvge.gpu.gl.Texture;
import org.lwjgl.assimp.AITexture;

import java.util.HashMap;
import java.util.Map;

public class Assets
{
    private static Map<String, Shader> shaders = new HashMap<>();
    private static Map<String, Texture> textures = new HashMap<>();

    public static Shader load_shader(String resourceName)
    {
        if (shaders.containsKey(resourceName))
        {
            return shaders.get(resourceName);
        }
        else
        {
            Shader shader;
            if (resourceName.contains("circle_shader")
                || resourceName.contains("water_shader")
                || resourceName.contains("mouse_shader"))
            {
                shader = new ThreeStageShader(resourceName);
            }
            else
            {
                shader = new TwoStageShader(resourceName);
            }

            shader.compile();
            Assets.shaders.put(resourceName, shader);
            return shader;
        }
    }

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
