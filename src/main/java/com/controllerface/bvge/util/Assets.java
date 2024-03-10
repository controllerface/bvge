package com.controllerface.bvge.util;

import com.controllerface.bvge.gl.AbstractShader;
import com.controllerface.bvge.gl.CircleShader;
import com.controllerface.bvge.gl.Shader;
import com.controllerface.bvge.gl.Texture;
import org.lwjgl.assimp.AITexture;

import java.util.HashMap;
import java.util.Map;

public class Assets
{
    private static Map<String, AbstractShader> shaders = new HashMap<>();
    private static Map<String, Texture> textures = new HashMap<>();

    public static AbstractShader load_shader(String resourceName)
    {
        if (shaders.containsKey(resourceName))
        {
            return shaders.get(resourceName);
        }
        else
        {
            AbstractShader shader;
            if (resourceName.contains("circle_shader") || resourceName.contains("bone_shader"))
            {
                shader = new CircleShader(resourceName);
            }
            else
            {
                shader = new Shader(resourceName);
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
}
