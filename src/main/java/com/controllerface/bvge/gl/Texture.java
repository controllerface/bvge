package com.controllerface.bvge.gl;

import org.lwjgl.assimp.AITexture;
import org.lwjgl.system.MemoryStack;

import java.util.Objects;

import static org.lwjgl.opengl.GL45C.*;
import static org.lwjgl.stb.STBImage.*;

public class Texture
{
    private String filepath;
    private int texId;
    private int width;
    private int height;
    private int channels;

    public Texture()
    {
        texId = -1;
        width = -1;
        height = -1;
        channels = -1;
    }

    public Texture(int width, int height)
    {
        this.filepath = "generated";
        texId = glCreateTextures(GL_TEXTURE_2D);
        glTextureParameteri(texId, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTextureParameteri(texId, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTextureStorage2D(texId, 1, GL_RGB8, width, height);
        glTextureSubImage2D(texId, 0,0,0, width, height, GL_RGB, GL_UNSIGNED_BYTE, 0);
    }

    public void init(AITexture raw_texture)
    {
        this.filepath = raw_texture.mFilename().dataString();

        texId = glCreateTextures(GL_TEXTURE_2D);
        glTextureParameteri(texId, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTextureParameteri(texId, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glTextureParameteri(texId, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTextureParameteri(texId, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

        try (var stack = MemoryStack.stackPush())
        {
            var width = stack.mallocInt(1);
            var height = stack.mallocInt(1);
            var channels = stack.mallocInt(1);
            var texture_buffer = raw_texture.pcDataCompressed();

            stbi_set_flip_vertically_on_load(true);
            var image = stbi_load_from_memory(texture_buffer, width, height, channels, 0);

            if (image != null)
            {
                this.width = width.get(0);
                this.height = height.get(0);
                this.channels = channels.get(0);

            if (channels.get(0) == 3)
            {
                glTextureStorage2D(texId, 1, GL_RGB8, width.get(0), height.get(0));
                glTextureSubImage2D(texId, 0,0,0, width.get(0), height.get(0), GL_RGB, GL_UNSIGNED_BYTE, image);
            }
            else if (channels.get(0) == 4)
            {
                glTextureStorage2D(texId, 1, GL_RGBA8, width.get(0), height.get(0));
                glTextureSubImage2D(texId, 0,0,0, width.get(0), height.get(0), GL_RGBA, GL_UNSIGNED_BYTE, image);
            }
            else
            {
                assert false : "Unexpected channel count" + channels.get(0);
            }
        }
        else
        {
            assert false : "Error: couldn't load image: " + this.filepath;
        }

            // do this or it will leak memory
            stbi_image_free(image);
        }
    }

    /**
     * Binds this texture to the provided texture unit slot
     *
     * @param texture_slot texture slot into which this texture will be stored
     */
    public void bind(int texture_slot)
    {
        glBindTextureUnit(texture_slot, texId);
    }

    public int getTexId()
    {
        return texId;
    }

    @Override
    public boolean equals(Object o)
    {
        if (this == o)
        {
            return true;
        }
        if (o == null || getClass() != o.getClass())
        {
            return false;
        }

        Texture texture = (Texture) o;

        if (texId != texture.texId)
        {
            return false;
        }
        if (width != texture.width)
        {
            return false;
        }
        if (height != texture.height)
        {
            return false;
        }
        if (channels != texture.channels)
        {
            return false;
        }
        return Objects.equals(filepath, texture.filepath);
    }

    @Override
    public int hashCode()
    {
        int result = filepath != null
            ? filepath.hashCode()
            : 0;
        result = 31 * result + texId;
        result = 31 * result + width;
        result = 31 * result + height;
        result = 31 * result + channels;
        return result;
    }
}
