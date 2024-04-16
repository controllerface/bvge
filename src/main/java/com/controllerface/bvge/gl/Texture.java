package com.controllerface.bvge.gl;

import org.lwjgl.BufferUtils;
import org.lwjgl.assimp.AITexture;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.system.MemoryUtil;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.IntBuffer;
import java.util.Objects;

import static org.lwjgl.opengl.GL11.GL_NEAREST;
import static org.lwjgl.opengl.GL11.GL_REPEAT;
import static org.lwjgl.opengl.GL11.GL_RGB;
import static org.lwjgl.opengl.GL11.GL_RGBA;
import static org.lwjgl.opengl.GL11.GL_TEXTURE_2D;
import static org.lwjgl.opengl.GL11.GL_TEXTURE_MAG_FILTER;
import static org.lwjgl.opengl.GL11.GL_TEXTURE_MIN_FILTER;
import static org.lwjgl.opengl.GL11.GL_TEXTURE_WRAP_S;
import static org.lwjgl.opengl.GL11.GL_TEXTURE_WRAP_T;
import static org.lwjgl.opengl.GL11.GL_UNSIGNED_BYTE;
import static org.lwjgl.opengl.GL11.glBindTexture;
import static org.lwjgl.opengl.GL11.glGenTextures;
import static org.lwjgl.opengl.GL11.glTexImage2D;
import static org.lwjgl.opengl.GL11.glTexParameteri;
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

    public void init(String resource_path)
    {
        this.filepath = resource_path;

        texId = glGenTextures();
        glBindTexture(GL_TEXTURE_2D, texId);

        // repeat image
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

        // when stretching, pixelate
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

        IntBuffer width = BufferUtils.createIntBuffer(1);
        IntBuffer height = BufferUtils.createIntBuffer(1);
        IntBuffer channels = BufferUtils.createIntBuffer(1);
        stbi_set_flip_vertically_on_load(true);


        ByteBuffer buf;
        var stream = Texture.class.getResourceAsStream(resource_path);
        try {
            var bytes = stream.readAllBytes();
            buf = MemoryUtil.memAlloc(bytes.length);
            buf.put(bytes);
            buf.flip();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        // todo: load from resource instead of disk
        //ByteBuffer image = stbi_load(resource_path, width, height, channels, 0);
        ByteBuffer image = stbi_load_from_memory(buf, width, height, channels, 0);


        if (image != null)
        {
            this.width = width.get(0);
            this.height = height.get(0);

            if (channels.get(0) == 3)
            {
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width.get(0), height.get(0), 0,
                    GL_RGB, GL_UNSIGNED_BYTE, image);
            }
            else if (channels.get(0) == 4)
            {
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width.get(0), height.get(0), 0,
                    GL_RGBA, GL_UNSIGNED_BYTE, image);
            }
            else
            {
                assert false : "Unexpected channel count" + channels.get(0);
            }
        }
        else
        {
            assert false : "Error: couldn't load image: " + resource_path;
        }

        glGenerateMipmap(texId);

        // do this or it will leak memory
        stbi_image_free(image);
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

    public void destroy()
    {
        glDeleteTextures(texId);
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
