package com.controllerface.bvge.gpu.gl;

import com.controllerface.bvge.gpu.GPUResource;
import org.lwjgl.BufferUtils;
import org.lwjgl.assimp.AITexture;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.system.MemoryUtil;

import java.io.IOException;
import java.nio.ByteBuffer;
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
import static org.lwjgl.opengl.GL45C.GL_RGB8;
import static org.lwjgl.opengl.GL45C.GL_RGBA8;
import static org.lwjgl.opengl.GL45C.GL_UNPACK_ALIGNMENT;
import static org.lwjgl.opengl.GL45C.glDeleteTextures;
import static org.lwjgl.opengl.GL45C.glPixelStorei;
import static org.lwjgl.opengl.GL45C.*;
import static org.lwjgl.stb.STBImage.*;

public class Texture implements GPUResource
{
    private String filepath;
    private int tex_id;
    private int width;
    private int height;
    private int channels;

    public Texture()
    {
        tex_id = -1;
        width = -1;
        height = -1;
        channels = -1;
    }

    public Texture(int width, int height)
    {
        this.filepath = "generated";
        glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
        tex_id = glCreateTextures(GL_TEXTURE_2D);
        glTextureParameteri(tex_id, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTextureParameteri(tex_id, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTextureStorage2D(tex_id, 1, GL_RGB8, width, height);
        glTextureSubImage2D(tex_id, 0, 0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, 0);
    }

    public void init_array()
    {
        this.filepath = "generated";
        this.tex_id = glCreateTextures(GL_TEXTURE_2D_ARRAY);
    }

    public void init(AITexture raw_texture)
    {
        this.filepath = raw_texture.mFilename().dataString();

        glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
        tex_id = glCreateTextures(GL_TEXTURE_2D);
        glTextureParameteri(tex_id, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTextureParameteri(tex_id, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glTextureParameteri(tex_id, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTextureParameteri(tex_id, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

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
                    glTextureStorage2D(tex_id, 1, GL_RGB8, width.get(0), height.get(0));
                    glTextureSubImage2D(tex_id, 0, 0, 0, width.get(0), height.get(0), GL_RGB, GL_UNSIGNED_BYTE, image);
                }
                else if (channels.get(0) == 4)
                {
                    glTextureStorage2D(tex_id, 1, GL_RGBA8, width.get(0), height.get(0));
                    glTextureSubImage2D(tex_id, 0, 0, 0, width.get(0), height.get(0), GL_RGBA, GL_UNSIGNED_BYTE, image);
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

            stbi_image_free(image);
        }
        //glGenerateMipmap(texId);
    }

    public void init(String resource_path)
    {
        this.filepath = resource_path;

        tex_id = glCreateTextures(GL_TEXTURE_2D);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 4);

        glTextureParameteri(tex_id, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTextureParameteri(tex_id, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glTextureParameteri(tex_id, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTextureParameteri(tex_id, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

        var width = BufferUtils.createIntBuffer(1);
        var height = BufferUtils.createIntBuffer(1);
        var channels = BufferUtils.createIntBuffer(1);
        stbi_set_flip_vertically_on_load(true);

        ByteBuffer buf;
        var stream = Texture.class.getResourceAsStream(resource_path);
        try
        {
            var bytes = stream.readAllBytes();
            buf = MemoryUtil.memAlloc(bytes.length);
            buf.put(bytes);
            buf.flip();
        }
        catch (IOException e)
        {
            throw new RuntimeException(e);
        }
        var image = stbi_load_from_memory(buf, width, height, channels, 0);
        MemoryUtil.memFree(buf);

        if (image != null)
        {
            this.width = width.get(0);
            this.height = height.get(0);

            if (channels.get(0) == 3)
            {
                glTextureStorage2D(tex_id, 1, GL_RGB8, width.get(0), height.get(0));
                glTextureSubImage2D(tex_id, 0, 0, 0, width.get(0), height.get(0), GL_RGB, GL_UNSIGNED_BYTE, image);
            }
            else if (channels.get(0) == 4)
            {
                glTextureStorage2D(tex_id, 1, GL_RGBA8, width.get(0), height.get(0));
                glTextureSubImage2D(tex_id, 0, 0, 0, width.get(0), height.get(0), GL_RGBA, GL_UNSIGNED_BYTE, image);
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

        //glGenerateMipmap(texId);

        stbi_image_free(image);
    }

    /**
     * Binds this texture to the provided texture unit slot
     *
     * @param texture_slot texture slot into which this texture will be stored
     */
    public void bind(int texture_slot)
    {
        glBindTextureUnit(texture_slot, tex_id);
    }

    public int getTex_id()
    {
        return tex_id;
    }

    public void release()
    {
        glDeleteTextures(tex_id);
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

        if (tex_id != texture.tex_id)
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
        result = 31 * result + tex_id;
        result = 31 * result + width;
        result = 31 * result + height;
        result = 31 * result + channels;
        return result;
    }
}
