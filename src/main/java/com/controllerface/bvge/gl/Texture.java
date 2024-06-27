package com.controllerface.bvge.gl;

import com.controllerface.bvge.cl.buffers.Destoryable;
import org.lwjgl.BufferUtils;
import org.lwjgl.assimp.AITexture;
import org.lwjgl.stb.STBImageWrite;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.system.MemoryUtil;
import org.lwjgl.util.freetype.FT_Bitmap;
import org.lwjgl.util.freetype.FT_GlyphSlot;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
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

public class Texture implements Destoryable
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

    public Texture(FT_Bitmap bitmap)
    {
        int width = bitmap.width();
        int height = bitmap.rows();

        ByteBuffer buffer = bitmap.buffer(width * height);
        var image = MemoryUtil.memAlloc(width * height).order(ByteOrder.nativeOrder());
        for (int row = 0; row < height; row++) {
            for (int col = 0; col < width; col++) {
                int flipped_row = height - row - 1;
                var pixel = buffer.get((row * bitmap.pitch()) + col);
                image.put((flipped_row * width) + col, pixel);
            }
        }

        this.filepath = "generated";
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        texId = glCreateTextures(GL_TEXTURE_2D);
        glTextureParameteri(texId, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTextureParameteri(texId, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTextureParameteri(texId, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTextureParameteri(texId, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTextureStorage2D(texId, 1, GL_RGB8, bitmap.width(), bitmap.rows());
        glTextureSubImage2D(texId, 0,0,0, width, height, GL_RED, GL_UNSIGNED_BYTE, image);

        MemoryUtil.memFree(image);
    }

    public Texture(int width, int height)
    {
        this.filepath = "generated";
        glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
        texId = glCreateTextures(GL_TEXTURE_2D);
        glTextureParameteri(texId, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTextureParameteri(texId, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTextureStorage2D(texId, 1, GL_RGB8, width, height);
        glTextureSubImage2D(texId, 0,0,0, width, height, GL_RGB, GL_UNSIGNED_BYTE, 0);
    }

    public void init(AITexture raw_texture)
    {
        this.filepath = raw_texture.mFilename().dataString();

        glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
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
        //glGenerateMipmap(texId);
    }

    public void init(String resource_path)
    {
        this.filepath = resource_path;

        texId = glGenTextures();
        glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
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
        ByteBuffer image = stbi_load_from_memory(buf, width, height, channels, 0);
        MemoryUtil.memFree(buf);

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

        //glGenerateMipmap(texId);

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
