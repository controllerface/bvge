package com.controllerface.bvge.gl;

import static org.lwjgl.opengl.GL11.GL_DEPTH_COMPONENT;
import static org.lwjgl.opengl.GL11.GL_TEXTURE_2D;
import static org.lwjgl.opengl.GL30.*;
import static org.lwjgl.opengl.GL30.GL_FRAMEBUFFER;

public class FrameBuffer
{
    private int fboId = 0;
    private Texture texture = null;
    public FrameBuffer(int width, int height)
    {
        // generate framebuffer
        fboId = glGenFramebuffers();
        glBindFramebuffer(GL_FRAMEBUFFER, fboId);

        // create texture/ attach to buffer
        this.texture = new Texture(width, height);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, this.texture.getTexId(), 0);

        // create render buffer
        int rboId = glGenRenderbuffers();
        glBindRenderbuffer(GL_RENDERBUFFER, rboId);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, width, height);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rboId);

        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        {
            assert false : "Frame buffer didn't work";
        }
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }

    public void bind()
    {
        glBindFramebuffer(GL_FRAMEBUFFER, fboId);
    }

    public void unbind()
    {
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }

    public int getFboId()
    {
        return fboId;
    }

    public int getTextureId()
    {
        return texture.getTexId();
    }
}
