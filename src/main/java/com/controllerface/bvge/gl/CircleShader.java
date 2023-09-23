package com.controllerface.bvge.gl;

import java.io.IOException;
import java.nio.charset.StandardCharsets;

import static org.lwjgl.opengl.GL11.GL_FALSE;
import static org.lwjgl.opengl.GL20.GL_COMPILE_STATUS;
import static org.lwjgl.opengl.GL20.GL_FRAGMENT_SHADER;
import static org.lwjgl.opengl.GL20.GL_INFO_LOG_LENGTH;
import static org.lwjgl.opengl.GL20.GL_LINK_STATUS;
import static org.lwjgl.opengl.GL20.GL_VERTEX_SHADER;
import static org.lwjgl.opengl.GL20.glAttachShader;
import static org.lwjgl.opengl.GL20.glCompileShader;
import static org.lwjgl.opengl.GL20.glCreateProgram;
import static org.lwjgl.opengl.GL20.glCreateShader;
import static org.lwjgl.opengl.GL20.glGetProgramInfoLog;
import static org.lwjgl.opengl.GL20.glGetProgrami;
import static org.lwjgl.opengl.GL20.glGetShaderInfoLog;
import static org.lwjgl.opengl.GL20.glGetShaderi;
import static org.lwjgl.opengl.GL20.glLinkProgram;
import static org.lwjgl.opengl.GL20.glShaderSource;
import static org.lwjgl.opengl.GL20.glUseProgram;
import static org.lwjgl.opengl.GL32.GL_GEOMETRY_SHADER;

public class CircleShader extends AbstractShader
{
    private String vertexSource;
    private String fragmentSource;
    private String geometrySource;
    private boolean beingUsed = false;

    public CircleShader(String filePath)
    {
        try
        {
            var resource = CircleShader.class.getResourceAsStream("/gl/" + filePath);
            var glsl_source = new String(resource.readAllBytes(), StandardCharsets.UTF_8);

            // normalize source so it works on all platforms
            glsl_source = glsl_source.replaceAll("\\r\\n?", "\n");

            // split out each of the shader stages' source
            String[] shader_stages = glsl_source.split("(#type)( )+([a-zA-Z]+)");

            int index = glsl_source.indexOf("#type") + 6;
            int eol = glsl_source.indexOf("\n", index);
            var type_1 = glsl_source.substring(index, eol).trim();

            index = glsl_source.indexOf("#type", eol) + 6;
            eol = glsl_source.indexOf("\n", index);
            var type_2 = glsl_source.substring(index, eol).trim();

            index = glsl_source.indexOf("#type", eol) + 6;
            eol = glsl_source.indexOf("\n", index);
            var type_3 = glsl_source.substring(index, eol).trim();

            setSource(type_1, shader_stages[1]);
            setSource(type_2, shader_stages[2]);
            setSource(type_3, shader_stages[3]);
        }
        catch (IOException ioe)
        {
            ioe.printStackTrace();
            assert false : "could not open file for shader" + filePath;
        }
    }

    private void setSource(String pattern, String source)
    {
        switch (pattern)
        {
            case "vertex" -> vertexSource = source;
            case "fragment" -> fragmentSource = source;
            case "geometry" -> geometrySource = source;
            default -> throw new RuntimeException("incorrect type:" + pattern);
        }
    }

    private int compile_shader(int shader_type, String source)
    {
        int shaderId = glCreateShader(shader_type);
        glShaderSource(shaderId, source);
        glCompileShader(shaderId);
        int success = glGetShaderi(shaderId, GL_COMPILE_STATUS);
        if (success == GL_FALSE)
        {
            int len = glGetShaderi(shaderId, GL_INFO_LOG_LENGTH);
            System.out.println("ERROR: shader compilation failed for type: " + shader_type);
            System.out.println(glGetShaderInfoLog(shaderId, len));
            assert false : "";
        }
        return shaderId;
    }

    private int link_shader(int vertexID, int geometryID, int fragmentID)
    {
        // link and check step
        int shaderProgramId = glCreateProgram();
        glAttachShader(shaderProgramId, vertexID);
        glAttachShader(shaderProgramId, geometryID);
        glAttachShader(shaderProgramId, fragmentID);
        glLinkProgram(shaderProgramId);

        // check error
        int success = glGetProgrami(shaderProgramId, GL_LINK_STATUS);
        if (success == GL_FALSE)
        {
            int len = glGetProgrami(shaderProgramId, GL_INFO_LOG_LENGTH);
            System.out.println("ERROR: shader linking failed");
            System.out.println(glGetProgramInfoLog(shaderProgramId, len));
            assert false : "";
        }
        return shaderProgramId;
    }

    public void compile()
    {
        int vertexID = compile_shader(GL_VERTEX_SHADER, vertexSource);
        int geometryID = compile_shader(GL_GEOMETRY_SHADER, geometrySource);
        int fragmentID = compile_shader(GL_FRAGMENT_SHADER, fragmentSource);
        shaderProgramId = link_shader(vertexID, geometryID, fragmentID);
    }

    public void use()
    {
        if (!beingUsed)
        {
            glUseProgram(shaderProgramId);
            beingUsed = true;
        }
    }

    public void detach()
    {
        glUseProgram(0);
        beingUsed = false;
    }
}