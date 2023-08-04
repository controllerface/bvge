package com.controllerface.bvge.gl;

import org.joml.*;
import org.lwjgl.BufferUtils;

import java.io.IOException;
import java.nio.FloatBuffer;
import java.nio.charset.StandardCharsets;

import static org.lwjgl.opengl.GL11.GL_FALSE;
import static org.lwjgl.opengl.GL20.*;
import static org.lwjgl.opengl.GL32.GL_GEOMETRY_SHADER;

public class Shader2 extends AbstractShader
{
    private String vertexSource;
    private String fragmentSource;
    private String geometrySource;
    private boolean beingUsed = false;

    public Shader2(String filePath)
    {
        try
        {
            // todo: stop this string split stuff and just put the shaders in their own files
            var st= Shader2.class.getResourceAsStream("/gl/" + filePath);
            String source = new String(st.readAllBytes(), StandardCharsets.UTF_8);//new String(Files.readAllBytes(Paths.get(filePath)));
            source = source.replaceAll("\\r\\n?", "\n");
            String[] splits = source.split("(#type)( )+([a-zA-Z]+)");

            int index = source.indexOf("#type") + 6;
            int eol = source.indexOf("\n", index);
            String firstPattern = source.substring(index, eol).trim();

            index = source.indexOf("#type", eol) + 6;
            eol = source.indexOf("\n", index);
            String secondPattern = source.substring(index, eol).trim();

            index = source.indexOf("#type", eol) + 6;
            eol = source.indexOf("\n", index);
            String thirdPattern = source.substring(index, eol).trim();

            switch (firstPattern)
            {
                case "vertex" -> vertexSource = splits[1];
                case "fragment" -> fragmentSource = splits[1];
                case "geometry" -> geometrySource = splits[1];
                default -> throw new IOException("incorrect type:" + firstPattern);
            }

            switch (secondPattern)
            {
                case "vertex" -> vertexSource = splits[2];
                case "fragment" -> fragmentSource = splits[2];
                case "geometry" -> geometrySource = splits[2];
                default -> throw new IOException("incorrect type:" + firstPattern);
            }

            switch (thirdPattern)
            {
                case "vertex" -> vertexSource = splits[3];
                case "fragment" -> fragmentSource = splits[3];
                case "geometry" -> geometrySource = splits[3];
                default -> throw new IOException("incorrect type:" + firstPattern);
            }
        }
        catch (IOException ioe)
        {
            ioe.printStackTrace();
            assert false : "could not open file for shader" + filePath;
        }
    }

    public void compile()
    {
        int vertexID, fragmentID, geometryID;



        // vertex shader
        vertexID = glCreateShader(GL_VERTEX_SHADER);
        glShaderSource(vertexID, vertexSource);
        glCompileShader(vertexID);
        int success = glGetShaderi(vertexID, GL_COMPILE_STATUS);
        if (success == GL_FALSE)
        {
            int len = glGetShaderi(vertexID, GL_INFO_LOG_LENGTH);
            System.out.println("ERROR: vertex shader compilation failed");
            System.out.println(glGetShaderInfoLog(vertexID, len));
            assert false : "";
        }


        // geo shader
        geometryID = glCreateShader(GL_GEOMETRY_SHADER);
        glShaderSource(geometryID, geometrySource);
        glCompileShader(geometryID);
        success = glGetShaderi(geometryID, GL_COMPILE_STATUS);
        if (success == GL_FALSE)
        {
            int len = glGetShaderi(geometryID, GL_INFO_LOG_LENGTH);
            System.out.println("ERROR: geometry shader compilation failed");
            System.out.println(glGetShaderInfoLog(geometryID, len));
            assert false : "";
        }



        // fragment shader
        fragmentID = glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(fragmentID, fragmentSource);
        glCompileShader(fragmentID);
        success = glGetShaderi(fragmentID, GL_COMPILE_STATUS);
        if (success == GL_FALSE)
        {
            int len = glGetShaderi(fragmentID, GL_INFO_LOG_LENGTH);
            System.out.println("ERROR: fragment shader compilation failed");
            System.out.println(glGetShaderInfoLog(fragmentID, len));
            assert false : "";
        }


        // link and check step
        shaderProgramId = glCreateProgram();
        glAttachShader(shaderProgramId, vertexID);
        glAttachShader(shaderProgramId, fragmentID);
        glAttachShader(shaderProgramId, geometryID);
        glLinkProgram(shaderProgramId);

        // check erorr
        success = glGetProgrami(shaderProgramId, GL_LINK_STATUS);
        if (success == GL_FALSE)
        {
            int len = glGetProgrami(shaderProgramId, GL_INFO_LOG_LENGTH);
            System.out.println("ERROR: shader linking failed");
            System.out.println(glGetProgramInfoLog(shaderProgramId, len));
            assert false : "";
        }
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