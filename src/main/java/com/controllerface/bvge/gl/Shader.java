package com.controllerface.bvge.gl;

import java.io.IOException;
import java.nio.charset.StandardCharsets;

import static org.lwjgl.opengl.GL11.GL_FALSE;
import static org.lwjgl.opengl.GL20.*;

public class Shader extends AbstractShader
{
    private String vertexSource;
    private String fragmentSource;
    private boolean beingUsed = false;

    public Shader(String filePath)
    {
        try
        {
            // todo: clean this up a bit, make it lok more like the circle shader
            var st= Shader.class.getResourceAsStream("/gl/" + filePath);
            String source = new String(st.readAllBytes(), StandardCharsets.UTF_8);//new String(Files.readAllBytes(Paths.get(filePath)));
            source = source.replaceAll("\\r\\n?", "\n");
            String[] splits = source.split("(#type)( )+([a-zA-Z]+)");

            int index = source.indexOf("#type") + 6;
            int eol = source.indexOf("\n", index);
            String firstPattern = source.substring(index, eol).trim();

            index = source.indexOf("#type", eol) + 6;
            eol = source.indexOf("\n", index);
            String secondPattern = source.substring(index, eol).trim();

            if (firstPattern.equals("vertex"))
            {
                vertexSource = splits[1];
            }
            else if (firstPattern.equals("fragment"))
            {
                fragmentSource = splits[1];
            }
            else
            {
                //throw new IOException("incorrect type:" + firstPattern);
            }

            if (secondPattern.equals("vertex"))
            {
                vertexSource = splits[2];
            }
            else if (secondPattern.equals("fragment"))
            {
                fragmentSource = splits[2];
            }
            else
            {
                //throw new IOException("incorrect type:" + firstPattern);
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
        int vertexID, fragmentID;

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
        shader_program_id = glCreateProgram();
        glAttachShader(shader_program_id, vertexID);
        glAttachShader(shader_program_id, fragmentID);
        glLinkProgram(shader_program_id);

        // check error
        success = glGetProgrami(shader_program_id, GL_LINK_STATUS);
        if (success == GL_FALSE)
        {
            int len = glGetProgrami(shader_program_id, GL_INFO_LOG_LENGTH);
            System.out.println("ERROR: shader linking failed");
            System.out.println(glGetProgramInfoLog(shader_program_id, len));
            assert false : "";
        }
    }

    public void use()
    {
        if (!beingUsed)
        {
            glUseProgram(shader_program_id);
            beingUsed = true;
        }
    }

    public void detach()
    {
        glUseProgram(0);
        beingUsed = false;
    }
}