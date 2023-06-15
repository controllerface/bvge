package com.controllerface.bvge.rendering;

import org.joml.*;
import org.lwjgl.BufferUtils;

import java.io.IOException;
import java.nio.FloatBuffer;
import java.nio.file.Files;
import java.nio.file.Paths;

import static org.lwjgl.opengl.GL11.GL_FALSE;
import static org.lwjgl.opengl.GL20.*;
import static org.lwjgl.opengl.GL20.glGetShaderInfoLog;

public class Shader
{
    private int shaderProgramId;
    private String vertexSource;
    private String fragmentSource;
    private String filepath;

    private boolean beingUsed = false;

    public Shader(String filePath)
    {
        this.filepath = filePath;
        try
        {
            String source = new String(Files.readAllBytes(Paths.get(filePath)));
            String[] splits = source.split("(#type)( )+([a-zA-Z]+)");

            int index = source.indexOf("#type") + 6;
            int eol = source.indexOf("\r\n", index);
            String firstPattern = source.substring(index, eol).trim();

            index = source.indexOf("#type", eol) + 6;
            eol = source.indexOf("\r\n", index);
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
                throw new IOException("incorrect type:" + firstPattern);
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
                throw new IOException("incorrect type:" + firstPattern);
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
        shaderProgramId = glCreateProgram();
        glAttachShader(shaderProgramId, vertexID);
        glAttachShader(shaderProgramId, fragmentID);
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




    public void uploadMat4f(String varname, Matrix4f mat4)
    {
        int varlocation = glGetUniformLocation(shaderProgramId, varname);
        use();
        FloatBuffer matBuffer = BufferUtils.createFloatBuffer(16);
        mat4.get(matBuffer);
        glUniformMatrix4fv(varlocation, false, matBuffer);
    }

    public void uploadMat3f(String varname, Matrix3f mat3)
    {
        int varlocation = glGetUniformLocation(shaderProgramId, varname);
        use();
        FloatBuffer matBuffer = BufferUtils.createFloatBuffer(9);
        mat3.get(matBuffer);
        glUniformMatrix3fv(varlocation, false, matBuffer);
    }





    public void uploadvec4f(String varName, Vector4f vec)
    {
        int varLocation = glGetUniformLocation(shaderProgramId, varName);
        use();
        glUniform4f(varLocation, vec.x, vec.y, vec.z, vec.w);
    }

    public void uploadvec3f(String varName, Vector3f vec)
    {
        int varLocation = glGetUniformLocation(shaderProgramId, varName);
        use();
        glUniform3f(varLocation, vec.x, vec.y, vec.z);
    }

    public void uploadvec2f(String varName, Vector2f vec)
    {
        int varLocation = glGetUniformLocation(shaderProgramId, varName);
        use();
        glUniform2f(varLocation, vec.x, vec.y);
    }




    public void uploadfloat(String varName, float val)
    {
        int varLocation = glGetUniformLocation(shaderProgramId, varName);
        use();
        glUniform1f(varLocation, val);
    }

    public void uploadInt(String varName, int val)
    {
        int varLocation = glGetUniformLocation(shaderProgramId, varName);
        use();
        glUniform1i(varLocation, val);
    }

    public void uploadTexture(String varName, int slot)
    {
        int varLocation = glGetUniformLocation(shaderProgramId, varName);
        use();
        glUniform1i(varLocation, slot);
    }

    public void uploadIntArray(String varName, int[] array)
    {
        int varLocation = glGetUniformLocation(shaderProgramId, varName);
        use();
        glUniform1iv(varLocation, array);
    }
}