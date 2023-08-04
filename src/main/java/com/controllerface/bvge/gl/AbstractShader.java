package com.controllerface.bvge.gl;

import org.joml.*;
import org.lwjgl.BufferUtils;

import java.nio.FloatBuffer;

import static org.lwjgl.opengl.GL20.*;

public abstract class AbstractShader
{
    protected int shaderProgramId;

    public abstract void compile();

    public abstract void use();

    public abstract void detach();

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

    public void uploadFloat2(String varName, float x, float y)
    {
        int varLocation = glGetUniformLocation(shaderProgramId, varName);
        use();
        glUniform2f(varLocation, x, y);
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
