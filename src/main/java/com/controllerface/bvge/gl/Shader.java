package com.controllerface.bvge.gl;

import com.controllerface.bvge.cl.buffers.Destroyable;
import org.joml.*;
import org.lwjgl.opengl.GL20;
import org.lwjgl.system.MemoryStack;

import java.util.ArrayList;
import java.util.List;

import static org.lwjgl.opengl.GL20.*;

public abstract class Shader implements Destroyable
{
    protected int shader_program_id;
    protected List<Integer> shader_ids = new ArrayList<>();
    protected boolean being_used = false;

    public abstract void compile();

    public void use()
    {
        if (!being_used)
        {
            glUseProgram(shader_program_id);
            being_used = true;
        }
    }

    public void detach()
    {
        glUseProgram(0);
        being_used = false;
    }

    public void uploadMat4f(String varname, Matrix4f mat4)
    {
        try (var stack = MemoryStack.stackPush())
        {
            int var_location = glGetUniformLocation(shader_program_id, varname);
            use();
            var buffer = stack.mallocFloat(16);
            mat4.get(buffer);
            glUniformMatrix4fv(var_location, false, buffer);
        }
    }

    public void uploadMat3f(String varname, Matrix3f mat3)
    {
        try (var stack = MemoryStack.stackPush())
        {
            int var_location = glGetUniformLocation(shader_program_id, varname);
            use();
            var buffer = stack.mallocFloat(9);
            mat3.get(buffer);
            glUniformMatrix3fv(var_location, false, buffer);
        }
    }

    public void uploadvec4f(String varName, Vector4f vec)
    {
        int varLocation = glGetUniformLocation(shader_program_id, varName);
        use();
        glUniform4f(varLocation, vec.x, vec.y, vec.z, vec.w);
    }

    public void uploadvec3f(String varName, Vector3f vec)
    {
        int varLocation = glGetUniformLocation(shader_program_id, varName);
        use();
        glUniform3f(varLocation, vec.x, vec.y, vec.z);
    }

    public void uploadvec2f(String varName, Vector2f vec)
    {
        int varLocation = glGetUniformLocation(shader_program_id, varName);
        use();
        glUniform2f(varLocation, vec.x, vec.y);
    }

    public void uploadFloat2(String varName, float x, float y)
    {
        int varLocation = glGetUniformLocation(shader_program_id, varName);
        use();
        glUniform2f(varLocation, x, y);
    }

    public void uploadfloat(String varName, float val)
    {
        int varLocation = glGetUniformLocation(shader_program_id, varName);
        use();
        glUniform1f(varLocation, val);
    }

    public void uploadInt(String varName, int val)
    {
        int varLocation = glGetUniformLocation(shader_program_id, varName);
        use();
        glUniform1i(varLocation, val);
    }

    public void uploadTexture(String varName, int slot)
    {
        int varLocation = glGetUniformLocation(shader_program_id, varName);
        use();
        glUniform1i(varLocation, slot);
    }

    public void uploadIntArray(String varName, int[] array)
    {
        int varLocation = glGetUniformLocation(shader_program_id, varName);
        use();
        glUniform1iv(varLocation, array);
    }

    public void destroy()
    {
        shader_ids.forEach(GL20::glDeleteShader);
        glDeleteProgram(shader_program_id);
    }
}
