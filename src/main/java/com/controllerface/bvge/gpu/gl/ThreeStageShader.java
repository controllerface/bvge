package com.controllerface.bvge.gpu.gl;

import java.io.IOException;
import java.nio.charset.StandardCharsets;

import static org.lwjgl.opengl.GL11.GL_FALSE;
import static org.lwjgl.opengl.GL20.*;
import static org.lwjgl.opengl.GL32.GL_GEOMETRY_SHADER;

public class ThreeStageShader extends Shader
{
    private String vertex_source;
    private String fragment_source;
    private String geometry_source;

    public ThreeStageShader(String filePath)
    {
        try (var resource = ThreeStageShader.class.getResourceAsStream("/gl/shaders/" + filePath))
        {
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
        catch (IOException | NullPointerException e)
        {
            e.printStackTrace();
            assert false : "could not open file for shader" + filePath;
        }
    }

    private void setSource(String pattern, String source)
    {
        switch (pattern)
        {
            case "vertex" -> vertex_source = source;
            case "fragment" -> fragment_source = source;
            case "geometry" -> geometry_source = source;
            default -> throw new RuntimeException("incorrect type:" + pattern);
        }
    }

    private int compile_shader(int shader_type, String source)
    {
        int shader_id = glCreateShader(shader_type);
        glShaderSource(shader_id, source);
        glCompileShader(shader_id);
        int success = glGetShaderi(shader_id, GL_COMPILE_STATUS);
        if (success == GL_FALSE)
        {
            int len = glGetShaderi(shader_id, GL_INFO_LOG_LENGTH);
            System.out.println("ERROR: shader compilation failed for type: " + shader_type);
            System.out.println(glGetShaderInfoLog(shader_id, len));
            assert false : "";
        }
        return shader_id;
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
        int vertex_id = compile_shader(GL_VERTEX_SHADER, vertex_source);
        int geometry_id = compile_shader(GL_GEOMETRY_SHADER, geometry_source);
        int fragment_id = compile_shader(GL_FRAGMENT_SHADER, fragment_source);
        shader_program_id = link_shader(vertex_id, geometry_id, fragment_id);
        shader_ids.add(vertex_id);
        shader_ids.add(geometry_id);
        shader_ids.add(fragment_id);
    }
}