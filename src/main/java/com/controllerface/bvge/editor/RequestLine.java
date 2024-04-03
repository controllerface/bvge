package com.controllerface.bvge.editor;

public record RequestLine(String method, String uri, String version)
{
    @Override
    public String toString()
    {
        return "{" + method + " " + uri + " " + version + "}";
    }
}
