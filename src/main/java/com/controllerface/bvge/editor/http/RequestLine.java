package com.controllerface.bvge.editor.http;

public record RequestLine(String method, String uri, String version)
{
    @Override
    public String toString()
    {
        return STR."[\{method} \{uri} \{version}]";
    }
}