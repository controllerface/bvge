package com.controllerface.bvge.editor.http;

import java.util.List;

public record Request(RequestLine request_line, List<Header> headers)
{
    @Override
    public String toString()
    {
        var buffer = new StringBuilder();
        buffer.append("\n").append(request_line.toString()).append("\n");
        headers.forEach(header -> buffer.append(header).append("\n"));
        buffer.append("\n");
        return buffer.toString();
    }

    public String method()
    {
        return request_line().method();
    }

    public String uri()
    {
        return request_line().uri();
    }
}
