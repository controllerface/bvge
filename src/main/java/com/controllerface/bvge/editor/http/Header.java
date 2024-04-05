package com.controllerface.bvge.editor.http;

public record Header(String name, String value)
{
    @Override
    public String toString()
    {
        return STR."[\{name} : \{value}]";
    }
}
