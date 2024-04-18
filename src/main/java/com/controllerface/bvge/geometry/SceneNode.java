package com.controllerface.bvge.geometry;

import org.joml.Matrix4f;

import java.util.ArrayList;
import java.util.List;

/**
 * A container class used for storing a tree of nodes, as is present in a model with an armature.
 * Typically, a model is defined with some starting mesh, and child nodes beneath that mesh that
 * contain more meshes. Bones are defined in a similar way, in fact the bone structure will generally
 * be used to define the actual structure of the model when loaded into memory, hierarchy of the
 * meshes themselves in the data is largely irrelevant.
 */
public class SceneNode
{
    public final String name;
    public final SceneNode parent;
    public final Matrix4f transform;
    public final List<SceneNode> children = new ArrayList<>();

    public SceneNode(String name, SceneNode parent, Matrix4f transform)
    {
        this.name = name;
        this.parent = parent;
        this.transform = transform;
    }

    public void addChild(SceneNode child)
    {
        children.add(child);
    }

    public static SceneNode empty()
    {
        return new SceneNode("", null, new Matrix4f());
    }
}
