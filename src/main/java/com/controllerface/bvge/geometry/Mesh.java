package com.controllerface.bvge.geometry;

public record Mesh(Vertex[] vertices, Face[] faces, Bone bone, Models.SceneNode sceneNode, int[] hull){ }
