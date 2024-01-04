package com.controllerface.bvge.geometry;

public record Mesh(Vertex[] vertices, Face[] faces, MeshBone bone, Models.SceneNode sceneNode, int[] hull){ }
