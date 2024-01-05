package com.controllerface.bvge.geometry;

import java.util.List;

public record Mesh(Vertex[] vertices, Face[] faces, List<MeshBone> bone, Models.SceneNode sceneNode, int[] hull){ }
