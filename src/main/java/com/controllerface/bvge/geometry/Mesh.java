package com.controllerface.bvge.geometry;

import java.util.List;

public record Mesh(Vertex[] vertices, Face[] faces, List<BoneOffset> bone_offsets, Models.SceneNode sceneNode, int[] hull){ }
