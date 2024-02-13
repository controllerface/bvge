package com.controllerface.bvge.animation;

import org.joml.Matrix4f;

public record BoneBindPose(int parent, Matrix4f transform, String bone_name) { }
