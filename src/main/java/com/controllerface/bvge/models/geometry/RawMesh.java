package com.controllerface.bvge.models.geometry;

/**
 * A "raw" version of a Mesh object. Raw copies are useful in renderers, where mesh data must be
 * loaded into primitive arrays.
 */
public record RawMesh(float[] vertices, float[] uvs, int[] faces) { }
