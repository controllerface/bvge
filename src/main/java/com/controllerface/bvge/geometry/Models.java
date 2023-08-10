package com.controllerface.bvge.geometry;

import org.joml.Matrix4f;
import org.lwjgl.PointerBuffer;
import org.lwjgl.assimp.*;

import java.util.ArrayList;
import java.util.List;

import static org.lwjgl.assimp.Assimp.*;
import static org.lwjgl.assimp.Assimp.aiImportFile;

public class Models
{
    private static AIScene loadFile(String path)
    {
        int flags = aiProcess_JoinIdenticalVertices | aiProcess_Triangulate | aiProcess_FixInfacingNormals;
        return aiImportFile(path, flags);
    }


    private static void loadTestModel()
    {
        var aiScene = loadFile("C:/Users/Stephen/mdl/test_humanoid.fbx");
        int numMeshes = aiScene.mNumMeshes();
        var root_node  = aiScene.mRootNode();

        var newNode = processNodesHierarchy(root_node, null);

//        int children  = root_node.mNumChildren();
//        processBide
//        for (int i = 0; i < children; i++)
//        {
//            AINode next = AINode.create(root_node.mChildren(i));
//        }

        PointerBuffer aiMeshes = aiScene.mMeshes();
        for (int i = 0; i < numMeshes; i++) {
            AIMesh aiMesh = AIMesh.create(aiMeshes.get(i));

            var name = aiMesh.mName().dataString();
            System.out.println("\nMesh name: " + name);
            System.out.printf("verts: %d faces: %d \n",
                aiMesh.mNumVertices(),
                aiMesh.mNumFaces());

            int bone_count = aiMesh.mNumBones();
            PointerBuffer mBones = aiMesh.mBones();
            for (int j = 0; j < bone_count; j++)
            {
                AIBone bone = AIBone.create(mBones.get(j));
                if (bone.mNumWeights() > 0)
                {
                    var mOffset = bone.mOffsetMatrix();
                    Matrix4f offset = new Matrix4f();
                    offset.set(mOffset.a1(), mOffset.a2(), mOffset.a3(), mOffset.a4(),
                        mOffset.b1(), mOffset.b2(), mOffset.b3(), mOffset.b4(),
                        mOffset.c1(), mOffset.c2(), mOffset.c3(), mOffset.c4(),
                        mOffset.d1(), mOffset.d2(), mOffset.d3(), mOffset.d4());
                    System.out.println("bone name: " + bone.mName().dataString());
                    System.out.println("bone weights: " + bone.mNumWeights());
                    System.out.println("bone offset: \n" + offset);
                }
            }

            AIVector3D.Buffer buffer = aiMesh.mVertices();
            while (buffer.remaining() > 0) {
                AIVector3D aiVertex = buffer.get();
                System.out.printf("Vertex dump: x: %f y:%f\n", aiVertex.x(), aiVertex.y());
            }

            AIFace.Buffer buffer1 = aiMesh.mFaces();
            while (buffer1.remaining() > 0) {
                AIFace aiFace = buffer1.get();
                var b = aiFace.mIndices();
                List<Integer> indices = new ArrayList<>();
                for (int x = 0; x < aiFace.mNumIndices(); x++)
                {
                    int index = b.get(x);
                    indices.add(index);
                }
                System.out.printf("Face dump: raw: %s\n", indices);

            }

            //System.out.println("bone count: " + aiMesh.mNumBones());
        }
    }


    public static void init()
    {
        loadTestModel();
    }




    private static Node processNodesHierarchy(AINode aiNode, Node parentNode) {
        String nodeName = aiNode.mName().dataString();
        var mTransform = aiNode.mTransformation();

        // this is a transform object we can use
        Matrix4f transform = new Matrix4f();
        transform.set(mTransform.a1(), mTransform.a2(), mTransform.a3(), mTransform.a4(),
            mTransform.b1(), mTransform.b2(), mTransform.b3(), mTransform.b4(),
            mTransform.c1(), mTransform.c2(), mTransform.c3(), mTransform.c4(),
            mTransform.d1(), mTransform.d2(), mTransform.d3(), mTransform.d4());

        System.out.println("Node: " + nodeName);
        System.out.println(transform);


        Node node = new Node(nodeName, parentNode);
        int numChildren = aiNode.mNumChildren();
        PointerBuffer aiChildren = aiNode.mChildren();
        for (int i = 0; i < numChildren; i++) {
            AINode aiChildNode = AINode.create(aiChildren.get(i));
            Node childNode = processNodesHierarchy(aiChildNode, node);
            node.addChild(childNode);
        }

        return node;
    }

    private static class Node
    {
        private final String name;
        private final Node parent;
        private final List<Node> children = new ArrayList<>();

        private Node(String name, Node parent)
        {
            this.name = name;
            this.parent = parent;
        }

        private void addChild(Node child)
        {
            children.add(child);
        }
    }
}
