package com.controllerface.bvge.geometry;

import org.joml.Matrix4f;
import org.lwjgl.PointerBuffer;
import org.lwjgl.assimp.*;

import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;

import static org.lwjgl.assimp.Assimp.*;
import static org.lwjgl.assimp.Assimp.aiImportFile;

public class Models
{
    private static final AtomicInteger next_model_index = new AtomicInteger(0);

    public static final int CIRCLE_MODEL = next_model_index.getAndIncrement();
    public static final int BOX_MODEL = next_model_index.getAndIncrement();
    public static final int POLYGON1_MODEL = next_model_index.getAndIncrement();

    public static int TEST_MODEL_INDEX = 99;

    private static final Map<Integer, Model> loaded_models = new HashMap<>();
    private static final Map<Integer, Boolean> dirty_models = new HashMap<>();
    private static final Map<Integer, Set<Integer>> model_instances = new HashMap<>();

    private static AIScene loadFile(String path)
    {
        int flags = aiProcess_JoinIdenticalVertices | aiProcess_Triangulate | aiProcess_FixInfacingNormals;
        return aiImportFile(path, flags);
    }

    private static void loadTestModel()
    {
        var aiScene = loadFile("C:/Users/Stephen/mdl/test_humanoid.fbx");
        int numMeshes = aiScene.mNumMeshes();
        var root_node = aiScene.mRootNode();

        Map<String, SceneNode> node_map = new HashMap<>();
        // todo: will need this for transforms, make the tree searchable so
        //  models can find their default transforms
        var sceneTree = processNodesHierarchy(root_node, null, node_map);

//        int children  = root_node.mNumChildren();
//        processBide
//        for (int i = 0; i < children; i++)
//        {
//            AINode next = AINode.create(root_node.mChildren(i));
//        }

        Mesh[] meshes = new Mesh[numMeshes];
        PointerBuffer aiMeshes = aiScene.mMeshes();
        for (int i = 0; i < numMeshes; i++)
        {
            AIMesh aiMesh = AIMesh.create(aiMeshes.get(i));
            var name = aiMesh.mName().dataString();
            var sceneNode = node_map.get(name);
            if (sceneNode == null)
            {
                throw new NullPointerException("No scene node for mesh: " + name
                    + " ensure node and geometry names match in blender");
            }
            System.out.println("\nMesh name: " + name);
            System.out.printf("verts: %d faces: %d \n",
                aiMesh.mNumVertices(),
                aiMesh.mNumFaces());

            int bone_count = aiMesh.mNumBones();
            PointerBuffer mBones = aiMesh.mBones();
            Bone mesh_bone = null;
            for (int j = 0; j < bone_count; j++)
            {
                AIBone bone = AIBone.create(mBones.get(j));
                if (bone.mNumWeights() > 0)
                {
                    if (mesh_bone != null)
                    {
                        throw new IllegalStateException("Multiple bones per mesh is not currently supported");
                    }
                    var mOffset = bone.mOffsetMatrix();
                    Matrix4f offset = new Matrix4f();
                    offset.set(mOffset.a1(), mOffset.a2(), mOffset.a3(), mOffset.a4(),
                        mOffset.b1(), mOffset.b2(), mOffset.b3(), mOffset.b4(),
                        mOffset.c1(), mOffset.c2(), mOffset.c3(), mOffset.c4(),
                        mOffset.d1(), mOffset.d2(), mOffset.d3(), mOffset.d4());
                    System.out.println("bone name: " + bone.mName().dataString());
                    System.out.println("bone weights: " + bone.mNumWeights());

                    int weight_index = 0;
                    AIVertexWeight.Buffer w_buf = bone.mWeights();
                    BoneWeight[] weights = new BoneWeight[bone.mNumWeights()];
                    while (w_buf.remaining() > 0)
                    {
                        AIVertexWeight weight = w_buf.get();
                        weights[weight_index++] = new BoneWeight(weight.mVertexId(), weight.mWeight());
                        System.out.println("vert id: " + weight.mVertexId() + " weight: " + weight.mWeight());
                    }

                    mesh_bone = new Bone(bone.mName().dataString(), offset, weights);
                    System.out.println("bone offset: \n" + offset);
                }
            }

            int vert_index = 0;
            Vertex[] vertices = new Vertex[aiMesh.mNumVertices()];
            AIVector3D.Buffer buffer = aiMesh.mVertices();
            while (buffer.remaining() > 0)
            {
                int this_vert = vert_index++;
                AIVector3D aiVertex = buffer.get();
                vertices[this_vert] = new Vertex(aiVertex.x(), aiVertex.y());
                System.out.printf("Vertex dump: vert id: %d x: %f y:%f\n", this_vert, aiVertex.x(), aiVertex.y());
            }

            int face_index = 0;
            Face[] faces = new Face[aiMesh.mNumFaces()];
            AIFace.Buffer buffer1 = aiMesh.mFaces();
            while (buffer1.remaining() > 0)
            {
                AIFace aiFace = buffer1.get();
                var b = aiFace.mIndices();
                List<Integer> indices = new ArrayList<>();
                for (int x = 0; x < aiFace.mNumIndices(); x++)
                {
                    int index = b.get(x);
                    indices.add(index);
                }
                faces[face_index++] = new Face(indices.get(0), indices.get(1), indices.get(2));
                System.out.printf("Face dump: raw: %s\n", indices);
            }

            var new_mesh = new Mesh(vertices, faces, mesh_bone, sceneNode);
            int new_index = Meshes.register_mesh(name, new_mesh);
            meshes[i] = new_mesh;
            System.out.printf("registered mesh [%s] with id [%d]", name, new_index);
        }

        loaded_models.put(TEST_MODEL_INDEX, new Model(meshes));
        System.out.println("\nLoaded model: " + TEST_MODEL_INDEX + " with " + meshes.length + " meshes");
    }

    public static Model get_model_by_index(int index)
    {
        return loaded_models.get(index);
    }

    public static void register_model_instance(int model_id, int hull_id)
    {
        model_instances.computeIfAbsent(model_id, _k -> new HashSet<>()).add(hull_id);
        dirty_models.put(model_id, true);
    }

    public static Set<Integer> get_model_instances(int model_id)
    {
        return model_instances.get(model_id);
    }

    public static boolean is_model_dirty(int model_id)
    {
        var r = dirty_models.get(model_id);
        return r != null && r;
    }

    public static int get_instance_count(int model_id)
    {
        return model_instances.get(model_id).size();
    }

    public static void set_model_clean(int model_id)
    {
        dirty_models.put(model_id, false);
    }

    public static void init()
    {
        loaded_models.put(CIRCLE_MODEL, Model.fromBasicMesh(Meshes.get_mesh_by_index(Meshes.CIRCLE_MESH)));
        loaded_models.put(BOX_MODEL, Model.fromBasicMesh(Meshes.get_mesh_by_index(Meshes.BOX_MESH)));
        loaded_models.put(POLYGON1_MODEL, Model.fromBasicMesh(Meshes.get_mesh_by_index(Meshes.POLYGON1_MESH)));
        loadTestModel();
    }


    private static SceneNode processNodesHierarchy(AINode aiNode, SceneNode parentNode, Map<String, SceneNode> nodeMap)
    {
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

        SceneNode currentNode = new SceneNode(nodeName, parentNode, transform);
        nodeMap.put(nodeName, currentNode);
        int numChildren = aiNode.mNumChildren();
        PointerBuffer aiChildren = aiNode.mChildren();
        for (int i = 0; i < numChildren; i++)
        {
            AINode aiChildNode = AINode.create(aiChildren.get(i));
            SceneNode childNode = processNodesHierarchy(aiChildNode, currentNode, nodeMap);
            currentNode.addChild(childNode);
        }

        return currentNode;
    }

    public static class SceneNode
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
}
