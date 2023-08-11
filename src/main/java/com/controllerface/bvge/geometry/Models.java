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
    public static final int CRATE_MODEL = next_model_index.getAndIncrement();
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

        var scene_node = processNodeHierarchy(root_node, null, node_map);

        Mesh[] meshes = new Mesh[numMeshes];
        PointerBuffer aiMeshes = aiScene.mMeshes();
        Map<String, Bone> bone_map = new HashMap<>();
        for (int i = 0; i < numMeshes; i++)
        {
            AIMesh aiMesh = AIMesh.create(aiMeshes.get(i));
            var name = aiMesh.mName().dataString();
            var mesh_node = node_map.get(name);
            if (mesh_node == null)
            {
                throw new NullPointerException("No scene node for mesh: " + name
                    + " ensure node and geometry names match in blender");
            }

            int bone_count = aiMesh.mNumBones();
            PointerBuffer mBones = aiMesh.mBones();
            Bone mesh_bone = null;
            for (int j = 0; j < bone_count; j++)
            {
                AIBone bone = AIBone.create(mBones.get(j));
                if (bone.mNumWeights() > 0)
                {
                    var bone_name = bone.mName().dataString();
                    if (mesh_bone != null)
                    {
                        throw new IllegalStateException("Multiple bones per mesh is not currently supported");
                    }
                    var mOffset = bone.mOffsetMatrix();
                    Matrix4f offset = new Matrix4f();
                    offset.set(mOffset.a1(), mOffset.b1(), mOffset.c1(), mOffset.d1(),
                        mOffset.a2(), mOffset.b2(), mOffset.c2(), mOffset.d2(),
                        mOffset.a3(), mOffset.b3(), mOffset.c3(), mOffset.d3(),
                        mOffset.a4(), mOffset.b4(), mOffset.c4(), mOffset.d4());

                    int weight_index = 0;
                    AIVertexWeight.Buffer w_buf = bone.mWeights();
                    BoneWeight[] weights = new BoneWeight[bone.mNumWeights()];
                    while (w_buf.remaining() > 0)
                    {
                        AIVertexWeight weight = w_buf.get();
                        weights[weight_index++] = new BoneWeight(weight.mVertexId(), weight.mWeight());
                    }

                    var bone_node = node_map.get(bone_name);
                    if (bone_node == null)
                    {
                        throw new NullPointerException("No scene node for bone: " + bone_name
                            + " ensure node and geometry names match in blender");
                    }
                    mesh_bone = new Bone(bone_name, offset, weights, bone_node);
                    bone_map.put(bone_name, mesh_bone);
                }
            }

            int vert_index = 0;
            var mesh_vertices = new Vertex[aiMesh.mNumVertices()];
            var buffer = aiMesh.mVertices();
            while (buffer.remaining() > 0)
            {
                int this_vert = vert_index++;
                var aiVertex = buffer.get();
                mesh_vertices[this_vert] = new Vertex(aiVertex.x(), aiVertex.y());
            }

            int face_index = 0;
            var mesh_faces = new Face[aiMesh.mNumFaces()];
            var face_buffer = aiMesh.mFaces();
            while (face_buffer.remaining() > 0)
            {
                var aiFace = face_buffer.get();
                var b = aiFace.mIndices();
                var indices = new ArrayList<Integer>();
                for (int x = 0; x < aiFace.mNumIndices(); x++)
                {
                    int index = b.get(x);
                    indices.add(index);
                }
                mesh_faces[face_index++] = new Face(indices.get(0), indices.get(1), indices.get(2));
            }

            var new_mesh = new Mesh(mesh_vertices, mesh_faces, mesh_bone, mesh_node);
            int new_index = Meshes.register_mesh(name, new_mesh);
            meshes[i] = new_mesh;
            //System.out.printf("registered mesh [%s] with id [%d]\n", name, new_index);
        }

        int root_index = -1;
        for (int mi = 0; mi < meshes.length; mi++)
        {
            if (meshes[mi].bone().sceneNode().parent.parent.name.equalsIgnoreCase("RootNode"))
            {
                root_index = mi;
                break;
            }
        }

        if (root_index == -1)
        {
            throw new IllegalStateException("No root mesh found. " +
                "Root mesh is determined by root bone in Armature under RootNode in scene");
        }

        Map<String, Matrix4f> bone_transforms = new HashMap<>();
        generate_transforms(scene_node, bone_map, bone_transforms, new Matrix4f());

        loaded_models.put(TEST_MODEL_INDEX, new Model(meshes, bone_map, bone_transforms, root_index));
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
        loaded_models.put(CRATE_MODEL, Model.fromBasicMesh(Meshes.get_mesh_by_index(Meshes.BOX_MESH)));
        loaded_models.put(POLYGON1_MODEL, Model.fromBasicMesh(Meshes.get_mesh_by_index(Meshes.POLYGON1_MESH)));
        loadTestModel();
    }


    private static SceneNode processNodeHierarchy(AINode aiNode, SceneNode parentNode, Map<String, SceneNode> nodeMap)
    {
        var nodeName = aiNode.mName().dataString();
        var mTransform = aiNode.mTransformation();
        var transform = new Matrix4f();
        transform.set(mTransform.a1(), mTransform.b1(), mTransform.c1(), mTransform.d1(),
            mTransform.a2(), mTransform.b2(), mTransform.c2(), mTransform.d2(),
            mTransform.a3(), mTransform.b3(), mTransform.c3(), mTransform.d3(),
            mTransform.a4(), mTransform.b4(), mTransform.c4(), mTransform.d4());

        var currentNode = new SceneNode(nodeName, parentNode, transform);
        System.out.println("DEBUG name: " + nodeName);
        nodeMap.put(nodeName, currentNode);
        int numChildren = aiNode.mNumChildren();
        var aiChildren = aiNode.mChildren();
        for (int i = 0; i < numChildren; i++)
        {
            var aiChildNode = AINode.create(aiChildren.get(i));
            var childNode = processNodeHierarchy(aiChildNode, currentNode, nodeMap);
            currentNode.addChild(childNode);
        }

        return currentNode;
    }

    // todo: this method or a variant needs to be callable in the game loop, this will be required
    //  to add animations. the
    private static void generate_transforms(SceneNode current_node,
                                            Map<String, Bone> bone_map,
                                            Map<String, Matrix4f> transforms,
                                            Matrix4f parent_transform)
    {
        var name = current_node.name;
        var node_transform = current_node.transform;
        var global_transform = parent_transform.mul(node_transform, new Matrix4f());

        // if this node is a bone, update the
        if (bone_map.containsKey(name))
        {
            var bone = bone_map.get(name);
            transforms.put(name, global_transform.mul(bone.offset(), new Matrix4f()));
        }
        current_node.children
            .forEach(child -> generate_transforms(child, bone_map, transforms, global_transform));
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
