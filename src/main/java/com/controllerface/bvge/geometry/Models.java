package com.controllerface.bvge.geometry;

import com.controllerface.bvge.Main;
import com.controllerface.bvge.data.PhysicsObjects;
import org.joml.Matrix4f;
import org.lwjgl.PointerBuffer;
import org.lwjgl.assimp.*;
import org.lwjgl.system.MemoryUtil;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;

import static org.lwjgl.assimp.Assimp.*;
import static org.lwjgl.assimp.Assimp.aiImportFile;

public class Models
{
    private static final AtomicInteger next_model_index = new AtomicInteger(0);

    public static final int CIRCLE_MODEL = next_model_index.getAndIncrement();
    public static final int TRIANGLE_MODEL = next_model_index.getAndIncrement();
    public static final int CRATE_MODEL = next_model_index.getAndIncrement();
    public static final int POLYGON1_MODEL = next_model_index.getAndIncrement();

    public static int TEST_MODEL_INDEX = -1;
    public static int TEST_SQUARE_INDEX = -1;

    private static final Map<Integer, Model> loaded_models = new HashMap<>();
    private static final Map<Integer, Boolean> dirty_models = new HashMap<>();
    private static final Map<Integer, Set<Integer>> model_instances = new HashMap<>();

    private static AIScene loadFile(String path)
    {
        int flags = aiProcess_JoinIdenticalVertices | aiProcess_Triangulate | aiProcess_FixInfacingNormals;
        return aiImportFile(path, flags);
    }

    private static AIScene loadModelResource(String name) throws IOException
    {
        var model_stream = Models.class.getResourceAsStream(name);
        var model_data = model_stream.readAllBytes();
        ByteBuffer data = MemoryUtil.memCalloc(model_data.length);
        data.put(model_data);
        data.flip();
        int flags = aiProcess_JoinIdenticalVertices | aiProcess_Triangulate | aiProcess_FixInfacingNormals;
        return aiImportFileFromMemory(data, flags, "");
    }

    private static Bone load_bone(AIMesh aiMesh,
                                  Map<String, SceneNode> node_map,
                                  Map<String, Bone> bone_map,
                                  Map<Integer, Float> bone_weight_map)
    {
        int bone_count = aiMesh.mNumBones();
        PointerBuffer bone_buffer = aiMesh.mBones();
        Bone mesh_bone = null;

        for (int bone_index = 0; bone_index < bone_count; bone_index++)
        {
            var raw_bone = AIBone.create(bone_buffer.get(bone_index));
            boolean expect_empty = mesh_bone != null;
            var next_bone = load_raw_bone(raw_bone, node_map, bone_map, bone_weight_map, expect_empty);
            if (next_bone != null)
            {
                mesh_bone = next_bone;
            }
        }

        if (mesh_bone == null)
        {
            throw new NullPointerException("No bone for mesh: " + aiMesh.mName().dataString()
                + " ensure mesh has an assigned bone");
        }
        return mesh_bone;
    }

    private static Bone load_raw_bone(AIBone raw_bone,
                                      Map<String, SceneNode> node_map,
                                      Map<String, Bone> bone_map,
                                      Map<Integer, Float> bone_weight_map,
                                      boolean expect_empty)
    {
        Bone bone;
        if (raw_bone.mNumWeights() <= 0)
        {
            return null;
        }
        if (expect_empty)
        {
            throw new IllegalStateException("Multiple bones per mesh is not currently supported");
        }

        var bone_name = raw_bone.mName().dataString();
        var mOffset = raw_bone.mOffsetMatrix();
        Matrix4f offset = new Matrix4f();
        var raw_matrix = new float[16];
        raw_matrix[0]  = mOffset.a1();
        raw_matrix[1]  = mOffset.b1();
        raw_matrix[2]  = mOffset.c1();
        raw_matrix[3]  = mOffset.d1();
        raw_matrix[4]  = mOffset.a2();
        raw_matrix[5]  = mOffset.b2();
        raw_matrix[6]  = mOffset.c2();
        raw_matrix[7]  = mOffset.d2();
        raw_matrix[8]  = mOffset.a3();
        raw_matrix[9]  = mOffset.b3();
        raw_matrix[10] = mOffset.c3();
        raw_matrix[11] = mOffset.d3();
        raw_matrix[12] = mOffset.a4();
        raw_matrix[13] = mOffset.b4();
        raw_matrix[14] = mOffset.c4();
        raw_matrix[15] = mOffset.d4();
        offset.set(raw_matrix);

        int weight_index = 0;
        AIVertexWeight.Buffer w_buf = raw_bone.mWeights();
        BoneWeight[] weights = new BoneWeight[raw_bone.mNumWeights()];
        while (w_buf.remaining() > 0)
        {
            AIVertexWeight weight = w_buf.get();
            weights[weight_index++] = new BoneWeight(weight.mVertexId(), weight.mWeight());
            bone_weight_map.put(weight.mVertexId(), weight.mWeight());
        }

        var bone_node = node_map.get(bone_name);
        if (bone_node == null)
        {
            throw new NullPointerException("No scene node for bone: " + bone_name
                + " ensure node and geometry names match in blender");
        }

        int bone_ref_id = Main.Memory.new_bone_reference(raw_matrix);
        bone = new Bone(bone_ref_id, bone_name, offset, weights, bone_node);
        bone_map.put(bone_name, bone);
        return bone;
    }

    private static Face[] load_faces(AIMesh aiMesh)
    {
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
        return mesh_faces;
    }

    private static Vertex[] load_vertices(AIMesh aiMesh,
                                          Map<Integer, Float> bone_weight_map,
                                          Bone mesh_bone)
    {
        int vert_index = 0;
        var mesh_vertices = new Vertex[aiMesh.mNumVertices()];
        var buffer = aiMesh.mVertices();
        var tex = aiMesh.mTextureCoords(0);
        // todo: add texture co-ordinates to the Vertex class and extract them here. requires
        //  a model with a texture attached
        while (buffer.remaining() > 0)
        {
            int this_vert = vert_index++;
            var aiVertex = buffer.get();
            if (tex != null)
            {
                // todo: this needs to check if the text co-ord buffer is empty, not null. Seems
                //  it is always some value.

                // todo: need to update the base model to have UV mappings and a texture
                //AIBone.create(bone_buffer.get(bone_index));
                //var t = AIVector3D.create(tex.get());
                //System.out.println("test");
            }
            float bone_weight = bone_weight_map.get(this_vert);
            var vert_ref_id = Main.Memory.new_vertex_reference(aiVertex.x(), aiVertex.y());
            //System.out.printf("DEBUG CPU (in): id: %d x:%f y:%f\n", vert_ref_id, aiVertex.x(), aiVertex.y());
            mesh_vertices[this_vert] = new Vertex(vert_ref_id, aiVertex.x(), aiVertex.y(), mesh_bone.name(), bone_weight);
        }
        return mesh_vertices;
    }

    private static void load_mesh(int mesh_index,
                                  String model_name,
                                  AIMesh raw_mesh,
                                  Mesh[] meshes,
                                  Map<String, SceneNode> node_map,
                                  Map<String, Bone> bone_map)
    {
        var mesh_name = raw_mesh.mName().dataString();
        var mesh_node = node_map.get(mesh_name);
        if (mesh_node == null)
        {
            throw new NullPointerException("No scene node for mesh: " + mesh_name
                + " ensure node and geometry names match in blender");
        }
        var bone_weight_map = new HashMap<Integer, Float>();
        var mesh_bone = load_bone(raw_mesh, node_map, bone_map, bone_weight_map);
        var mesh_vertices = load_vertices(raw_mesh, bone_weight_map, mesh_bone);
        var mesh_faces = load_faces(raw_mesh);
        var hull_table = PhysicsObjects.calculate_convex_hull_table(mesh_vertices);
        var new_mesh = new Mesh(mesh_vertices, mesh_faces, mesh_bone, mesh_node, hull_table);

        // todo: generate the convex hull here, just once and re-use later

        Meshes.register_mesh(model_name, mesh_name, new_mesh);
        meshes[mesh_index] = new_mesh;
    }

    private static void load_raw_meshes(int numMeshes,
                                        String model_name,
                                        Mesh[] meshes,
                                        PointerBuffer mesh_buffer,
                                        Map<String, SceneNode> node_map,
                                        Map<String, Bone> bone_map)
    {
        // load raw mesh data for all meshes
        for (int i = 0; i < numMeshes; i++)
        {
            var raw_mesh = AIMesh.create(mesh_buffer.get(i));
            load_mesh(i, model_name, raw_mesh, meshes, node_map, bone_map);
        }
    }

    private static int find_root_index(Mesh[] meshes)
    {
        int root_index = -1;
        for (int mi = 0; mi < meshes.length; mi++)
        {
            // note the chained parent call, the logic is that we find the first bone that is a direct
            // descendant of the armature, which is itself a child of the root scene node, which is
            // given the default name "RootNode".
            var grandparent = meshes[mi].bone().sceneNode().parent.parent;
            if (grandparent.name.equalsIgnoreCase("RootNode"))
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

        return root_index;
    }

    private static int load_model(String model_path, String model_name)
    {
        // the number of meshes associated with the loaded model
        int numMeshes;

        // the root node of the imported file. contains raw mesh buffer data
        AINode root_node;

        // a parsed version of the raw root node and associated data, in a tree structure
        SceneNode scene_node;

        // this is the raw mesh buffer extracted fom the main scene data. it is parsed for meshes
        PointerBuffer mesh_buffer;

        // an array to hold all the parsed meshes
        Mesh[] meshes;

        // used to map nodes by their node names, more convenient than tree for loading process
        var node_map = new HashMap<String, SceneNode>();

        // similar to the node map, though this map is returned in the output value
        var bone_map = new HashMap<String, Bone>();

        // maps each bone to a calculated bind pose transformation matrix
        var bone_transforms = new HashMap<String, Matrix4f>();

        // read initial data
        try (AIScene aiScene = loadModelResource(model_path))
        {
            numMeshes = aiScene.mNumMeshes();
            root_node = aiScene.mRootNode();
            scene_node = process_node_hierarchy(root_node, null, node_map);
            meshes = new Mesh[numMeshes];
            mesh_buffer = aiScene.mMeshes();
        }
        catch (NullPointerException | IOException e)
        {
            throw new RuntimeException("Unable to load model data", e);
        }

        load_raw_meshes(numMeshes, model_name, meshes, mesh_buffer, node_map, bone_map);

        // we need to calculate the root node for the body, which is the mesh that is tracking the root bone.
        // the root bone is determined by checking if the current bone's parent is a direct descendant of the scene
        int root_index = find_root_index(meshes);

        // generate the bind pose transforms, setting the initial state of the armature
        generate_transforms(scene_node, bone_map, bone_transforms, new Matrix4f());

        // register the model
        var next_model_id = next_model_index.getAndIncrement();
        var model = new Model(meshes, bone_map, bone_transforms, root_index);
        loaded_models.put(next_model_id, model);
        return next_model_id;
    }

    public static Model get_model_by_index(int index)
    {
        return loaded_models.get(index);
    }

    // todo: need to move to just tracking a count of models and not holding onto their
    //  object ids. Instead, renderers should use a CL call to set up batches. Main
    //  memory segments will still be kept at accurate counts, so batching logic can
    //  still work the same as it currently does.

    // todo: need a way to decrement a model count when an instance is deleted

    public static void register_model_instance(int model_id, int object_id)
    {
        model_instances.computeIfAbsent(model_id, _k -> new HashSet<>()).add(object_id);
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
        loaded_models.put(TRIANGLE_MODEL, Model.fromBasicMesh(Meshes.get_mesh_by_index(Meshes.TRIANGLE_MESH)));
        loaded_models.put(CRATE_MODEL, Model.fromBasicMesh(Meshes.get_mesh_by_index(Meshes.BOX_MESH)));
        loaded_models.put(POLYGON1_MODEL, Model.fromBasicMesh(Meshes.get_mesh_by_index(Meshes.POLYGON1_MESH)));
        TEST_MODEL_INDEX = load_model("/models/test_humanoid.fbx", "Humanoid");
        TEST_SQUARE_INDEX = load_model("/models/test_square.fbx", "Crate");
    }

    private static SceneNode process_node_hierarchy(AINode aiNode, SceneNode parentNode, Map<String, SceneNode> nodeMap)
    {
        var nodeName = aiNode.mName().dataString();
        var mTransform = aiNode.mTransformation();
        var transform = new Matrix4f();
        transform.set(mTransform.a1(), mTransform.b1(), mTransform.c1(), mTransform.d1(),
            mTransform.a2(), mTransform.b2(), mTransform.c2(), mTransform.d2(),
            mTransform.a3(), mTransform.b3(), mTransform.c3(), mTransform.d3(),
            mTransform.a4(), mTransform.b4(), mTransform.c4(), mTransform.d4());

        var currentNode = new SceneNode(nodeName, parentNode, transform);
        nodeMap.put(nodeName, currentNode);
        int numChildren = aiNode.mNumChildren();
        var aiChildren = aiNode.mChildren();
        for (int i = 0; i < numChildren; i++)
        {
            var aiChildNode = AINode.create(aiChildren.get(i));
            var childNode = process_node_hierarchy(aiChildNode, currentNode, nodeMap);
            currentNode.addChild(childNode);
        }

        return currentNode;
    }

    // todo: this method or a variant needs to be callable in the game loop, this will be required
    //  to add animations.
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
