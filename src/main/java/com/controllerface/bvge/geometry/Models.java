package com.controllerface.bvge.geometry;

import com.controllerface.bvge.Main;
import com.controllerface.bvge.ecs.systems.physics.PhysicsObjects;
import com.controllerface.bvge.gl.Texture;
import com.controllerface.bvge.util.Assets;
import org.joml.Matrix4f;
import org.joml.Vector2f;
import org.lwjgl.PointerBuffer;
import org.lwjgl.assimp.*;
import org.lwjgl.system.MemoryUtil;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;

import static org.lwjgl.assimp.Assimp.*;

public class Models
{
    private static final AtomicInteger next_model_index = new AtomicInteger(0);

    public static final int CIRCLE_PARTICLE = next_model_index.getAndIncrement();
    public static final int TRIANGLE_PARTICLE = next_model_index.getAndIncrement();
    public static final int SQUARE_PARTICLE = next_model_index.getAndIncrement();
    public static final int POLYGON1_MODEL = next_model_index.getAndIncrement();

    public static int TEST_MODEL_INDEX = -1;
    public static int TEST_SQUARE_INDEX = -1;

    private static final Map<Integer, Model> loaded_models = new HashMap<>();
    private static final Map<Integer, Boolean> dirty_models = new HashMap<>();
    private static final Map<Integer, Set<Integer>> model_instances = new HashMap<>();

    private static AIScene loadModelResource(String name) throws IOException
    {
        var model_stream = Models.class.getResourceAsStream(name);
        var model_data = model_stream.readAllBytes();
        ByteBuffer data = MemoryUtil.memCalloc(model_data.length);
        data.put(model_data);
        data.flip();
        int flags = aiProcess_JoinIdenticalVertices | aiProcess_Triangulate | aiProcess_FixInfacingNormals;
        var imported = aiImportFileFromMemory(data, flags, "");
        MemoryUtil.memFree(data);
        return imported;
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
        raw_matrix[0] = mOffset.a1();
        raw_matrix[1] = mOffset.b1();
        raw_matrix[2] = mOffset.c1();
        raw_matrix[3] = mOffset.d1();
        raw_matrix[4] = mOffset.a2();
        raw_matrix[5] = mOffset.b2();
        raw_matrix[6] = mOffset.c2();
        raw_matrix[7] = mOffset.d2();
        raw_matrix[8] = mOffset.a3();
        raw_matrix[9] = mOffset.b3();
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

        List<List<AIVector3D>> uvChannels = new ArrayList<>();
        for (int i = 0; i < AI_MAX_NUMBER_OF_TEXTURECOORDS; i++)
        {
            var uvBuffer = aiMesh.mTextureCoords(i);
            if (uvBuffer != null)
            {
                List<AIVector3D> currentChannel = new ArrayList<>();
                while (uvBuffer.remaining() > 0)
                {
                    var aiVector = uvBuffer.get();
                    currentChannel.add(aiVector);
                }
                uvChannels.add(currentChannel);
            }
        }

        AtomicInteger count = new AtomicInteger();
        while (buffer.remaining() > 0)
        {
            int this_vert = vert_index++;
            var aiVertex = buffer.get();
            List<Vector2f> uvData = new ArrayList<>();
            float bone_weight = bone_weight_map.get(this_vert);
            var vert_ref_id = Main.Memory.new_vertex_reference(aiVertex.x(), aiVertex.y());

            uvChannels.forEach(channel ->
            {
                var next = channel.get(count.get());
                uvData.add(new Vector2f(next.x(), next.y()));
            });

            mesh_vertices[this_vert] = new Vertex(vert_ref_id, aiVertex.x(), aiVertex.y(), uvData, mesh_bone.name(), bone_weight);
            count.getAndIncrement();
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

        //System.out.println("Debug mat index:" + raw_mesh.mMaterialIndex() + " for: " + mesh_name);

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

    // todo: material list needs to be built and made available during mesh loading,
    //  and each mesh should have it's material set during construction.
    private static void load_materials(AIScene aiScene)
    {
        if (aiScene.mNumMaterials() <= 0)
        {
            return;
        }

        var buffer = aiScene.mMaterials();

        for (int i = 0; i < aiScene.mNumMaterials(); i++)
        {
            var raw_mat = AIMaterial.create(buffer.get(i));
            var p_buf = raw_mat.mProperties();
            for (int j = 0; j < raw_mat.mNumProperties(); j++)
            {
                var raw_prop = AIMaterialProperty.create(p_buf.get(j));
                var prop_name = raw_prop.mKey().dataString();

                if (prop_name.startsWith("$clr."))
                {
                    var color_data = raw_prop.mData().asFloatBuffer();
                    float r = color_data.get(0);
                    float g = color_data.get(1);
                    float b = color_data.get(2);
                    System.out.println("Mat name=" + prop_name + " r=" + r + " g=" + g + " b=" + b);
                }
                else if (prop_name.startsWith("$mat."))
                {
                    float v = raw_prop.mData().asFloatBuffer().get(0);
                    System.out.println("Mat name=" + prop_name + " v=" + v);
                }
                else
                {
                    switch (raw_prop.mType())
                    {
                        case 1:
                            var float_buffer = raw_prop.mData().asFloatBuffer();
                            int f_count = raw_prop.mDataLength() / 4;
                            float[] float_out = new float[f_count];
                            float_buffer.get(float_out);
                            System.out.println("Mat name=" + prop_name + " float=" + Arrays.toString(float_out));
                            break;

                        case 3:
                            var string_buffer = raw_prop.mData();
                            int s_count = raw_prop.mDataLength();
                            byte[] bytes_out = new byte[s_count];
                            string_buffer.get(bytes_out);
                            var string = new String(bytes_out, StandardCharsets.UTF_8);
                            System.out.println("Mat name=" + prop_name + " string=" + string);
                            break;

                        case 4:
                            var int_buffer = raw_prop.mData().asIntBuffer();
                            int i_count = raw_prop.mDataLength() / 4;
                            int[] int_out = new int[i_count];
                            int_buffer.get(int_out);
                            System.out.println("Mat name=" + prop_name + " int=" + Arrays.toString(int_out));
                            break;

                        default:
                            System.out.println("Debug mat prop:"
                                + " type=" + raw_prop.mType()
                                + " len=" + raw_prop.mDataLength()
                                + " key=" + raw_prop.mKey().dataString());
                            break;
                    }
                }
            }

        }
    }

    private static void load_textures(AIScene aiScene, List<Texture> textures)
    {
        if (aiScene.mNumTextures() <= 0)
        {
            return;
        }

        var texture_buffer = aiScene.mTextures();
        for (int tex_index = 0; tex_index < aiScene.mNumTextures(); tex_index++)
        {
            var raw_texture = AITexture.create(texture_buffer.get(tex_index));
            textures.add(Assets.load_texture(raw_texture));
        }
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

        // if textures are present in the model, they are loaded into this list
        var textures = new ArrayList<Texture>();

        // read initial data
        try (AIScene aiScene = loadModelResource(model_path))
        {
            numMeshes = aiScene.mNumMeshes();
            root_node = aiScene.mRootNode();
            scene_node = process_node_hierarchy(root_node, null, node_map);
            meshes = new Mesh[numMeshes];
            mesh_buffer = aiScene.mMeshes();
            load_textures(aiScene, textures);
            load_materials(aiScene);
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
        var model = new Model(meshes, bone_map, bone_transforms, textures, root_index);
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
        return model_instances.get(model_id) == null
            ? 0
            : model_instances.get(model_id).size();
    }

    public static void set_model_clean(int model_id)
    {
        dirty_models.put(model_id, false);
    }

    public static void init()
    {
        loaded_models.put(CIRCLE_PARTICLE, Model.fromBasicMesh(Meshes.get_mesh_by_index(Meshes.CIRCLE_MESH)));
        loaded_models.put(TRIANGLE_PARTICLE, Model.fromBasicMesh(Meshes.get_mesh_by_index(Meshes.TRIANGLE_MESH)));
        loaded_models.put(SQUARE_PARTICLE, Model.fromBasicMesh(Meshes.get_mesh_by_index(Meshes.BOX_MESH)));
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
    //  to add animations. The method will need to derive the current transform matrix by doing a
    //  lerp between the most recent and next keyframes.
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
        for (SceneNode child : current_node.children)
        {
            generate_transforms(child, bone_map, transforms, global_transform);
        }
    }

    /**
     * A container class used for storing a tree of nodes, as is present in a model with an armature.
     * Typically, a model is defined with some starting mesh, and child nodes beneath that mesh that
     * contain more meshes. Bones are defined in a similar way, in fact the bone structure will generally
     * be used to define the actual structure of the model when loaded into memory, hierarchy of the
     * meshes themselves in the data is largely irrelevant.
     */
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
