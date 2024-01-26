package com.controllerface.bvge.geometry;

import com.controllerface.bvge.Main;
import com.controllerface.bvge.cl.GPU;
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
import java.util.stream.Stream;

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

    private static int load_model(String model_path, String model_name)
    {
        // the number of meshes associated with the loaded model
        int mesh_count;

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

        // if textures are present in the model, they are loaded into this list
        var textures = new ArrayList<Texture>();

        // read initial data
        try (AIScene aiScene = loadModelResource(model_path))
        {
            mesh_count = aiScene.mNumMeshes();
            root_node = aiScene.mRootNode();
            scene_node = process_node_hierarchy(root_node, null, node_map);
            meshes = new Mesh[mesh_count];
            mesh_buffer = aiScene.mMeshes();
            var anims = aiScene.mNumAnimations();
            System.out.println("anim count: " + anims);
            load_textures(aiScene, textures);
            load_materials(aiScene);
        }
        catch (NullPointerException | IOException e)
        {
            throw new RuntimeException("Unable to load model data", e);
        }

        // maps each bone to a calculated bind pose transformation matrix
        var bone_transforms = new HashMap<String, Matrix4f>();

        // this map must be linked to preserve insert order
        var bind_pose_map = new LinkedHashMap<Integer, BoneBindPose>();
        var bind_name_map = new HashMap<String, Integer>();

        // generate the bind pose transforms, setting the initial state of the armature
        generate_transforms(scene_node, bone_transforms, new Matrix4f(), bind_name_map, bind_pose_map, -1);

        load_raw_meshes(mesh_count, model_name, meshes, mesh_buffer, node_map);

        // we need to calculate the root node for the body, which is the mesh that is tracking the root bone.
        // the root bone is determined by checking if the current bone's parent is a direct descendant of the scene
        int root_index = find_root_index(meshes);

        // register the model
        var next_model_id = next_model_index.getAndIncrement();
        var model = new Model(meshes, bone_transforms, bind_name_map, bind_pose_map, textures, root_index);
        loaded_models.put(next_model_id, model);
        return next_model_id;
    }

    private static List<BoneOffset> load_mesh_bones(AIMesh aiMesh,
                                                    Map<String, SceneNode> node_map,
                                                    Map<Integer, String[]> mesh_bone_names,
                                                    Map<Integer, float[]> bone_weight_map)
    {
        var mesh_bone_offsets = new ArrayList<BoneOffset>();
        int bone_count = aiMesh.mNumBones();
        PointerBuffer bone_buffer = aiMesh.mBones();

        for (int bone_index = 0; bone_index < bone_count; bone_index++)
        {
            var raw_bone = AIBone.create(bone_buffer.get(bone_index));
            var next_bone = load_raw_bone(raw_bone, node_map, mesh_bone_names, bone_weight_map);
            if (next_bone != null)
            {
                mesh_bone_offsets.add(next_bone);
            }
        }

        if (mesh_bone_offsets.isEmpty())
        {
            throw new NullPointerException("No bone for mesh: " + aiMesh.mName().dataString()
                + " ensure mesh has an assigned bone");
        }
        return mesh_bone_offsets;
    }

    private static BoneOffset load_raw_bone(AIBone raw_bone,
                                            Map<String, SceneNode> node_map,
                                            Map<Integer, String[]> mesh_bone_names,
                                            Map<Integer, float[]> bone_weight_map)
    {
        BoneOffset bone_offset;

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

        if (raw_bone.mNumWeights() > 0)
        {
            var bone_node = node_map.get(bone_name);
            int bone_ref_id = GPU.Memory.new_bone_reference(raw_matrix);
            bone_offset = new BoneOffset(bone_ref_id, bone_name, offset, bone_node);
        }
        else
        {
            return null;
        }


        AIVertexWeight.Buffer w_buf = raw_bone.mWeights();
        while (w_buf.remaining() > 0)
        {
            AIVertexWeight weight = w_buf.get();
            var names = mesh_bone_names.computeIfAbsent(weight.mVertexId(), (_x) -> new String[4]);
            var weights = bone_weight_map.computeIfAbsent(weight.mVertexId(), (_x) -> new float[4]);

            int slot = 0;
            while (slot < 4)
            {
                if (weights[slot] == 0f)
                {
                    weights[slot] = weight.mWeight();
                    names[slot] = bone_name;
                    break;
                }
                slot++;
            }
        }

        return bone_offset;
    }

    private static Face[] load_faces(AIMesh aiMesh, int mesh_id)
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
            int[] raw_face = new int[4];
            raw_face[0] = indices.get(0);
            raw_face[1] = indices.get(1);
            raw_face[2] = indices.get(2);
            raw_face[3] = mesh_id;
            int face_id = GPU.Memory.new_mesh_face(raw_face);
            mesh_faces[face_index++] = new Face(face_id, indices.get(0), indices.get(1), indices.get(2));
        }
        return mesh_faces;
    }

    private static Vertex[] load_vertices(AIMesh aiMesh,
                                          Map<Integer, String[]> bone_name_map,
                                          Map<Integer, float[]> bone_weight_map)
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
            String[] names = bone_name_map.get(this_vert);
            float[] weights = bone_weight_map.get(this_vert);

            float sum = 0;
            for (float w : weights)
            {
                sum += w;
            }
            weights[0] /= sum;
            weights[1] /= sum;
            weights[2] /= sum;
            weights[3] /= sum;

            uvChannels.forEach(channel ->
            {
                var next = channel.get(count.get());
                uvData.add(new Vector2f(next.x(), next.y()));
            });

            int[] uv_table =  new int[2];
            uv_table[0] = -1;
            uvData.forEach(uv ->
            {
                var uv_ref = GPU.Memory.new_texture_uv(uv.x, uv.y);
                if (uv_table[0] == -1)
                {
                    uv_table[0] = uv_ref;
                }
                uv_table[1] = uv_ref;
            });


            var vert_ref_id = GPU.Memory.new_vertex_reference(aiVertex.x(), aiVertex.y(), weights, uv_table);

            mesh_vertices[this_vert] = new Vertex(vert_ref_id, aiVertex.x(), aiVertex.y(), uvData, names, weights);
            count.getAndIncrement();
        }
        return mesh_vertices;
    }

    private static void load_mesh(int mesh_index,
                                  String model_name,
                                  AIMesh raw_mesh,
                                  Mesh[] meshes,
                                  Map<String, SceneNode> node_map)
    {
        var mesh_name = raw_mesh.mName().dataString();
        var mesh_node = node_map.get(mesh_name);
        if (mesh_node == null)
        {
            throw new NullPointerException("No scene node for mesh: " + mesh_name
                + " ensure node and geometry names match in blender");
        }

        int next_mesh = GPU.Memory.next_mesh();
        var bone_name_map = new HashMap<Integer, String[]>();
        var bone_weight_map = new HashMap<Integer, float[]>();
        var mesh_bones = load_mesh_bones(raw_mesh, node_map, bone_name_map, bone_weight_map);
        var mesh_vertices = load_vertices(raw_mesh, bone_name_map, bone_weight_map);
        var mesh_faces = load_faces(raw_mesh, next_mesh);
        var hull_table = PhysicsObjects.calculate_convex_hull_table(mesh_vertices);
        int[] table = new int[4];
        //
        table[0] = mesh_vertices[0].vert_ref_id();
        table[1] = mesh_vertices[mesh_vertices.length - 1].vert_ref_id();
        table[2] = mesh_faces[0].index();
        table[3] = mesh_faces[mesh_faces.length - 1].index();
        var mesh_id = GPU.Memory.new_mesh_reference(table);

        assert mesh_id == next_mesh : "Mesh alignment error";

        var new_mesh = new Mesh(mesh_id, mesh_vertices, mesh_faces, mesh_bones, mesh_node, hull_table);

        //System.out.println("Debug mat index:" + raw_mesh.mMaterialIndex() + " for: " + mesh_name);

        Meshes.register_mesh(model_name, mesh_name, new_mesh);
        meshes[mesh_index] = new_mesh;
    }

    private static void load_raw_meshes(int numMeshes,
                                        String model_name,
                                        Mesh[] meshes,
                                        PointerBuffer mesh_buffer,
                                        Map<String, SceneNode> node_map)
    {
        // load raw mesh data for all meshes
        for (int i = 0; i < numMeshes; i++)
        {
            var raw_mesh = AIMesh.create(mesh_buffer.get(i));
            load_mesh(i, model_name, raw_mesh, meshes, node_map);
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
            var match = meshes[mi].bone_offsets().stream()
                .anyMatch(b->b.sceneNode().parent.parent.name.equalsIgnoreCase("RootNode"));

            if (match)
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



    public static Model get_model_by_index(int index)
    {
        return loaded_models.get(index);
    }


    public static void init()
    {
        loaded_models.put(CIRCLE_PARTICLE, Model.fromBasicMesh(Meshes.get_mesh_by_index(Meshes.CIRCLE_MESH)));
        loaded_models.put(TRIANGLE_PARTICLE, Model.fromBasicMesh(Meshes.get_mesh_by_index(Meshes.TRIANGLE_MESH)));
        loaded_models.put(SQUARE_PARTICLE, Model.fromBasicMesh(Meshes.get_mesh_by_index(Meshes.BOX_MESH)));
        loaded_models.put(POLYGON1_MODEL, Model.fromBasicMesh(Meshes.get_mesh_by_index(Meshes.POLYGON1_MESH)));
        TEST_MODEL_INDEX = load_model("/models/test_humanoid_b.fbx", "Humanoid");
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
                                            Map<String, Matrix4f> transforms,
                                            Matrix4f parent_transform,
                                            Map<String, Integer> bind_name_map,
                                            Map<Integer, BoneBindPose> bind_pose_map,
                                            int parent_index)
    {
        var name = current_node.name;
        var node_transform = current_node.transform;
        var global_transform = parent_transform.mul(node_transform, new Matrix4f());

        var parent = parent_index;

        // if this node is a bone, update the
        if (name.toLowerCase().contains("bone")
            && !name.toLowerCase().contains("_end"))
        {
            var raw_matrix = new float[16];
            raw_matrix[0] = node_transform.m00();
            raw_matrix[1] = node_transform.m01();
            raw_matrix[2] = node_transform.m02();
            raw_matrix[3] = node_transform.m03();
            raw_matrix[4] = node_transform.m10();
            raw_matrix[5] = node_transform.m11();
            raw_matrix[6] = node_transform.m12();
            raw_matrix[7] = node_transform.m13();
            raw_matrix[8] = node_transform.m20();
            raw_matrix[9] = node_transform.m21();
            raw_matrix[10] = node_transform.m22();
            raw_matrix[11] = node_transform.m23();
            raw_matrix[12] = node_transform.m30();
            raw_matrix[13] = node_transform.m31();
            raw_matrix[14] = node_transform.m32();
            raw_matrix[15] = node_transform.m33();

            var p = new BoneBindPose(parent, node_transform, name);
            parent = GPU.Memory.new_bone_bind_pose(parent, raw_matrix);
            bind_name_map.put(name, parent);
            bind_pose_map.put(parent, p);
            transforms.put(name, global_transform);
        }
        for (SceneNode child : current_node.children)
        {
            generate_transforms(child, transforms, global_transform, bind_name_map, bind_pose_map, parent);
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
