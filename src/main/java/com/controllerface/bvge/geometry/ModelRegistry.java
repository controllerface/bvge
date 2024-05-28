package com.controllerface.bvge.geometry;

import com.controllerface.bvge.animation.BoneBindPose;
import com.controllerface.bvge.animation.BoneChannel;
import com.controllerface.bvge.animation.BoneOffset;
import com.controllerface.bvge.cl.GPGPU;
import com.controllerface.bvge.game.AnimationState;
import com.controllerface.bvge.gl.Texture;
import com.controllerface.bvge.physics.PhysicsObjects;
import com.controllerface.bvge.util.Assets;
import com.controllerface.bvge.util.MathEX;
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
import java.util.concurrent.atomic.AtomicReference;

import static org.lwjgl.assimp.Assimp.*;

public class ModelRegistry
{
    private static final AtomicInteger next_model_index = new AtomicInteger(0);

    public static final int BASE_BLOCK_INDEX = next_model_index.getAndIncrement();
    public static final int CIRCLE_PARTICLE = next_model_index.getAndIncrement();
    public static final int BASE_SHARD_INDEX = next_model_index.getAndIncrement();
    public static final int BASE_SPIKE_INDEX = next_model_index.getAndIncrement();

    public static int CURSOR = next_model_index.getAndIncrement();

    public static int TEST_MODEL_INDEX_2 = -1;
    public static int PLAYER_MODEL_INDEX = -1;

    private static final Map<Integer, Model> loaded_models = new HashMap<>();

    private static final BlockAtlas BLOCK_ATLAS = new BlockAtlas();

    private static final int DEFAULT_MODEL_LOAD_FLAGS =
        aiProcess_JoinIdenticalVertices | aiProcess_Triangulate | aiProcess_FixInfacingNormals;

    private static AIScene loadModelResource(String name) throws IOException
    {
        ByteBuffer model_buffer = null;
        try (var model_stream = ModelRegistry.class.getResourceAsStream(name))
        {
            assert model_stream != null : "Model data is null: " + name;
            byte[] model_data = model_stream.readAllBytes();
            model_buffer = MemoryUtil.memCalloc(model_data.length).put(model_data).flip();
            return aiImportFileFromMemory(model_buffer, DEFAULT_MODEL_LOAD_FLAGS, "");
        }
        finally
        {
            MemoryUtil.memFree(model_buffer);
        }
    }

    private static int load_model(String model_path, String model_name)
    {
        return load_model(model_path, model_name, Collections.emptyList());
    }

    private static int load_model(String model_path, String model_name, List<List<Vector2f>> baked_uvs)
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

        // maps each bone to a calculated bind pose transformation matrix
        var bone_transforms = new HashMap<String, Matrix4f>();

        // this map must be linked to preserve insert order
        var bind_pose_map = new LinkedHashMap<Integer, BoneBindPose>();

        var bind_name_map = new HashMap<String, Integer>();
        var model_transform = new AtomicReference<Matrix4f>();
        var entity_transform = new AtomicReference<Matrix4f>();

        var animation_map = new EnumMap<AnimationState, Integer>(AnimationState.class);

        // read scene data
        try (AIScene ai_scene = loadModelResource(model_path))
        {
            mesh_count = ai_scene.mNumMeshes();
            root_node = ai_scene.mRootNode();
            scene_node = process_node_hierarchy(root_node, null, node_map);
            meshes = new Mesh[mesh_count];
            mesh_buffer = ai_scene.mMeshes();

            load_textures(ai_scene, textures);
            load_materials(ai_scene);

            // generate the bind pose transforms, setting the initial state of the armature
            generate_transforms(scene_node, bone_transforms, new Matrix4f(), bind_name_map, bind_pose_map, model_transform, entity_transform,-1);

            load_animations(ai_scene, bind_name_map, animation_map);
            load_raw_meshes(mesh_count, model_name, meshes, mesh_buffer, node_map, baked_uvs);

            // we need to calculate the root node for the body, which is the mesh that is tracking the root bone.
            // the root bone is determined by checking if the current bone's parent is a direct descendant of the scene
            int root_index = find_root_index(meshes);

            // register the model
            var next_model_id = next_model_index.getAndIncrement();
            int root_transform_index = -1;
            if (model_transform.get() != null)
            {
                root_transform_index = GPGPU.core_memory.new_model_transform(MathEX.raw_matrix(model_transform.get()));
            }
            var model = new Model(meshes,
                entity_transform.get(),
                bone_transforms,
                bind_name_map,
                bind_pose_map,
                textures,
                root_index,
                root_transform_index);
            loaded_models.put(next_model_id, model);
            return next_model_id;
        }
        catch (NullPointerException | IOException e)
        {
            throw new RuntimeException("Unable to load model data", e);
        }
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
        var raw_matrix = MathEX.raw_matrix(mOffset);
        offset.set(raw_matrix);

        if (raw_bone.mNumWeights() > 0)
        {
            var bone_node = node_map.get(bone_name);
            int bone_ref_id = GPGPU.core_memory.new_bone_reference(raw_matrix);
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
            int face_id = GPGPU.core_memory.new_mesh_face(raw_face);
            mesh_faces[face_index++] = new Face(face_id, indices.get(0), indices.get(1), indices.get(2));
        }
        return mesh_faces;
    }

    // todo: add mechanism for passing in pre-computed color sets
    private static Vertex[] load_vertices(AIMesh aiMesh,
                                          Map<Integer, String[]> bone_name_map,
                                          Map<Integer, float[]> bone_weight_map,
                                          List<List<Vector2f>> baked_uvs)
    {
        int vert_index = 0;
        var mesh_vertices = new Vertex[aiMesh.mNumVertices()];
        var buffer = aiMesh.mVertices();

        List<List<Vector2f>> uvChannels = new ArrayList<>();
        if (baked_uvs.isEmpty())
        {
            for (int i = 0; i < AI_MAX_NUMBER_OF_TEXTURECOORDS; i++)
            {
                var uvBuffer = aiMesh.mTextureCoords(i);
                if (uvBuffer != null)
                {
                    List<Vector2f> currentChannel = new ArrayList<>();
                    while (uvBuffer.remaining() > 0)
                    {
                        var aiVector = uvBuffer.get();
                        currentChannel.add(new Vector2f(aiVector.x(), aiVector.y()));
                    }
                    uvChannels.add(currentChannel);
                }
            }
        }
        else
        {
            uvChannels.addAll(baked_uvs);
        }

        // todo: define color sets for vertex objects and set them using the backed in sets, or passed in ones
        int color_sets = 0;
        for (int i = 0; i < AI_MAX_NUMBER_OF_COLOR_SETS; i++)
        {
            var color_set = aiMesh.mColors(i);

            if (color_set != null)
            {
                color_sets++;
                int colors_in_set = 0;
                while (color_set.remaining() > 0)
                {
                    var n = color_set.get();
                    //System.out.println(STR."color: r:\{n.r()} g:\{n.g()} b:\{n.b()} a:\{n.a()}");
                    colors_in_set++;
                }
                //System.out.println(STR."colors in set: \{+colors_in_set}");

                var color_s = AIColor4D.create(color_set.address());
                //System.out.println(color_s);
            }
        }
        //System.out.println(STR."color sets: \{aiMesh.mName().dataString()} - \{+color_sets}");

        AtomicInteger current_vertex_index = new AtomicInteger();
        while (buffer.remaining() > 0)
        {
            int this_index = current_vertex_index.getAndIncrement();
            int this_vert = vert_index++;
            var raw_vertex = buffer.get();
            System.out.println(raw_vertex.getClass().getCanonicalName());
            List<Vector2f> vertex_uvs = new ArrayList<>();
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

            uvChannels.forEach(channel -> vertex_uvs.add(channel.get(this_index)));

            int[] uv_table =  new int[2];
            uv_table[0] = -1;
            vertex_uvs.forEach(uv ->
            {
                var uv_ref = GPGPU.core_memory.new_texture_uv(uv.x, uv.y);
                if (uv_table[0] == -1)
                {
                    uv_table[0] = uv_ref;
                }
                uv_table[1] = uv_ref;
            });

            var vert_ref_id = GPGPU.core_memory.new_vertex_reference(raw_vertex.x(), raw_vertex.y(), weights, uv_table);
            mesh_vertices[this_vert] = new Vertex(vert_ref_id, raw_vertex.x(), raw_vertex.y(), vertex_uvs, names, weights);
        }
        return mesh_vertices;
    }

    private static void load_mesh(int mesh_index,
                                  String model_name,
                                  AIMesh raw_mesh,
                                  Mesh[] meshes,
                                  Map<String, SceneNode> node_map,
                                  List<List<Vector2f>> baked_uvs)
    {
        var mesh_name = raw_mesh.mName().dataString();
        var mesh_node = node_map.get(mesh_name);
        if (mesh_node == null)
        {
            throw new NullPointerException("No scene node for mesh: " + mesh_name
                + " ensure node and geometry names match in blender");
        }

        int next_mesh = GPGPU.core_memory.next_mesh();
        var bone_name_map = new HashMap<Integer, String[]>();
        var bone_weight_map = new HashMap<Integer, float[]>();
        var mesh_bones = load_mesh_bones(raw_mesh, node_map, bone_name_map, bone_weight_map);
        var mesh_vertices = load_vertices(raw_mesh, bone_name_map, bone_weight_map, baked_uvs);
        var mesh_faces = load_faces(raw_mesh, next_mesh);
        var hull_table = PhysicsObjects.calculate_convex_hull_table(mesh_vertices);
        int[] vertex_table = new int[2];
        int[] face_table = new int[2];
        vertex_table[0] = mesh_vertices[0].index();
        vertex_table[1] = mesh_vertices[mesh_vertices.length - 1].index();
        face_table[0] = mesh_faces[0].index();
        face_table[1] = mesh_faces[mesh_faces.length - 1].index();
        var mesh_id = GPGPU.core_memory.new_mesh_reference(vertex_table, face_table);

        assert mesh_id == next_mesh : "Mesh alignment error";

        // todo: add material data to the mesh class
        var new_mesh = new Mesh(mesh_name, mesh_id, mesh_vertices, mesh_faces, mesh_bones, mesh_node, hull_table);

        MeshRegistry.register_mesh(model_name, mesh_name, new_mesh);
        meshes[mesh_index] = new_mesh;
    }

    private static void load_raw_meshes(int numMeshes,
                                        String model_name,
                                        Mesh[] meshes,
                                        PointerBuffer mesh_buffer,
                                        Map<String, SceneNode> node_map,
                                        List<List<Vector2f>> baked_uvs)
    {
        // load raw mesh data for all meshes
        for (int i = 0; i < numMeshes; i++)
        {
            var raw_mesh = AIMesh.create(mesh_buffer.get(i));
            load_mesh(i, model_name, raw_mesh, meshes, node_map, baked_uvs);
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
    //  and each mesh should have its material set during construction.
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
                    System.out.println("Mat type=" + raw_prop.mType()
                            + " name=" + prop_name
                            + " r=" + r
                            + " g=" + g
                            + " b=" + b);
                }
                else if (prop_name.startsWith("$mat."))
                {
                    float v = raw_prop.mData().asFloatBuffer().get(0);
                    System.out.println("Mat type=" + raw_prop.mType()
                            + " name=" + prop_name
                            + " v=" + v);
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
                            System.out.println("Mat type=" + raw_prop.mType()
                                    + " name=" + prop_name
                                    + " float=" + Arrays.toString(float_out));
                            break;

                        case 3:
                            var string_buffer = raw_prop.mData();
                            int s_count = raw_prop.mDataLength();
                            byte[] bytes_out = new byte[s_count];
                            string_buffer.get(bytes_out);
                            var string = new String(bytes_out, StandardCharsets.UTF_8);
                            System.out.println("Mat type=" + raw_prop.mType()
                                    + " name=" + prop_name
                                    + " string=" + string);
                            break;

                        case 4:
                            var int_buffer = raw_prop.mData().asIntBuffer();
                            int i_count = raw_prop.mDataLength() / 4;
                            int[] int_out = new int[i_count];
                            int_buffer.get(int_out);
                            System.out.println("Mat type=" + raw_prop.mType()
                                    + " name=" + prop_name
                                    + " int=" + Arrays.toString(int_out));
                            break;

                        default:
                            System.out.println("Debug mat prop: type=" + raw_prop.mType()
                                    + " len=" + raw_prop.mDataLength()
                                    + " key=" + raw_prop.mKey().dataString());
                            break;
                    }
                }
            }
        }
    }

    private static void load_animations(AIScene aiScene,
                                        Map<String, Integer> bind_name_map,
                                        Map<AnimationState, Integer> animation_map)
    {
        var raw_animation_count = aiScene.mNumAnimations();
        if (raw_animation_count < 1) return;

        var anim_map = new HashMap<Integer, BoneChannel[]>();

        var anim_buffer = aiScene.mAnimations();
        int current_index = 0;

        var anim_buf = new EnumMap<AnimationState, AIAnimation>(AnimationState.class);

        for (int animation_index = 0; animation_index < raw_animation_count; animation_index++)
        {
            var raw_animation = AIAnimation.create(anim_buffer.get(animation_index));
            var animation_name = raw_animation.mName().dataString();
            var animation_state = AnimationState.fuzzy_match(animation_name);
            anim_buf.put(animation_state, raw_animation);
        }

        int animation_count = anim_buf.size();

        int animation_index = 0;
        for (Map.Entry<AnimationState, AIAnimation> entry : anim_buf.entrySet())
        {
            var animation_state = entry.getKey();
            var raw_animation = entry.getValue();

            // store the timings so bone channels can use them
            int anim_timing_id = GPGPU.core_memory.new_animation_timings((float)raw_animation.mDuration(), (float)raw_animation.mTicksPerSecond());
            animation_map.put(animation_state, current_index++);

            System.out.println("anim" + raw_animation.mName().dataString()
                    + " state:" + animation_state
                    + " id:" + anim_timing_id
                    + " duration:" + raw_animation.mDuration());


            int channel_count = raw_animation.mNumChannels();
            var channel_buffer = raw_animation.mChannels();
            for (int channel_index = 0; channel_index < channel_count; channel_index++)
            {
                var raw_channel = AINodeAnim.create(channel_buffer.get(channel_index));
                var bone_name = raw_channel.mNodeName().dataString();

                if (bind_name_map.get(bone_name) == null) continue;

                int bind_pose_id = bind_name_map.get(bone_name);

                int pos_key_count = raw_channel.mNumPositionKeys();
                int rot_key_count = raw_channel.mNumRotationKeys();
                int scl_key_count = raw_channel.mNumScalingKeys();

                var pos_buffer = raw_channel.mPositionKeys();
                var rot_buffer = raw_channel.mRotationKeys();
                var scl_buffer = raw_channel.mScalingKeys();

                int p_start = -1;
                int p_end = -1;

                int r_start = -1;
                int r_end = -1;

                int s_start = -1;
                int s_end = -1;

                for (int current_pos_key = 0; current_pos_key < pos_key_count; current_pos_key++)
                {
                    var raw_pos_key = pos_buffer.get(current_pos_key);
                    var pos_vector = raw_pos_key.mValue();
                    float[] frame_data = new float[]{ pos_vector.x(), pos_vector.y(), pos_vector.z(), 1.0f };
                    int next_pos_key = GPGPU.core_memory.new_keyframe(frame_data, (float)raw_pos_key.mTime());
                    if (p_start == -1) p_start = next_pos_key;
                    p_end = next_pos_key;
                }

                for (int current_rot_key = 0; current_rot_key < rot_key_count; current_rot_key++)
                {
                    var raw_rot_key = rot_buffer.get(current_rot_key);
                    var rot_quaternion = raw_rot_key.mValue();
                    float[] frame_data = new float[]{ rot_quaternion.x(), rot_quaternion.y(), rot_quaternion.z(), rot_quaternion.w() };
                    int next_rot_key = GPGPU.core_memory.new_keyframe(frame_data, (float)raw_rot_key.mTime());
                    if (r_start == -1) r_start = next_rot_key;
                    r_end = next_rot_key;
                }

                for (int current_scl_key = 0; current_scl_key < scl_key_count; current_scl_key++)
                {
                    var raw_scl_key = scl_buffer.get(current_scl_key);
                    var scale_vector = raw_scl_key.mValue();
                    float[] frame_data = new float[]{ scale_vector.x(), scale_vector.y(), scale_vector.z(), 1.0f };
                    int next_scl_key = GPGPU.core_memory.new_keyframe(frame_data, (float)raw_scl_key.mTime());
                    if (s_start == -1) s_start = next_scl_key;
                    s_end = next_scl_key;
                }

                var new_channel = new BoneChannel(anim_timing_id, p_start, p_end, r_start, r_end, s_start, s_end);
                var channels = anim_map.computeIfAbsent(bind_pose_id, (_k) -> new BoneChannel[animation_count]);
                channels[animation_index] = new_channel;
            }
            animation_index++;
        }

        anim_map.forEach((bind_pose_id, bone_channels) ->
        {
            int c_start = -1;
            int c_end = -1;
            for (BoneChannel channel : bone_channels)
            {
                if (channel == null) continue;
                try
                {
                    int[] pos_table = new int[]{ channel.pos_start(), channel.pos_end() };
                    int[] rot_table = new int[]{ channel.rot_start(), channel.rot_end() };
                    int[] scl_table = new int[]{ channel.scl_start(), channel.scl_end() };

                    int next_channel = GPGPU.core_memory.new_bone_channel(channel.anim_timing_id(), pos_table, rot_table, scl_table);
                    if (c_start == -1) c_start = next_channel;
                    c_end = next_channel;
                }
                catch (Exception e)
                {
                    e.printStackTrace();
                    throw new RuntimeException("Could not load animation data");
                }
            }
            GPGPU.core_memory.set_bone_channel_table(bind_pose_id, new int[]{ c_start, c_end });
        });
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

    private static SceneNode process_node_hierarchy(AINode aiNode, SceneNode parentNode, Map<String, SceneNode> nodeMap)
    {
        var nodeName = aiNode.mName().dataString();
        var mTransform = aiNode.mTransformation();
        var transform = new Matrix4f();
        transform.set(mTransform.a1(), mTransform.b1(), mTransform.c1(), mTransform.d1(),
            mTransform.a2(), mTransform.b2(), mTransform.c2(), mTransform.d2(),
            mTransform.a3(), mTransform.b3(), mTransform.c3(), mTransform.d3(),
            mTransform.a4(), mTransform.b4(), mTransform.c4(), mTransform.d4());

        var x = aiNode.mMetadata();
        if (x != null)
        {
            int p = x.mNumProperties();
            var ks = x.mKeys();
            var vs = x.mValues();

            for (int i = 0; i < p; i++)
            {
                var key = ks.get();
                var val = vs.get();
                var key_string = key.dataString();
                System.out.println("n:" + nodeName + " k:" + key_string + " v:" + val.mType());
                if (key_string.equalsIgnoreCase("pin"))
                {
                    int k_int = val.mData(4).asIntBuffer().get();
                    System.out.println("int: " + k_int);
                }
            }

            System.out.println(p);
        }

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

    private static void generate_transforms(SceneNode current_node,
                                            Map<String, Matrix4f> transforms,
                                            Matrix4f parent_transform,
                                            Map<String, Integer> bind_name_map,
                                            Map<Integer, BoneBindPose> bind_pose_map,
                                            AtomicReference<Matrix4f> model_matrix,
                                            AtomicReference<Matrix4f> entity_matrix,
                                            int parent_index)
    {
        var name = current_node.name;
        boolean is_bone = name.toLowerCase().contains("bone")
            && !name.toLowerCase().contains("_end");
        boolean is_armature = name.equalsIgnoreCase("Armature");
        var node_transform = current_node.transform;
        var global_transform = parent_transform.mul(node_transform, new Matrix4f());

        var parent = parent_index;

        if (is_armature)
        {
            boolean model_ok = model_matrix.compareAndSet(null, node_transform);
            assert model_ok : "model transform already set";
            boolean armature_ok = entity_matrix.compareAndSet(null, node_transform);
            assert armature_ok : "armature transform already set";
        }

        if (is_bone)
        {
            var raw_matrix = MathEX.raw_matrix(node_transform);
            var bind_pose = new BoneBindPose(parent, node_transform, name);
            int bind_pose_id = GPGPU.core_memory.new_bone_bind_pose(raw_matrix);
            bind_name_map.put(name, bind_pose_id);
            bind_pose_map.put(bind_pose_id, bind_pose);
            transforms.put(name, global_transform);
            parent = bind_pose_id;
        }
        for (SceneNode child : current_node.children)
        {
            generate_transforms(child, transforms, global_transform, bind_name_map, bind_pose_map, model_matrix, entity_matrix, parent);
        }
    }

    public static Model get_model_by_index(int index)
    {
        return loaded_models.get(index);
    }

    public static void init()
    {
        var texture = Assets.load_texture("/img/blocks.png");
        loaded_models.put(CURSOR, Model.fromBasicMesh(MeshRegistry.get_mesh_by_index(MeshRegistry.CIRCLE_MESH)));
        loaded_models.put(CIRCLE_PARTICLE, Model.fromBasicMesh(MeshRegistry.get_mesh_by_index(MeshRegistry.CIRCLE_MESH)));

        loaded_models.put(BASE_BLOCK_INDEX, Model.fromBasicMesh(MeshRegistry.get_mesh_by_index(MeshRegistry.BLOCK_MESH), texture));
        loaded_models.put(BASE_SHARD_INDEX, Model.fromBasicMesh(MeshRegistry.get_mesh_by_index(MeshRegistry.SHARD_MESH), texture));
        loaded_models.put(BASE_SPIKE_INDEX, Model.fromBasicMesh(MeshRegistry.get_mesh_by_index(MeshRegistry.SPIKE_MESH), texture));

        PLAYER_MODEL_INDEX = load_model("/models/humanoid_redux.fbx", "Humanoid2");
        TEST_MODEL_INDEX_2 = load_model("/models/test_humanoid_2.fbx", "Humanoid");
    }
}
