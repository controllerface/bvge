package com.controllerface.bvge.game;

import com.controllerface.bvge.cl.GPGPU;
import com.controllerface.bvge.ecs.ECS;
import com.controllerface.bvge.ecs.components.*;
import com.controllerface.bvge.ecs.systems.CameraTracking;
import com.controllerface.bvge.ecs.systems.GameSystem;
import com.controllerface.bvge.ecs.systems.sectors.Sector;
import com.controllerface.bvge.ecs.systems.sectors.WorldLoader;
import com.controllerface.bvge.ecs.systems.sectors.WorldUnloader;
import com.controllerface.bvge.game.state.PlayerInventory;
import com.controllerface.bvge.geometry.MeshRegistry;
import com.controllerface.bvge.geometry.ModelRegistry;
import com.controllerface.bvge.gl.renderers.*;
import com.controllerface.bvge.physics.PhysicsEntityBatch;
import com.controllerface.bvge.physics.PhysicsObjects;
import com.controllerface.bvge.physics.PhysicsSimulation;
import com.controllerface.bvge.physics.UniformGrid;
import com.controllerface.bvge.util.Constants;
import com.github.benmanes.caffeine.cache.Cache;
import com.github.benmanes.caffeine.cache.Caffeine;
import org.lwjgl.PointerBuffer;
import org.lwjgl.stb.STBImageWrite;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.system.MemoryUtil;
import org.lwjgl.util.freetype.FT_Bitmap;
import org.lwjgl.util.freetype.FT_Face;
import org.lwjgl.util.freetype.FT_GlyphSlot;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.time.Duration;
import java.time.temporal.ChronoUnit;
import java.util.*;
import java.util.concurrent.LinkedBlockingDeque;

import static com.controllerface.bvge.geometry.ModelRegistry.*;
import static org.lwjgl.util.freetype.FreeType.*;
import static org.lwjgl.util.harfbuzz.HarfBuzz.*;
import static org.lwjgl.util.harfbuzz.HarfBuzz.hb_blob_destroy;

public class TestGame extends GameMode
{
    private final GameSystem blanking_system;
    private final int GRID_WIDTH = 3840;
    private final int GRID_HEIGHT = 2160;

    private final Cache<Sector, PhysicsEntityBatch> sector_cache;
    private final Queue<PhysicsEntityBatch> spawn_queue;
    private final PlayerInventory player_inventory;

    private enum RenderType
    {
        GAME,       // normal objects
        HULLS,      // physics hulls
        BOUNDS,     // bounding boxes
        POINTS,     // model vertices
        ENTITIES,   // entity roots
        GRID,       // uniform grid
    }

    private static final EnumSet<RenderType> ACTIVE_RENDERERS =
        EnumSet.of(RenderType.GAME
//            ,RenderType.HULLS
//            ,RenderType.POINTS
//            ,RenderType.ENTITIES
//            ,RenderType.BOUNDS
//            ,RenderType.GRID
        );

//    private static final EnumSet<RenderType> ACTIVE_RENDERERS =
//        EnumSet.of(RenderType.HULLS);

    //private final UniformGrid uniformGrid = new UniformGrid(Window.get().width(), Window.get().height());
    private final UniformGrid uniformGrid = new UniformGrid(GRID_WIDTH, GRID_HEIGHT);


    public TestGame(ECS ecs, GameSystem blanking_system)
    {
        super(ecs);

        MeshRegistry.init();
        ModelRegistry.init();

        this.blanking_system = blanking_system;
        this.spawn_queue = new LinkedBlockingDeque<>();
        this.player_inventory = new PlayerInventory();
        this.sector_cache = Caffeine.newBuilder()
            .expireAfterAccess(Duration.of(1, ChronoUnit.HOURS))
            .build();

    }

    private void gen_player(float size, float x, float y)
    {
        var player = ecs.register_entity("player");
        var entity_id = PhysicsObjects.wrap_model(GPGPU.core_memory, PLAYER_MODEL_INDEX, x, y, size, 100.5f, 0.05f, 0, 0, Constants.EntityFlags.CAN_COLLECT.bits);
        var cursor_id = PhysicsObjects.circle_cursor(GPGPU.core_memory, 0, 0, 10, entity_id[1]);

        ecs.attach_component(player, Component.EntityId, new EntityIndex(entity_id[0]));
        ecs.attach_component(player, Component.CursorId, new EntityIndex(cursor_id));
        ecs.attach_component(player, Component.ControlPoints, new ControlPoints());
        ecs.attach_component(player, Component.CameraFocus, new CameraFocus());
        ecs.attach_component(player, Component.LinearForce, new LinearForce(1600));
    }

    private void load_systems()
    {
        ecs.register_system(new WorldLoader(ecs, uniformGrid, sector_cache, spawn_queue));
        ecs.register_system(new PhysicsSimulation(ecs, uniformGrid));
        ecs.register_system(new WorldUnloader(ecs, sector_cache, spawn_queue, player_inventory));
        ecs.register_system(new CameraTracking(ecs, uniformGrid));

        ecs.register_system(blanking_system);

        ecs.register_system(new BackgroundRenderer(ecs));
        ecs.register_system(new MouseRenderer(ecs));

        if (ACTIVE_RENDERERS.contains(RenderType.GAME))
        {
            ecs.register_system(new ModelRenderer(ecs, uniformGrid, PLAYER_MODEL_INDEX, BASE_BLOCK_INDEX, BASE_SPIKE_INDEX, R_SHARD_INDEX, L_SHARD_INDEX));
            ecs.register_system(new LiquidRenderer(ecs, uniformGrid));
        }

        //ecs.register_system(new HUDRenderer(ecs));


        // debug renderers

        if (ACTIVE_RENDERERS.contains(RenderType.HULLS))
        {
            ecs.register_system(new EdgeRenderer(ecs));
            ecs.register_system(new CircleRenderer(ecs));
        }

        if (ACTIVE_RENDERERS.contains(RenderType.BOUNDS))
        {
            ecs.register_system(new BoundingBoxRenderer(ecs));
        }

        if (ACTIVE_RENDERERS.contains(RenderType.POINTS))
        {
            ecs.register_system(new PointRenderer(ecs));
        }

        if (ACTIVE_RENDERERS.contains(RenderType.GRID))
        {
            ecs.register_system(new UniformGridRenderer(ecs, uniformGrid));
        }

        if (ACTIVE_RENDERERS.contains(RenderType.ENTITIES))
        {
            ecs.register_system(new EntityPositionRenderer(ecs));
        }
    }

    @Override
    public void load()
    {
        gen_player(1.2f, -250, 0);
        load_systems();
    }


    // WORKING AREA BELOW

    private static FT_Face loadFontFace(long ftLibrary, String fontPath)
    {
        try (MemoryStack stack = MemoryStack.stackPush())
        {
            PointerBuffer pp = stack.mallocPointer(1);
            int error = FT_New_Face(ftLibrary, fontPath, 0, pp);
            if (error != 0)
            {
                System.err.println("FT_New_Face error: " + error);
                return null;
            }
            return FT_Face.create(pp.get(0));
        }
    }

    private static long initFreeType() {
        try (MemoryStack stack = MemoryStack.stackPush()) {
            PointerBuffer pp = stack.mallocPointer(1);
            if (FT_Init_FreeType(pp) != 0) {
                throw new RuntimeException("Could not initialize FreeType library");
            }
            return pp.get(0);
        }
    }

    private final String font_file = "C:\\Users\\Stephen\\IdeaProjects\\bvge\\src\\main\\resources\\font\\Inconsolata-Light.ttf";

    private static int gcount = 0;

    private static void drawGlyph(FT_Face ftFace, int glyphID, int x, int y) {
        if (FT_Load_Glyph(ftFace, glyphID, FT_LOAD_DEFAULT) != 0) {
            throw new RuntimeException("Could not load glyph");
        }

        FT_GlyphSlot glyph = ftFace.glyph();
        if (FT_Render_Glyph(glyph, FT_RENDER_MODE_NORMAL) != 0) {
            throw new RuntimeException("Could not render glyph");
        }

        FT_Bitmap bitmap = glyph.bitmap();
        int width = bitmap.width();
        int height = bitmap.rows();
        ByteBuffer buffer = bitmap.buffer(width * height);

        ByteBuffer image = ByteBuffer.allocateDirect(width * height).order(ByteOrder.nativeOrder());

        for (int row = 0; row < height; row++) {
            for (int col = 0; col < width; col++) {
                var xxx = buffer.get((row * bitmap.pitch()) + col);
                image.put((row * width) + col, xxx);
            }
        }

        STBImageWrite.stbi_write_png((gcount++) + "test.png", width, height, 1, image, width);
    }


    @Override
    public void start()
    {
        long ftLibrary = initFreeType();
        FT_Face ftFace = loadFontFace(ftLibrary, font_file);
        FT_Set_Char_Size(ftFace, 64, 0, 1920, 0);
        var text = "hello there";

        var buffer = hb_buffer_create();
        hb_buffer_add_utf8(buffer, text, 0, text.length());
        hb_buffer_set_direction(buffer, HB_DIRECTION_LTR);
        hb_buffer_set_script(buffer, HB_SCRIPT_LATIN);
        hb_buffer_set_language(buffer, hb_language_from_string("en"));

        var font = hb_ft_font_create_referenced(ftFace.address());
        var face = hb_ft_face_create_referenced(ftFace.address());

        hb_shape(font, buffer, null);
        var glyph_info = hb_buffer_get_glyph_infos(buffer);
        var glyph_pos = hb_buffer_get_glyph_positions(buffer);
        int glyphCount = hb_buffer_get_length(buffer);
        int cursor_x = 0;
        int cursor_y = 0;
        for (int i = 0; i < glyphCount; i++)
        {
            var glyphid = glyph_info.get(i).codepoint();
            var x_offset = glyph_pos.get(i).x_offset();
            var y_offset = glyph_pos.get(i).y_offset();
            var x_advance = glyph_pos.get(i).x_advance();
            var y_advance = glyph_pos.get(i).y_advance();
            // render here?
            drawGlyph(ftFace, glyphid, cursor_x + x_offset, cursor_y + y_offset);
            cursor_x += x_advance;
            cursor_y += y_advance;
        }
        hb_buffer_destroy(buffer);
        hb_font_destroy(font);
        hb_face_destroy(face);
    }

    @Override
    public void update(float dt)
    {

    }
}
