package com.controllerface.bvge.editor;

import com.controllerface.bvge.editor.http.Request;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.net.Socket;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

public class StaticAsset
{
    public final String location;
    public final String mimetype;

    private static final byte[] EOL_BYTES = new byte[]{'\r', '\n'};
    private static final byte[] LINE_200_OK = ("HTTP/1.1 " + 200 + " OK").getBytes(StandardCharsets.UTF_8);
    private static final byte[] CONTENT_TYPE = "Content-Type: ".getBytes(StandardCharsets.UTF_8);
    private static final byte[] CONTENT_LENGTH = "Content-Length: ".getBytes(StandardCharsets.UTF_8);
    private static final byte[] CHARSET_UTF_8 = "; charset=utf-8".getBytes(StandardCharsets.UTF_8);

    public static Map<String, String> mimeTypes = Map.of
        (
            "css",  "text/css",
            "html", "text/html",
            "ttf",  "font/ttf",
            "svg",  "image/svg+xml",
            "js",   "text/javascript"
        );

    public static final Map<String, StaticAsset> static_assets = make_static_assets();

    private StaticAsset(String location, String mimetype)
    {
        this.location = location;
        this.mimetype = mimetype;
    }

    private static StaticAsset make(String location, String mime_type)
    {
        return new StaticAsset(location, mime_type);
    }

    private static StaticAsset make(String location)
    {
        return make(location, determineMimeType(location));
    }

    private static String determineMimeType(String location)
    {
        var dot = location.lastIndexOf(".");
        if (dot == -1)
        {
            throw new RuntimeException("Filename must have extension");
        }
        var extension = location.substring(dot + 1);
        var mimeType = mimeTypes.get(extension);
        if (mimeType == null)
        {
            throw new RuntimeException("Unknown extension: " + extension);
        }
        return mimeType;
    }

    private static byte[] int_bytes(int number)
    {
        if (number < 0x80)
        {
            return new byte[]{(byte) number};
        }
        else if (number < 0x800)
        {
            byte byte_0 = (byte) (0xC0 | (number >> 6));
            byte byte_1 = (byte) (0x80 | (number & 0x3F));
            return new byte[]{ byte_0, byte_1 };
        }
        else if (number < 0x10000)
        {
            byte byte_0 = (byte) (0xE0 | (number >> 12));
            byte byte_1 = (byte) (0x80 | ((number >> 6) & 0x3F));
            byte byte_2 = (byte) (0x80 | (number & 0x3F));
            return new byte[]{ byte_0, byte_1, byte_2 };
        }
        else
        {
            byte byte_0 = (byte) (0xF0 | (number >> 18));
            byte byte_1 = (byte) (0x80 | ((number >> 12) & 0x3F));
            byte byte_2 = (byte) (0x80 | ((number >> 6) & 0x3F));
            byte byte_3 = (byte) (0x80 | (number & 0x3F));
            return new byte[]{ byte_0, byte_1, byte_2, byte_3 };
        }
    }

    private static void writeResourceResponse(Socket client_connection, StaticAsset static_asset)
    {
        try (client_connection;
             var buffer = new ByteArrayOutputStream();
             var data = StaticAsset.class.getResourceAsStream(static_asset.location);
             var response_stream = client_connection.getOutputStream())
        {
            var data_bytes = Objects.requireNonNull(data).readAllBytes();
            buffer.writeBytes(LINE_200_OK);
            buffer.writeBytes(EOL_BYTES);
            buffer.writeBytes(CONTENT_TYPE);
            buffer.writeBytes(static_asset.mimetype.getBytes(StandardCharsets.UTF_8));
            buffer.writeBytes(CHARSET_UTF_8);
            buffer.writeBytes(EOL_BYTES);
            buffer.writeBytes(CONTENT_LENGTH);
            buffer.writeBytes(int_bytes(data_bytes.length));
            buffer.writeBytes(EOL_BYTES);
            buffer.writeBytes(EOL_BYTES);
            buffer.writeBytes(data_bytes);
            response_stream.write(buffer.toByteArray());
            response_stream.flush();
        }
        catch (IOException | NullPointerException e)
        {
            System.err.println("Error writing response stream");
        }
    }

    private static Map<String, StaticAsset> make_static_assets()
    {
        var assetMap = new HashMap<String, StaticAsset>();

        assetMap.put("/",           StaticAsset.make("/ui/html/editor.html"));
        assetMap.put("/stats.html", StaticAsset.make("/ui/html/stats.html"));
        assetMap.put("/editor.js",  StaticAsset.make("/ui/js/editor.js"));
        assetMap.put("/editor.css", StaticAsset.make("/ui/css/editor.css"));

        return assetMap;
    }

    public void writeTo(Socket client_connection)
    {
        writeResourceResponse(client_connection, this);
    }

    public static boolean is_asset(String uri_key)
    {
        return static_assets.containsKey(uri_key);
    }

    public static void serve_asset(Request request, Socket client_connection, EditorServer server)
    {
        // todo: inform server of failure to serve, if any
        static_assets.get(request.uri()).writeTo(client_connection);
    }
}
