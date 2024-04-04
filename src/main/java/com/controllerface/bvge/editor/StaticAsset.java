package com.controllerface.bvge.editor;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.net.Socket;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.Map;

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

    private StaticAsset(String location, String mimetype)
    {
        this.location = location;
        this.mimetype = mimetype;
    }

    private static StaticAsset make(String location, String mimetype)
    {
        return new StaticAsset(location, mimetype);
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

    private static byte[] toUTF8Bytes(int number)
    {
        if (number < 0x80)
        {
            return new byte[]{(byte) number};
        }
        else if (number < 0x800)
        {
            return new byte[]{(byte) (0xC0 | (number >> 6)), (byte) (0x80 | (number & 0x3F))};
        }
        else if (number < 0x10000)
        {
            return new byte[]{(byte) (0xE0 | (number >> 12)), (byte) (0x80 | ((number >> 6) & 0x3F)), (byte) (0x80 | (number & 0x3F))};
        }
        else
        {
            return new byte[]{(byte) (0xF0 | (number >> 18)), (byte) (0x80 | ((number >> 12) & 0x3F)), (byte) (0x80 | ((number >> 6) & 0x3F)), (byte) (0x80 | (number & 0x3F))};
        }
    }

    private static void writeResourceResponse(Socket client_connection, StaticAsset staticAsset)
    {
        try (var buffer = new ByteArrayOutputStream();
             var data = StaticAsset.class.getResourceAsStream(staticAsset.location);
             var response_stream = client_connection.getOutputStream())
        {
            var data_bytes = data.readAllBytes();
            buffer.writeBytes(LINE_200_OK);
            buffer.writeBytes(EOL_BYTES);
            buffer.writeBytes(CONTENT_TYPE);
            buffer.writeBytes(staticAsset.mimetype.getBytes(StandardCharsets.UTF_8));
            buffer.writeBytes(CHARSET_UTF_8);
            buffer.writeBytes(EOL_BYTES);
            buffer.writeBytes(CONTENT_LENGTH);
            buffer.writeBytes(toUTF8Bytes(data_bytes.length));
            buffer.writeBytes(EOL_BYTES);
            buffer.writeBytes(EOL_BYTES);
            buffer.writeBytes(data_bytes);
            response_stream.write(buffer.toByteArray());
            response_stream.flush();
        }
        catch (IOException e)
        {
            System.err.println("Error writing response stream");
        }
    }

    private static Map<String, StaticAsset> makeStaticAssets()
    {
        var assetMap = new HashMap<String, StaticAsset>();

        assetMap.put("/",           StaticAsset.make("/ui/html/editor.html"));
        assetMap.put("/editor.js",  StaticAsset.make("/ui/js/editor.js"));

        return assetMap;
    }

    public void writeTo(Socket client_connection)
    {
        writeResourceResponse(client_connection, this);
    }

    public static final Map<String, StaticAsset> staticAssets = makeStaticAssets();
}
