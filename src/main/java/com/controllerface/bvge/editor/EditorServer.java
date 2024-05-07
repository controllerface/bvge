package com.controllerface.bvge.editor;

import com.controllerface.bvge.editor.http.Header;
import com.controllerface.bvge.editor.http.Request;
import com.controllerface.bvge.editor.http.RequestLine;

import java.io.BufferedInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.net.ServerSocket;
import java.net.Socket;
import java.net.SocketException;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.function.Predicate;

public class EditorServer
{
    private final int port;
    private final long EVENT_INTERVAL = Duration.ofMillis(200).toMillis();
    private ServerSocket server_socket;
    private Thread incoming;
    private Thread outgoing;

    private static final byte[] EOL_BYTES = new byte[]{'\r', '\n'};
    private static final byte[] EOM_BYTES = new byte[]{'\r', '\n', '\r', '\n'};

    private static final byte[] _404 = ("""
        HTTP/1.1 404 Not Found\r
        Content-Length:9\r
        Connection: close\r
        \r
        Not Found""").getBytes(StandardCharsets.UTF_8);

    private final List<EditorStream> streams = new CopyOnWriteArrayList<>();

    private final Map<String, String> stat_events = new ConcurrentHashMap<>();

    @FunctionalInterface
    private interface EndpointHandler
    {
        void respond(Request request, Socket clientConnection, EditorServer server);
    }

    private enum EndpointMethod
    {
        GET,
        POST,
        ANY;

        boolean matches(String method)
        {
            return method.equalsIgnoreCase(this.name());
        }
    }

    private enum EndPoint
    {
        NOT_FOUND(EndpointMethod.ANY,
            (_) -> false,
            EditorServer::respond_404),

        EVENT_SOURCE(EndpointMethod.GET,
            "/events",
            EditorServer::handle_sse),

        STATIC_ASSET(EndpointMethod.GET,
            StaticAsset::is_asset,
            StaticAsset::serve_asset);

        private final EndpointMethod method;
        private final Predicate<String> uri_filter;
        private final EndpointHandler handler;

        EndPoint(EndpointMethod method,
                 String uri,
                 EndpointHandler handler)
        {
            this(method, (test_uri) -> test_uri.equals(uri), handler);
        }

        EndPoint(EndpointMethod method,
                 Predicate<String> uri_filter,
                 EndpointHandler handler)
        {
            this.method = method;
            this.uri_filter = uri_filter;
            this.handler = handler;
        }

        public void handle(Request request, Socket client_connection, EditorServer server)
        {
            this.handler.respond(request, client_connection, server);
        }

        public static void handleRequest(Request request, Socket client_connection, EditorServer server)
        {
            Arrays.stream(EndPoint.values())
                .filter(endPoint -> endPoint.method.matches(request.method()))
                .filter(endpoint -> endpoint.uri_filter.test(request.uri()))
                .findFirst().orElse(NOT_FOUND)
                .handle(request, client_connection, server);
        }
    }

    public EditorServer(int port)
    {
        this.port = port;
    }

    private static void respond_404(Request request, Socket client_connection, EditorServer server)
    {
        try (client_connection;
             var response_stream = client_connection.getOutputStream())
        {
            response_stream.write(EditorServer._404);
            response_stream.flush();
        }
        catch (IOException ioException)
        {
            System.out.println("Error writing to response stream");
        }
    }

    private static void handle_sse(Request request, Socket client_connection, EditorServer server)
    {
        boolean isSSE = request.headers().stream()
            .filter(header -> "Accept".equalsIgnoreCase(header.name()))
            .anyMatch(header -> "text/event-stream".equals(header.value()));

        if (isSSE)
        {
            var new_stream = new EditorStream(client_connection, server);
            new_stream.start();
            server.add_stream(new_stream);
        }
        else
        {
            EndPoint.NOT_FOUND.handle(request, client_connection, server);
        }
    }

    private Request read_request(Socket client_connection) throws IOException
    {
        var request_stream = new BufferedInputStream(client_connection.getInputStream());
        int next;
        var input_buffer = new ByteArrayOutputStream();
        byte[] eom_buffer = new byte[4];
        boolean EOM = false;
        boolean error = false;
        var line = (RequestLine) null;
        var headers = new ArrayList<Header>();
        while (!error && !EOM && ((next = request_stream.read()) != -1))
        {
            input_buffer.write(next);
            eom_buffer[0] = eom_buffer[1];
            eom_buffer[1] = eom_buffer[2];
            eom_buffer[2] = eom_buffer[3];
            eom_buffer[3] = (byte) next;
            EOM = Arrays.compare(EOM_BYTES, eom_buffer) == 0;
            var EOL = Arrays.compare(EOL_BYTES, 0, EOL_BYTES.length, eom_buffer, 2, eom_buffer.length) == 0;
            if (!EOM && EOL)
            {
                var raw_header = input_buffer.toString(StandardCharsets.UTF_8);
                input_buffer.reset();
                if (line == null)
                {
                    var tokens = raw_header.trim().split(" ", 3);
                    if (tokens.length != 3)
                    {
                        error = true;
                        continue;
                    }
                    line = new RequestLine(tokens[0], tokens[1], tokens[2]);
                }
                else
                {
                    var tokens = raw_header.split(":", 2);
                    if (tokens.length < 1)
                    {
                        error = true;
                        continue;
                    }
                    headers.add(new Header(tokens[0].trim(), tokens[1].trim()));
                }
            }
        }
        if (error)
        {
            System.err.println("Invalid header, aborting connection");
        }
        else
        {
            return new Request(line, headers);
        }
        return null;
    }

    private void process_client(Socket client_connection)
    {
        try
        {
            var request = Objects.requireNonNull(read_request(client_connection));
            EndPoint.handleRequest(request, client_connection, this);
        }
        catch (Exception _)
        {
            try
            {
                client_connection.close();
            }
            catch (Exception _) { }
        }
    }

    private void accept_connection()
    {
        try
        {
            while (!Thread.currentThread().isInterrupted())
            {
                var new_client = server_socket.accept();
                Thread.ofVirtual().start(() -> process_client(new_client));
            }
        }
        catch (SocketException se)
        {
            System.out.println("EditorServer terminated");
        }
        catch (IOException e)
        {
            System.err.println("Error accepting client socket");
        }
    }

    private void update_connection()
    {
        while (!Thread.currentThread().isInterrupted())
        {
            try
            {
                //noinspection BusyWait
                Thread.sleep(EVENT_INTERVAL);
                stat_events.forEach((name, value) ->
                    streams.forEach(stream -> stream.queue_event(name, value)));
            }
            catch (Exception _)
            {
                Thread.currentThread().interrupt();
            }
        }
    }

    public void queue_stat_event(String name, String value)
    {
        stat_events.put(name, value);
    }

    public void add_stream(EditorStream stream)
    {
        streams.add(stream);
    }

    public void end_stream(EditorStream stream)
    {
        streams.remove(stream);
    }

    public void start()
    {
        try
        {
            server_socket = new ServerSocket(port);
            incoming = Thread.ofVirtual().start(this::accept_connection);
            outgoing = Thread.ofVirtual().start(this::update_connection);
        }
        catch (IOException e)
        {
            throw new RuntimeException(STR."Error starting editor server. port: \{port}", e);
        }
    }

    public void stop()
    {
        try
        {
            streams.forEach(EditorStream::stop);
            incoming.interrupt();
            outgoing.interrupt();
            server_socket.close();
        }
        catch (IOException e)
        {
            throw new RuntimeException(STR."Error stopping editor server. port: \{port}", e);
        }
    }
}
