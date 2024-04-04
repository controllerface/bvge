package com.controllerface.bvge.editor;

import java.io.BufferedInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.net.ServerSocket;
import java.net.Socket;
import java.net.SocketException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.Predicate;

import static com.controllerface.bvge.editor.StaticAsset.staticAssets;

public class EditorServer
{
    private final int port = 9000;
    private ServerSocket serverSocket;
    private Thread acceptor;

    private static final byte[] EOL_BYTES = new byte[]{'\r', '\n'};
    private static final byte[] EOM_BYTES = new byte[]{'\r', '\n', '\r', '\n'};

    private static final byte[] _404 = ("""
        HTTP/1.1 404 Not Found\r
        Content-Length:9\r
        Connection: close\r
        \r
        Not Found""").getBytes(StandardCharsets.UTF_8);

    private static final byte[] _200_SSE = ("""
        HTTP/1.1 200 OK\r
        Content-Type: text/event-stream\r
        \r""").getBytes(StandardCharsets.UTF_8);

    @FunctionalInterface
    private interface EndpointHandler
    {
        void respond(Request request, Socket clientConnection);
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

    private record RequestLine(String method, String uri, String version)
    {
        @Override
        public String toString()
        {
            return STR."{\{method} \{uri} \{version}}";
        }
    }

    private record Header(String name, String value)
    {
        @Override
        public String toString()
        {
            return STR."[\{name} : \{value}]";
        }
    }

    private record Request(RequestLine requestLine, List<Header> headers)
    {
        @Override
        public String toString()
        {
            var buffer = new StringBuilder();
            buffer.append("\n").append(requestLine.toString()).append("\n");
            headers.forEach(header -> buffer.append(header).append("\n"));
            buffer.append("\n");
            return buffer.toString();
        }

        public String method()
        {
            return requestLine().method();
        }

        public String uri()
        {
            return requestLine().uri();
        }
    }

    private static void handle_404(Request request, Socket client_connection)
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

    private static void handle_sse(Request request, Socket client_connection)
    {
        boolean isSSE = request.headers.stream()
            .filter(header -> "Accept".equalsIgnoreCase(header.name()))
            .anyMatch(header -> "text/event-stream".equals(header.value()));

        if (isSSE)
        {
            try
            {
                var response_stream = client_connection.getOutputStream();
                response_stream.write(EditorServer._200_SSE);
                response_stream.flush();
                Thread.ofVirtual().start(()->
                {
                   while (!Thread.currentThread().isInterrupted())
                   {
                       try
                       {
                           Thread.sleep(2000);
                           client_connection.getOutputStream().write("event: test_event\r\n".getBytes(StandardCharsets.UTF_8));
                           client_connection.getOutputStream().write((STR."data: test_data = \{System.currentTimeMillis()}\r\n").getBytes(StandardCharsets.UTF_8));
                           client_connection.getOutputStream().write("\r\n".getBytes(StandardCharsets.UTF_8));
                       }
                       catch (Exception _)
                       {
                           System.out.println("broken connection");
                           Thread.currentThread().interrupt();
                       }
                   }
                });
            }
            catch (Exception _)
            {
                EndPoint.NOT_FOUND.handle(request, client_connection);
            }
        }
        else
        {
            EndPoint.NOT_FOUND.handle(request, client_connection);
        }
    }

    private static void handle_static_asset(Request request, Socket clientConnection)
    {
        staticAssets.get(request.uri()).writeTo(clientConnection);
    }

    private enum EndPoint
    {
        NOT_FOUND(EndpointMethod.ANY,
            (_) -> false,
            EditorServer::handle_404),

        EVENT_SOURCE(EndpointMethod.GET,
        "/events",
            EditorServer::handle_sse),

        STATIC_ASSET(EndpointMethod.GET,
            staticAssets::containsKey,
            EditorServer::handle_static_asset);

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

        public void handle(Request request, Socket client_connection)
        {
            this.handler.respond(request, client_connection);
        }

        public static void handleRequest(Request request, Socket client_connection)
        {
            EndPoint.forRequest(request).handle(request, client_connection);
        }

        public static EndPoint forRequest(Request request)
        {
            return Arrays.stream(EndPoint.values())
                .filter(endPoint -> endPoint.method.matches(request.method()))
                .filter(endpoint -> endpoint.uri_filter.test(request.uri()))
                .findFirst().orElse(NOT_FOUND);
        }
    }

    private void accept()
    {
        try
        {
            while (!Thread.currentThread().isInterrupted())
            {
                var client = serverSocket.accept();
                Thread.ofVirtual().start(() -> process(client));
            }
        }
        catch (SocketException se)
        {
            System.out.println("Acceptor exiting, server shutting down");
        }
        catch (IOException e)
        {
            System.err.println("Error accepting client socket");
        }
    }

    private void process(Socket clientConnection)
    {
        try
        {
            var request_stream = new BufferedInputStream(clientConnection.getInputStream());
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
                var request = new Request(line, headers);
                System.out.println(request);
                EndPoint.handleRequest(request, clientConnection);
            }
        }
        catch (IOException e)
        {
            try
            {
                clientConnection.close();
            }
            catch (IOException _) { /* NOP for closing broken connection */}
        }
    }

    public void start()
    {
        try
        {
            serverSocket = new ServerSocket(port);
            acceptor = Thread.ofVirtual().start(this::accept);
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
            acceptor.interrupt();
            serverSocket.close();
        }
        catch (IOException e)
        {
            throw new RuntimeException(STR."Error stopping editor server. port: \{port}", e);
        }
    }
}
