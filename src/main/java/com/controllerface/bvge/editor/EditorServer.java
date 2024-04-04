package com.controllerface.bvge.editor;

import java.io.BufferedInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.net.*;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.Predicate;

public class EditorServer
{
    private final int port = 9000;
    private ServerSocket serverSocket;
    private Thread acceptor;
    private final int[] EOL_BYTES = new int[]{ '\r', '\n' };
    private final int[] EOM_BYTES = new int[]{ '\r', '\n', '\r', '\n' };

    private static final String CANNED_TEST = "HTTP/1.1 200 OK\r\n" +
        "Content-Length:6\r\n" +
        "Connection: close\r\n" +
        "\r\n" +
        "Hello!";

    private static final String _404 = "HTTP/1.1 404 Not Found\r\n" +
        "Content-Length:9\r\n" +
        "Connection: close\r\n" +
        "\r\n" +
        "Not Found";

    @FunctionalInterface
    private interface EndpointHandler
    {
        void respond(Request request, OutputStream response);
    }

    private enum EndpointType
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
            return "{" + method + " " + uri + " " + version + "}";
        }
    }

    private record Header(String name, String value)
    {
        @Override
        public String toString()
        {
            return "[" + name + " : " + value + "]";
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

    private enum EndPoint
    {
        /**
         * Default endpoint handler called when nothing matches.
         */
        NOT_FOUND(EndpointType.ANY, (_uri) -> false, (_req, res) ->
        {
            try (res)
            {
                res.write(_404.getBytes(StandardCharsets.UTF_8));
                res.flush();
            }
            catch (IOException ioe)
            {
                System.err.println("Error writing response");
            }
        }),

        HELLO(EndpointType.GET, "/", (req, res) ->
        {
            try (res)
            {
                res.write(CANNED_TEST.getBytes(StandardCharsets.UTF_8));
                res.flush();
            }
            catch (IOException ioe)
            {
                System.err.println("Error writing response");
            }
        });

        private final EndpointType type;
        private final Predicate<String> uri_filter;
        private final EndpointHandler handler;

        EndPoint(EndpointType type,
                 String uri,
                 EndpointHandler handler)
        {
            this(type, (requestUri) -> requestUri.equals(uri), handler);
        }

        EndPoint(EndpointType type,
                 Predicate<String> uri_filter,
                 EndpointHandler handler)
        {
            this.type = type;
            this.uri_filter = uri_filter;
            this.handler = handler;
        }

        public void handle(Request request, OutputStream response)
        {
            this.handler.respond(request, response);
        }

        public static EndPoint forRequest(Request request)
        {
            return Arrays.stream(EndPoint.values())
                .filter(endPoint -> endPoint.type.matches(request.method()))
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
        try (var request_stream = new BufferedInputStream(clientConnection.getInputStream()))
        {
            int next;
            var input_buffer = new ByteArrayOutputStream();
            int[] eom_buffer =  new int[4];
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
                eom_buffer[3] = next;
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
                EndPoint.forRequest(request)
                    .handle(request, clientConnection.getOutputStream());
            }
        }
        catch (IOException e)
        {
            System.err.println("Client connection closed: " + clientConnection.getRemoteSocketAddress().toString());
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
            throw new RuntimeException("Error starting editor server. port: " + port, e);
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
            throw new RuntimeException("Error stopping editor server. port: " + port, e);
        }
    }
}
