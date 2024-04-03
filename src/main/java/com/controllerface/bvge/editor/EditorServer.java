package com.controllerface.bvge.editor;

import java.io.BufferedInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.net.*;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;

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

    private void accept()
    {
        try
        {
            while (!Thread.currentThread().isInterrupted())
            {
                var client = serverSocket.accept();
                System.out.println("new client connection");
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
        try (var request_stream = new BufferedInputStream(clientConnection.getInputStream());
             var response_stream = clientConnection.getOutputStream())
        {
            int next;
            var input_buffer = new ByteArrayOutputStream();
            int[] eol_buffer =  new int[2];
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
                eol_buffer[0] = eol_buffer[1];
                eol_buffer[1] = next;
                EOM = Arrays.compare(EOM_BYTES, eom_buffer) == 0;
                if (!EOM && Arrays.compare(EOL_BYTES, eol_buffer) == 0)
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
                System.out.println("\n");
                System.out.println(line);
                headers.forEach(System.out::println);
                System.out.println("\n");
                response_stream.write(CANNED_TEST.getBytes(StandardCharsets.UTF_8));
                response_stream.flush();
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
            System.out.println("Editor server starting up...");
            serverSocket = new ServerSocket(port);
            acceptor = Thread.ofVirtual().start(this::accept);
            System.out.println("Editor server running");
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
            System.out.println("Editor server shutting down...");
            acceptor.interrupt();
            serverSocket.close();
            System.out.println("Editor server stopped");
        }
        catch (IOException e)
        {
            throw new RuntimeException("Error stopping editor server. port: " + port, e);
        }
    }
}
