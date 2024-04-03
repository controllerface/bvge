package com.controllerface.bvge.editor;

import java.io.BufferedInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.net.*;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;

public class EditorServer
{
    private final int port = 9000;
    private ServerSocket serverSocket;
    private Thread acceptor;
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
        try
        {
            var buffer = new BufferedInputStream(clientConnection.getInputStream());
            int next;
            var input_buffer = new ByteArrayOutputStream();
            int[] ring_buffer =  new int[4];
            boolean EOM = false;
            while (!EOM && ((next = buffer.read()) != -1))
            {
                input_buffer.write(next);
                ring_buffer[0] = ring_buffer[1];
                ring_buffer[1] = ring_buffer[2];
                ring_buffer[2] = ring_buffer[3];
                ring_buffer[3] = next;
                EOM = Arrays.compare(EOM_BYTES, ring_buffer) == 0;
            }
            System.out.println("debug: \n" + input_buffer.toString(StandardCharsets.UTF_8));
            clientConnection.getOutputStream().write(CANNED_TEST.getBytes(StandardCharsets.UTF_8));
            clientConnection.getOutputStream().flush();
            clientConnection.close();
            System.out.println("Client connection closed: "
                    + ((InetSocketAddress) clientConnection.getRemoteSocketAddress()).getPort());
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
