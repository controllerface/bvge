package com.controllerface.bvge.editor;

import java.net.Socket;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingDeque;

public class EditorStream
{
    private static final byte[] _200_SSE = ("""
        HTTP/1.1 200 OK\r
        Content-Type: text/event-stream\r
        \r""").getBytes(StandardCharsets.UTF_8);

    private final Socket client_connection;
    private final EditorServer server;
    private Thread task;
    private final BlockingQueue<StatEvent> event_queue = new LinkedBlockingDeque<>();

    private record StatEvent(String name, String value) { }

    public EditorStream(Socket client_connection, EditorServer server)
    {
        this.client_connection = client_connection;
        this.server = server;
    }

    public void stop()
    {
        task.interrupt();
    }

    public void start()
    {
        task = Thread.ofVirtual().start(() ->
        {
            try (client_connection)
            {
                var response_stream = client_connection.getOutputStream();
                response_stream.write(_200_SSE);
                response_stream.flush();
                while (!Thread.currentThread().isInterrupted())
                {
                    var next_event = event_queue.take();
                    var message = "event: " + next_event.name + "\r\ndata: " + next_event.value + "\r\n\r\n";
                    response_stream.write(message.getBytes(StandardCharsets.UTF_8));
                }
            }
            catch (Exception _)
            {
                Thread.currentThread().interrupt();
                server.end_stream(this);
                System.out.println("EventSource terminated");
            }
        });
    }

    public void queue_event(String name, String value)
    {
        event_queue.add(new StatEvent(name, value));
    }
}
