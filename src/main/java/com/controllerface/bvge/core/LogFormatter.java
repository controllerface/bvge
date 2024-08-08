package com.controllerface.bvge.core;

import java.io.PrintWriter;
import java.io.StringWriter;
import java.time.ZoneId;
import java.util.IllegalFormatException;
import java.util.logging.LogManager;
import java.util.logging.LogRecord;
import java.util.logging.SimpleFormatter;

public class LogFormatter extends SimpleFormatter
{
    private static final String FORMAT_PROPERTY = "java.util.logging.SimpleFormatter.format";
    private static final String DEFAULT_FORMAT = "[%1$tc] %4$s: %2$s() - %5$s %6$s%n";
    private static final ZoneId ZONE = ZoneId.systemDefault();

    private static String stripPackage(String name)
    {
        return "{ " + name.substring(name.lastIndexOf('.') + 1) + " }";
    }

    @Override
    public String format(LogRecord record)
    {
        String formatString = System.getProperty(FORMAT_PROPERTY);
        if (formatString == null)
        {
            formatString = LogManager.getLogManager().getProperty(FORMAT_PROPERTY);
        }
        if (formatString == null)
        {
            formatString = DEFAULT_FORMAT;
        }

        String name = record.getLoggerName();
        name = stripPackage(name);

        String sourceClass = record.getSourceClassName();
        String sourceMethod = record.getSourceMethodName();
        String source = sourceClass != null && sourceMethod != null
            ? stripPackage(sourceClass) + ' ' + sourceMethod
            : name;

        String stackTrace = "";
        if (record.getThrown() != null)
        {
            StringWriter writer = new StringWriter();
            record.getThrown().printStackTrace(new PrintWriter(writer));
            stackTrace = System.lineSeparator() + writer;
        }

        Object[] args =
        {
            record.getInstant().atZone(ZONE),
            source,
            name,
            record.getLevel().getLocalizedName(),
            formatMessage(record),
            stackTrace
        };

        try
        {
            return String.format(formatString, args);
        }
        catch (IllegalFormatException e)
        {
            return String.format(DEFAULT_FORMAT, args);
        }
    }
}
