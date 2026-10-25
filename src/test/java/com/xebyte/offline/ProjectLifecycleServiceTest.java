package com.xebyte.offline;

import com.xebyte.core.ProjectLifecycleService;
import com.xebyte.core.Response;
import com.xebyte.headless.HeadlessProgramProvider;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 * The GZF and GAR tools, now served by both servers: the argument checks that come before
 * any project work, so a provider with no project reaches them.
 */
public class ProjectLifecycleServiceTest {

    private final ProjectLifecycleService service = new ProjectLifecycleService(new HeadlessProgramProvider());

    /**
     * An output directory that exists on every OS. These checks run after the directory is
     * validated, so a POSIX-only TMP failed them on Windows for the wrong reason.
     */
    private static final String TMP = System.getProperty("java.io.tmpdir");

    private static String message(Response r) {
        assertTrue(r.toString(), r instanceof Response.Err);
        return ((Response.Err) r).message();
    }

    @Test
    public void exportNeedsAProgramName() {
        assertEquals("program_name required", message(service.exportProgram("", TMP, "")));
    }

    @Test
    public void importNeedsAGzfPath() {
        assertEquals("gzf_path required", message(service.importProgram("", "/", "", false)));
    }

    @Test
    public void restoreNeedsAnArchivePath() {
        assertEquals("gar_path required", message(service.restoreProject("", TMP, "p")));
    }

    @Test
    public void archiveWithNoProjectSaysToOpenOne() {
        assertTrue(message(service.archiveProject(TMP, "")).contains("No project open"));
    }

    @Test
    public void anOutputNameCannotEscapeTheOutputDirectory() {
        assertTrue(message(service.exportProgram("p", TMP, "../x.gzf")).contains("invalid output_name"));
        assertTrue(message(service.archiveProject(TMP, "../x.gar")).contains("invalid output_name")
            || message(service.archiveProject(TMP, "../x.gar")).contains("No project open"));
    }
}
