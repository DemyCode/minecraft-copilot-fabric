package net.demycode.copilot.inference;

import net.demycode.copilot.CopilotClient;
import ai.onnxruntime.*;

import java.nio.ByteBuffer;
import java.nio.LongBuffer;
import java.nio.file.Path;
import java.util.Map;

public class OnnxSession implements AutoCloseable {

    private final OrtEnvironment env;
    private final OrtSession session;

    public OnnxSession(Path modelPath) throws OrtException {
        System.err.println("[Copilot] OnnxSession CONSTRUCTOR: modelPath=" + modelPath);
        CopilotClient.LOGGER.info("[Copilot] OnnxSession: creating session for {}", modelPath);
        env = OrtEnvironment.getEnvironment();
        OrtSession.SessionOptions opts = new OrtSession.SessionOptions();
        int cpus = Runtime.getRuntime().availableProcessors();
        System.err.println("[Copilot] OnnxSession: detected " + cpus + " CPUs");
        opts.setInterOpNumThreads(cpus);
        opts.setIntraOpNumThreads(cpus);
        System.err.println("[Copilot] OnnxSession: calling env.createSession...");
        session = env.createSession(modelPath.toString(), opts);
        System.err.println("[Copilot] OnnxSession: session created!");
        CopilotClient.LOGGER.info("[Copilot] OnnxSession: session created! Inputs:");
        for (String inp : session.getInputNames()) {
            CopilotClient.LOGGER.info("[Copilot]   input name: {}", inp);
        }
        for (String outp : session.getOutputNames()) {
            CopilotClient.LOGGER.info("[Copilot]   output name: {}", outp);
        }
    }

    public float[] forward(long[] xT, boolean[] condMask, float t, int cs) throws OrtException {
        int n = cs * cs * cs;
        System.err.println("[Copilot] OnnxSession.forward: t=" + t + " n=" + n);

        long[] xShape = {1, cs, cs, cs};
        OnnxTensor xTensor = OnnxTensor.createTensor(env, LongBuffer.wrap(xT), xShape);

        long[] maskShape = {1, cs, cs, cs};
        byte[] maskBytes = new byte[n];
        for (int i = 0; i < n; i++) maskBytes[i] = condMask[i] ? (byte) 1 : (byte) 0;
        OnnxTensor maskTensor = OnnxTensor.createTensor(env, ByteBuffer.wrap(maskBytes), maskShape, OnnxJavaType.BOOL);

        float[] tArr = {t};
        OnnxTensor tTensor = OnnxTensor.createTensor(env, tArr);

        System.err.println("[Copilot] OnnxSession.forward: tensors created, calling session.run...");
        try (OrtSession.Result result = session.run(Map.of(
                "x_t", xTensor, "condition_mask", maskTensor, "t", tTensor))) {
            System.err.println("[Copilot] OnnxSession.forward: session.run returned");
            OnnxTensor out = (OnnxTensor) result.get(0);
            java.nio.FloatBuffer fb = out.getFloatBuffer();
            float[] logits = new float[fb.remaining()];
            fb.get(logits);
            System.err.println("[Copilot] OnnxSession.forward: logits length=" + logits.length);
            return logits;
        } finally {
            xTensor.close();
            maskTensor.close();
            tTensor.close();
        }
    }

    @Override
    public String toString() {
        return "OnnxSession@" + Integer.toHexString(System.identityHashCode(this));
    }

    @Override
    public void close() throws OrtException {
        session.close();
        env.close();
    }
}
