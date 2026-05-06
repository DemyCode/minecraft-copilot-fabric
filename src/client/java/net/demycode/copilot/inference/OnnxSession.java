package net.demycode.copilot.inference;

import ai.onnxruntime.*;

import java.nio.ByteBuffer;
import java.nio.LongBuffer;
import java.nio.file.Path;
import java.util.Map;

public class OnnxSession implements AutoCloseable {

    private final OrtEnvironment env;
    private final OrtSession session;

    public OnnxSession(Path modelPath) throws OrtException {
        env = OrtEnvironment.getEnvironment();
        OrtSession.SessionOptions opts = new OrtSession.SessionOptions();
        int cpus = Runtime.getRuntime().availableProcessors();
        opts.setInterOpNumThreads(cpus);
        opts.setIntraOpNumThreads(cpus);
        session = env.createSession(modelPath.toString(), opts);
    }

    public float[] forward(long[] xT, boolean[] condMask, float t, int cs) throws OrtException {
        int n = cs * cs * cs;

        long[] xShape = {1, cs, cs, cs};
        OnnxTensor xTensor = OnnxTensor.createTensor(env, LongBuffer.wrap(xT), xShape);

        long[] maskShape = {1, cs, cs, cs};
        byte[] maskBytes = new byte[n];
        for (int i = 0; i < n; i++) maskBytes[i] = condMask[i] ? (byte) 1 : (byte) 0;
        OnnxTensor maskTensor = OnnxTensor.createTensor(env, ByteBuffer.wrap(maskBytes), maskShape, OnnxJavaType.BOOL);

        float[] tArr = {t};
        OnnxTensor tTensor = OnnxTensor.createTensor(env, tArr);

        try (OrtSession.Result result = session.run(Map.of(
                "x_t", xTensor, "condition_mask", maskTensor, "t", tTensor))) {
            return (float[]) ((OnnxTensor) result.get(0)).getValue();
        } finally {
            xTensor.close();
            maskTensor.close();
            tTensor.close();
        }
    }

    @Override
    public void close() throws OrtException {
        session.close();
        env.close();
    }
}
