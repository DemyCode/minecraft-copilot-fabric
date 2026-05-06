package net.demycode.copilot.inference;

import ai.onnxruntime.OrtException;
import net.demycode.copilot.CopilotClient;

import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.function.Consumer;

public class DiffusionSampler {

    private static final int NUM_STEPS = 20;

    private final OnnxSession session;
    private final BlockMapper mapper;
    private final ExecutorService executor = Executors.newSingleThreadExecutor(r -> {
        Thread t = new Thread(r, "copilot-diffusion");
        t.setDaemon(true);
        return t;
    });
    private final Random rng = new Random();
    private Future<?> currentTask;

    public DiffusionSampler(OnnxSession session, BlockMapper mapper) {
        this.session = session;
        this.mapper = mapper;
    }

    public void submit(long[] conditionBlocks, boolean[] condMask, Consumer<long[]> stepCallback) {
        if (currentTask != null) currentTask.cancel(true);

        int cs = mapper.getChunkSize();
        int n = cs * cs * cs;
        int maskIdx = mapper.getMaskIdx();

        currentTask = executor.submit(() -> {
            long[] x = new long[n];
            for (int i = 0; i < n; i++) x[i] = condMask[i] ? conditionBlocks[i] : maskIdx;

            try {
                float[] lastLogits = null;
                for (int step = 0; step < NUM_STEPS; step++) {
                    if (Thread.currentThread().isInterrupted()) return;
                    float tNow  = 1.0f - (float) step / NUM_STEPS;
                    float tNext = 1.0f - (float) (step + 1) / NUM_STEPS;

                    long t0 = System.currentTimeMillis();
                    lastLogits = session.forward(x, condMask, tNow, cs);
                    CopilotClient.LOGGER.info("[Copilot] onnx forward {}ms", System.currentTimeMillis() - t0);
                    long[] x0 = sampleFromLogits(lastLogits, cs);

                    float unmaskProb = tNow > 1e-6f ? (tNow - tNext) / tNow : 1.0f;
                    int unmasked = 0;
                    for (int i = 0; i < n; i++) {
                        if (!condMask[i] && x[i] == maskIdx && rng.nextFloat() < unmaskProb) {
                            x[i] = x0[i];
                            unmasked++;
                        }
                    }
                    long nonAir = 0;
                    for (int i = 0; i < n; i++) if (!condMask[i] && x[i] != maskIdx && x[i] != 0) nonAir++;
                    CopilotClient.LOGGER.info("[Copilot] step {}/{} unmasked={} nonAirPredicted={}", step+1, NUM_STEPS, unmasked, nonAir);
                    stepCallback.accept(x.clone());
                }
                if (lastLogits != null) {
                    for (int i = 0; i < n; i++) {
                        if (!condMask[i] && x[i] == maskIdx)
                            x[i] = argmax(lastLogits, i, mapper.getVocabSize());
                    }
                }
                stepCallback.accept(x.clone());
            } catch (OrtException e) {
                CopilotClient.LOGGER.error("ONNX inference failed: {}", e.getMessage());
            }
        });
    }

    private long[] sampleFromLogits(float[] logits, int cs) {
        int n = cs * cs * cs;
        int vocabSize = mapper.getVocabSize() + 1;
        long[] result = new long[n];
        for (int i = 0; i < n; i++) {
            float max = Float.NEGATIVE_INFINITY;
            for (int c = 0; c < vocabSize; c++) { float v = logits[c * n + i]; if (v > max) max = v; }
            float sum = 0;
            float[] probs = new float[vocabSize];
            for (int c = 0; c < vocabSize; c++) { probs[c] = (float) Math.exp(logits[c * n + i] - max); sum += probs[c]; }
            float r = rng.nextFloat() * sum;
            int chosen = vocabSize - 1;
            for (int c = 0; c < vocabSize; c++) { r -= probs[c]; if (r <= 0) { chosen = c; break; } }
            result[i] = chosen;
        }
        return result;
    }

    private long argmax(float[] logits, int idx, int vocabSize) {
        int n = logits.length / vocabSize;
        long best = 0; float bestVal = Float.NEGATIVE_INFINITY;
        for (int c = 0; c < vocabSize; c++) { float v = logits[c * n + idx]; if (v > bestVal) { bestVal = v; best = c; } }
        return best;
    }
}
