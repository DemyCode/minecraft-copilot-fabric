package net.demycode.copilot.inference;

import ai.onnxruntime.OrtException;
import net.demycode.copilot.CopilotClient;

import java.util.Random;
import java.util.concurrent.*;
import java.util.function.Consumer;

public class DiffusionSampler {

    private static final int NUM_STEPS = 20;

    private final OnnxSession session;
    private final BlockMapper mapper;
    private final ExecutorService executor = Executors.newSingleThreadExecutor(r -> {
        Thread t = new Thread(r, "copilot-diffusion");
        t.setDaemon(true);
        t.setUncaughtExceptionHandler((thread, ex) -> {
            System.err.println("[Copilot] UNCAUGHT EXCEPTION in " + thread.getName() + ": " + ex);
            ex.printStackTrace(System.err);
            CopilotClient.LOGGER.error("[Copilot] UNCAUGHT in diffusion thread", ex);
        });
        return t;
    });
    private final ScheduledExecutorService watchdog = Executors.newSingleThreadScheduledExecutor(r -> {
        Thread t = new Thread(r, "copilot-watchdog");
        t.setDaemon(true);
        return t;
    });
    private final Random rng = new Random();
    private Future<?> currentTask;
    private volatile int watchdogStep = -1;
    private volatile String watchdogPhase = "idle";

    public DiffusionSampler(OnnxSession session, BlockMapper mapper) {
        this.session = session;
        this.mapper = mapper;
        System.err.println("[Copilot] DiffusionSampler CONSTRUCTOR called. session=" + session + " mapper=" + mapper);
        CopilotClient.LOGGER.info("[Copilot] DiffusionSampler CONSTRUCTOR: ready. session={} mapper={}", session, mapper);
        watchdog.scheduleAtFixedRate(() -> {
            if (!"idle".equals(watchdogPhase)) {
                String msg = "[Copilot] WATCHDOG: phase=" + watchdogPhase + " step=" + watchdogStep + "/" + NUM_STEPS;
                System.err.println(msg);
                CopilotClient.LOGGER.info(msg);
            }
        }, 5, 5, TimeUnit.SECONDS);
    }

    public void submit(long[] conditionBlocks, boolean[] condMask, Consumer<long[]> stepCallback) {
        System.err.println("[Copilot] DiffusionSampler.submit() ENTERED");
        CopilotClient.LOGGER.info("[Copilot] DiffusionSampler.submit() ENTERED");

        if (currentTask != null && !currentTask.isDone()) {
            System.err.println("[Copilot] DiffusionSampler: cancelling previous task");
            currentTask.cancel(true);
        }

        int cs = mapper.getChunkSize();
        int n = cs * cs * cs;
        int maskIdx = mapper.getMaskIdx();

        System.err.println("[Copilot] DiffusionSampler.submit: cs=" + cs + " n=" + n + " maskIdx=" + maskIdx);
        CopilotClient.LOGGER.info("[Copilot] DiffusionSampler.submit: cs={} n={} maskIdx={}", cs, n, maskIdx);

        watchdogStep = 0;
        watchdogPhase = "submitted";

        currentTask = executor.submit(() -> {
            System.err.println("[Copilot] DiffusionSampler: background thread STARTED on " + Thread.currentThread().getName());
            CopilotClient.LOGGER.info("[Copilot] DiffusionSampler: background thread STARTED on {}", Thread.currentThread().getName());
            watchdogPhase = "running";

            long[] x = new long[n];
            for (int i = 0; i < n; i++) x[i] = condMask[i] ? conditionBlocks[i] : maskIdx;

            int condCount = 0;
            for (boolean b : condMask) if (b) condCount++;
            System.err.println("[Copilot] Initial x: " + condCount + " conditioned, " + (n - condCount) + " masked");
            CopilotClient.LOGGER.info("[Copilot] Initial x: {} conditioned, {} masked", condCount, n - condCount);

            try {
                float[] lastLogits = null;
                for (int step = 0; step < NUM_STEPS; step++) {
                    if (Thread.currentThread().isInterrupted()) {
                        System.err.println("[Copilot] Thread interrupted at step " + step);
                        CopilotClient.LOGGER.info("[Copilot] Thread interrupted at step {}", step);
                        watchdogPhase = "interrupted";
                        return;
                    }
                    float tNow  = 1.0f - (float) step / NUM_STEPS;
                    float tNext = 1.0f - (float) (step + 1) / NUM_STEPS;

                    watchdogStep = step;
                    watchdogPhase = "onnx_step_" + step;

                    long t0 = System.currentTimeMillis();
                    System.err.println("[Copilot] step " + step + ": calling session.forward tNow=" + tNow);
                    CopilotClient.LOGGER.info("[Copilot] step {}: calling session.forward tNow={}", step, tNow);

                    lastLogits = session.forward(x, condMask, tNow, cs);

                    long elapsed = System.currentTimeMillis() - t0;
                    System.err.println("[Copilot] step " + step + ": forward done in " + elapsed + "ms");
                    CopilotClient.LOGGER.info("[Copilot] step {}: forward done in {}ms", step, elapsed);

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
                    CopilotClient.LOGGER.info("[Copilot] step {}/{} unmasked={} nonAirPredicted={}", step + 1, NUM_STEPS, unmasked, nonAir);
                    watchdogPhase = "callback_step_" + step;
                    stepCallback.accept(x.clone());
                }
                watchdogPhase = "finalizing";
                if (lastLogits != null) {
                    for (int i = 0; i < n; i++) {
                        if (!condMask[i] && x[i] == maskIdx)
                            x[i] = argmax(lastLogits, i, mapper.getVocabSize());
                    }
                }
                stepCallback.accept(x.clone());
                watchdogPhase = "idle";
                System.err.println("[Copilot] DiffusionSampler: inference COMPLETED");
                CopilotClient.LOGGER.info("[Copilot] DiffusionSampler: inference COMPLETED normally");
            } catch (OrtException e) {
                watchdogPhase = "error";
                System.err.println("[Copilot] OrtException: " + e.getMessage());
                CopilotClient.LOGGER.error("[Copilot] ONNX inference failed: {}", e.getMessage(), e);
            } catch (Exception e) {
                watchdogPhase = "error";
                System.err.println("[Copilot] Unexpected exception in diffusion thread: " + e);
                e.printStackTrace(System.err);
                CopilotClient.LOGGER.error("[Copilot] Unexpected exception in diffusion thread", e);
            }
        });

        System.err.println("[Copilot] DiffusionSampler.submit: task submitted, future=" + currentTask);
        CopilotClient.LOGGER.info("[Copilot] DiffusionSampler.submit: task submitted");
    }

    private long[] sampleFromLogits(float[] logits, int cs) {
        int n = cs * cs * cs;
        int vocabSize = mapper.getVocabSize();
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
