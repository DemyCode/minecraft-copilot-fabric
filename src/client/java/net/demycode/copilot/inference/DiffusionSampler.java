package net.demycode.copilot.inference;

import ai.onnxruntime.OrtException;
import net.demycode.copilot.CopilotClient;
import net.minecraft.client.Minecraft;
import net.minecraft.network.chat.Component;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.*;
import java.util.function.Consumer;

public class DiffusionSampler {

    private static final int NUM_STEPS = 100;
    // SPARSE mode: non-air predictions below this softmax confidence are forced to air
    private static final float SPARSE_THRESHOLD = 0.5f;

    private final OnnxSession session;
    private final BlockMapper mapper;
    private final java.util.concurrent.ThreadPoolExecutor executor = new java.util.concurrent.ThreadPoolExecutor(
        1, 1, 0L, TimeUnit.MILLISECONDS,
        new java.util.concurrent.LinkedBlockingQueue<>(),
        r -> {
            Thread t = new Thread(r, "copilot-diffusion");
            t.setDaemon(true);
            t.setUncaughtExceptionHandler((thread, ex) -> {
                System.err.println("[Copilot] UNCAUGHT EXCEPTION in " + thread.getName() + ": " + ex);
                ex.printStackTrace(System.err);
                CopilotClient.LOGGER.error("[Copilot] UNCAUGHT in diffusion thread", ex);
            });
            return t;
        }
    );
    private final ScheduledExecutorService watchdog = Executors.newSingleThreadScheduledExecutor(r -> {
        Thread t = new Thread(r, "copilot-watchdog");
        t.setDaemon(true);
        return t;
    });
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

    public void submit(long[] conditionBlocks, boolean[] condMask, Consumer<long[]> stepCallback, InferMode mode) {
        System.err.println("[Copilot] DiffusionSampler.submit() ENTERED");
        CopilotClient.LOGGER.info("[Copilot] DiffusionSampler.submit() ENTERED");

        if (currentTask != null && !currentTask.isDone()) {
            System.err.println("[Copilot] DiffusionSampler: cancelling previous task");
            currentTask.cancel(true);
        }
        executor.purge(); // drop any pending-but-not-started tasks from the queue

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
                if (mode == InferMode.CLASSIC) {
                    watchdogPhase = "classic_forward";
                    float[] logits = session.forward(x, condMask, 1.0f, cs);
                    int vocabSize = mapper.getVocabSize();
                    for (int i = 0; i < n; i++) {
                        if (!condMask[i]) x[i] = argmax(logits, i, vocabSize);
                    }
                    stepCallback.accept(x.clone());
                    watchdogPhase = "idle";
                    Minecraft.getInstance().execute(() -> {
                        var player = Minecraft.getInstance().player;
                        if (player != null)
                            player.sendOverlayMessage(Component.literal("[Copilot] Classic done"));
                    });
                    CopilotClient.LOGGER.info("[Copilot] Classic inference COMPLETED");
                    return;
                }

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

                    // Show t value in action bar (overlay) every step
                    final int stepSnap = step;
                    final float tSnap = tNow;
                    Minecraft.getInstance().execute(() -> {
                        var player = Minecraft.getInstance().player;
                        if (player != null) {
                            player.sendOverlayMessage(
                                Component.literal("[Copilot] t=" + String.format("%.2f", tSnap) + " (step " + stepSnap + "/" + NUM_STEPS + ")"));
                        }
                    });

                    long t0 = System.currentTimeMillis();
                    System.err.println("[Copilot] step " + step + ": calling session.forward tNow=" + tNow);
                    CopilotClient.LOGGER.info("[Copilot] step {}: calling session.forward tNow={}", step, tNow);

                    lastLogits = session.forward(x, condMask, tNow, cs);

                    long elapsed = System.currentTimeMillis() - t0;
                    System.err.println("[Copilot] step " + step + ": forward done in " + elapsed + "ms");
                    CopilotClient.LOGGER.info("[Copilot] step {}: forward done in {}ms", step, elapsed);

                    // Compute argmax predictions and softmax confidence for each position
                    int vocabSize = mapper.getVocabSize();
                    long[] x0 = new long[n];
                    float[] confidence = new float[n];
                    for (int i = 0; i < n; i++) {
                        float maxLogit = Float.NEGATIVE_INFINITY;
                        int argmax = 0;
                        for (int c = 0; c < vocabSize; c++) {
                            float v = lastLogits[c * n + i];
                            if (v > maxLogit) { maxLogit = v; argmax = c; }
                        }
                        // softmax(argmax) = exp(0) / sum_c(exp(logit_c - maxLogit)) = 1 / sum
                        float sum = 0;
                        for (int c = 0; c < vocabSize; c++) {
                            sum += Math.exp(lastLogits[c * n + i] - maxLogit);
                        }
                        x0[i] = argmax;
                        confidence[i] = 1.0f / sum;
                    }

                    // Collect still-masked positions; only unmask non-air predictions greedily.
                    // Air positions stay masked until the final argmax pass — avoids wasting
                    // 80-90% of steps confirming air that the model already settled on early.
                    List<Integer> stillMasked = new ArrayList<>();
                    List<Integer> nonAirCandidates = new ArrayList<>();
                    for (int i = 0; i < n; i++) {
                        if (!condMask[i] && x[i] == maskIdx) {
                            stillMasked.add(i);
                            if (x0[i] != 0) nonAirCandidates.add(i);
                        }
                    }
                    if (stillMasked.isEmpty()) break;

                    if (nonAirCandidates.isEmpty()) {
                        // All remaining masked positions predict air — commit and stop.
                        CopilotClient.LOGGER.info("[Copilot] step {}/{}: all remaining predict air, stopping early", step + 1, NUM_STEPS);
                        for (int i : stillMasked) x[i] = x0[i];
                        stepCallback.accept(x.clone());
                        stillMasked.clear(); // signal outer finalizer to skip
                        break;
                    }

                    // Unmask the most-confident non-air positions (MASKGIT schedule).
                    // SPARSE mode: positions below the confidence threshold are committed as air
                    // immediately, preventing low-confidence blocks from snowballing into fill.
                    if (tNow <= 1e-6f) {
                        for (int i : nonAirCandidates)
                            x[i] = (mode == InferMode.SPARSE && confidence[i] < SPARSE_THRESHOLD) ? 0 : x0[i];
                    } else {
                        float unmaskProb = (tNow - tNext) / tNow;
                        int nToUnmask = Math.max(1, Math.round(nonAirCandidates.size() * unmaskProb));

                        nonAirCandidates.sort((a, b) -> Float.compare(confidence[b], confidence[a]));
                        for (int k = 0; k < nToUnmask; k++) {
                            int i = nonAirCandidates.get(k);
                            x[i] = (mode == InferMode.SPARSE && confidence[i] < SPARSE_THRESHOLD) ? 0 : x0[i];
                        }
                    }

                    long remaining = 0, nonAir = 0;
                    for (int i = 0; i < n; i++) {
                        if (!condMask[i] && x[i] == maskIdx) remaining++;
                        else if (!condMask[i] && x[i] != 0) nonAir++;
                    }
                    CopilotClient.LOGGER.info("[Copilot] step {}/{} remaining={} nonAirPredicted={}", step + 1, NUM_STEPS, remaining, nonAir);
                    watchdogPhase = "callback_step_" + step;
                    stepCallback.accept(x.clone());
                }

                // Fill any remaining masked positions with argmax from last logits
                watchdogPhase = "finalizing";
                if (lastLogits != null) {
                    int vocabSize = mapper.getVocabSize();
                    for (int i = 0; i < n; i++) {
                        if (!condMask[i] && x[i] == maskIdx) {
                            long pred = argmax(lastLogits, i, vocabSize);
                            if (mode == InferMode.SPARSE && pred != 0) {
                                float maxLogit = Float.NEGATIVE_INFINITY;
                                for (int c = 0; c < vocabSize; c++) {
                                    float v = lastLogits[c * n + i];
                                    if (v > maxLogit) maxLogit = v;
                                }
                                float sum = 0;
                                for (int c = 0; c < vocabSize; c++) sum += (float) Math.exp(lastLogits[c * n + i] - maxLogit);
                                pred = (1.0f / sum >= SPARSE_THRESHOLD) ? pred : 0;
                            }
                            x[i] = pred;
                        }
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

    private long argmax(float[] logits, int idx, int vocabSize) {
        int n = logits.length / vocabSize;
        long best = 0;
        float bestVal = Float.NEGATIVE_INFINITY;
        for (int c = 0; c < vocabSize; c++) {
            float v = logits[c * n + idx];
            if (v > bestVal) { bestVal = v; best = c; }
        }
        return best;
    }
}
