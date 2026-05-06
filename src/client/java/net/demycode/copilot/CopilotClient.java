package net.demycode.copilot;

import net.demycode.copilot.inference.BlockMapper;
import net.demycode.copilot.inference.DiffusionSampler;
import net.demycode.copilot.inference.OnnxSession;
import net.demycode.copilot.render.SuggestionRenderer;
import net.fabricmc.api.ClientModInitializer;
import net.fabricmc.fabric.api.client.rendering.v1.level.LevelRenderEvents;
import net.fabricmc.loader.api.FabricLoader;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.file.Path;

public class CopilotClient implements ClientModInitializer {

    public static final Logger LOGGER = LoggerFactory.getLogger("minecraft_copilot");

    public static BlockMapper blockMapper;
    public static OnnxSession onnxSession;
    public static DiffusionSampler sampler;
    public static final SuggestionRenderer renderer = new SuggestionRenderer();

    @Override
    public void onInitializeClient() {
        LevelRenderEvents.COLLECT_SUBMITS.register(renderer);

        Path configDir = FabricLoader.getInstance().getConfigDir().resolve("minecraft-copilot");
        try {
            blockMapper = new BlockMapper(configDir.resolve("vocab.json"), configDir.resolve("meta.json"));
            onnxSession = new OnnxSession(configDir.resolve("model.onnx"));
            sampler = new DiffusionSampler(onnxSession, blockMapper);
            LOGGER.info("Minecraft Copilot loaded — vocab {}, chunk {}³",
                    blockMapper.getVocabSize(), blockMapper.getChunkSize());
        } catch (Exception e) {
            LOGGER.error("Copilot model not found in {}. Copy model.onnx, vocab.json, meta.json there.", configDir);
            LOGGER.error("Reason: {}", e.getMessage());
        }
    }
}
