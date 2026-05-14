package net.demycode.copilot;

import net.demycode.copilot.inference.BlockMapper;
import net.demycode.copilot.inference.DiffusionSampler;
import net.demycode.copilot.inference.InferMode;
import net.demycode.copilot.inference.OnnxSession;
import net.demycode.copilot.render.SuggestionRenderer;
import net.fabricmc.api.ClientModInitializer;
import net.fabricmc.fabric.api.client.command.v2.ClientCommandRegistrationCallback;
import net.fabricmc.fabric.api.client.command.v2.ClientCommands;
import net.fabricmc.fabric.api.client.rendering.v1.level.LevelRenderEvents;
import net.fabricmc.loader.api.FabricLoader;
import net.minecraft.network.chat.Component;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.file.Path;

public class CopilotClient implements ClientModInitializer {

    public static final Logger LOGGER = LoggerFactory.getLogger("minecraft_copilot");

    public static BlockMapper blockMapper;
    public static OnnxSession onnxSession;
    public static DiffusionSampler sampler;
    public static final SuggestionRenderer renderer = new SuggestionRenderer();
    public static volatile InferMode inferMode = InferMode.MASKGIT;

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

        ClientCommandRegistrationCallback.EVENT.register((dispatcher, registryAccess) ->
            dispatcher.register(
                ClientCommands.literal("copilot")
                    .then(ClientCommands.literal("infer")
                        .then(ClientCommands.literal("classic")
                            .executes(ctx -> {
                                inferMode = InferMode.CLASSIC;
                                ctx.getSource().sendFeedback(
                                    Component.literal("[Copilot] Mode: CLASSIC (single pass, instant)"));
                                return 1;
                            }))
                        .then(ClientCommands.literal("maskgit")
                            .executes(ctx -> {
                                inferMode = InferMode.MASKGIT;
                                ctx.getSource().sendFeedback(
                                    Component.literal("[Copilot] Mode: MASKGIT (iterative refinement)"));
                                return 1;
                            }))
                        .then(ClientCommands.literal("sparse")
                            .executes(ctx -> {
                                inferMode = InferMode.SPARSE;
                                ctx.getSource().sendFeedback(
                                    Component.literal("[Copilot] Mode: SPARSE (MASKGIT, confidence-gated)"));
                                return 1;
                            })))));
    }
}
