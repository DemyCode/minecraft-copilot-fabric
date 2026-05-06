package net.demycode.copilot.render;

import com.mojang.blaze3d.vertex.PoseStack;
import net.demycode.copilot.CopilotClient;
import net.fabricmc.fabric.api.client.rendering.v1.level.LevelRenderContext;
import net.fabricmc.fabric.api.client.rendering.v1.level.LevelRenderEvents;
import net.minecraft.client.Minecraft;
import net.minecraft.client.multiplayer.ClientLevel;
import net.minecraft.client.renderer.SubmitNodeCollector;
import net.minecraft.client.renderer.block.MovingBlockRenderState;
import net.minecraft.core.BlockPos;
import net.minecraft.world.level.block.state.BlockState;
import net.minecraft.world.phys.Vec3;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicReference;

public class SuggestionRenderer implements LevelRenderEvents.CollectSubmits {

    public record SuggestionState(Map<BlockPos, BlockState> ghostBlocks) {}

    private final AtomicReference<SuggestionState> current = new AtomicReference<>();

    // submitMovingBlock stores a reference, not a copy — each block needs its own object.
    // Rebuilt once per suggestion update, reused every frame until next update.
    private List<MovingBlockRenderState> cachedStates = List.of();
    private SuggestionState lastBuilt = null;

    public void update(SuggestionState state) {
        CopilotClient.LOGGER.info("[Copilot] SuggestionRenderer.update: {} ghost blocks",
                state == null ? 0 : state.ghostBlocks().size());
        current.set(state);
    }

    public void clear() {
        current.set(null);
    }

    @Override
    public void collectSubmits(LevelRenderContext ctx) {
        SuggestionState state = current.get();
        if (state == null || state.ghostBlocks().isEmpty()) {
            cachedStates = List.of();
            lastBuilt = null;
            return;
        }

        ClientLevel level = Minecraft.getInstance().level;
        if (level == null) return;

        // Rebuild one MovingBlockRenderState per ghost block whenever the suggestion changes.
        if (state != lastBuilt) {
            lastBuilt = state;
            var cardinalLighting = level.cardinalLighting();
            var lightEngine = level.getLightEngine();

            List<MovingBlockRenderState> newStates = new ArrayList<>(state.ghostBlocks().size());
            for (Map.Entry<BlockPos, BlockState> entry : state.ghostBlocks().entrySet()) {
                BlockPos pos = entry.getKey();
                BlockState blockState = entry.getValue();
                if (blockState.isAir()) continue;

                MovingBlockRenderState mrs = new MovingBlockRenderState();
                mrs.blockPos = pos;
                mrs.randomSeedPos = pos;
                mrs.blockState = blockState;
                mrs.biome = level.getBiome(pos);
                mrs.cardinalLighting = cardinalLighting;
                mrs.lightEngine = lightEngine;
                newStates.add(mrs);
            }
            cachedStates = newStates;
            CopilotClient.LOGGER.info("[Copilot] SuggestionRenderer: rebuilt {} render states", cachedStates.size());
        }

        PoseStack ps = ctx.poseStack();
        SubmitNodeCollector collector = ctx.submitNodeCollector();
        Vec3 cam = Minecraft.getInstance().gameRenderer.getMainCamera().position();

        ps.pushPose();
        ps.translate(-cam.x, -cam.y, -cam.z);

        int drawn = 0;
        for (MovingBlockRenderState mrs : cachedStates) {
            BlockPos pos = mrs.blockPos;
            ps.pushPose();
            ps.translate(pos.getX(), pos.getY(), pos.getZ());
            collector.submitMovingBlock(ps, mrs);
            ps.popPose();
            drawn++;
        }

        ps.popPose();
        CopilotClient.LOGGER.info("[Copilot] SuggestionRenderer: submitted {} ghost blocks", drawn);
    }
}
