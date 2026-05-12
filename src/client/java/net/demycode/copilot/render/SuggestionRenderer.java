package net.demycode.copilot.render;

import com.mojang.blaze3d.vertex.PoseStack;
import com.mojang.blaze3d.vertex.QuadInstance;
import com.mojang.blaze3d.vertex.VertexConsumer;
import net.demycode.copilot.CopilotClient;
import net.fabricmc.fabric.api.client.rendering.v1.level.LevelRenderContext;
import net.fabricmc.fabric.api.client.rendering.v1.level.LevelRenderEvents;
import net.minecraft.client.Minecraft;
import net.minecraft.client.renderer.SubmitNodeCollector;
import net.minecraft.client.renderer.block.dispatch.BlockStateModelPart;
import net.minecraft.client.renderer.rendertype.RenderTypes;
import net.minecraft.client.resources.model.geometry.BakedQuad;
import net.minecraft.core.BlockPos;
import net.minecraft.core.Direction;
import net.minecraft.util.RandomSource;
import net.minecraft.world.level.block.state.BlockState;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicReference;

public class SuggestionRenderer implements LevelRenderEvents.CollectSubmits {

    public record SuggestionState(Map<BlockPos, BlockState> ghostBlocks, BlockPos chunkOrigin, int chunkSize) {}

    private static final int FULL_BRIGHT = (15 << 20) | (15 << 4);
    // Alpha: 0.9 at center, 0.1 at edge (Chebyshev distance falloff)
    private static final int ALPHA_CENTER = Math.round(0.9f * 255);
    private static final int ALPHA_EDGE   = Math.round(0.1f * 255);

    private final AtomicReference<SuggestionState> current = new AtomicReference<>();
    private final RandomSource random = RandomSource.create(0L);

    public SuggestionRenderer() {}

    public void update(SuggestionState state) {
        CopilotClient.LOGGER.debug("[Copilot] SuggestionRenderer.update: {} ghost blocks",
                state == null ? 0 : state.ghostBlocks().size());
        current.set(state);
    }

    public void clear() {
        current.set(null);
    }

    @Override
    public void collectSubmits(LevelRenderContext ctx) {
        SuggestionState state = current.get();
        if (state == null) return;
        boolean hasBlocks = !state.ghostBlocks().isEmpty();
        boolean hasBox = state.chunkOrigin() != null;
        if (!hasBlocks && !hasBox) return;
        if (Minecraft.getInstance().level == null) return;

        var cam = ctx.levelState().cameraRenderState.pos;
        PoseStack ps = ctx.poseStack();
        SubmitNodeCollector collector = ctx.submitNodeCollector();
        var blockModelSet = Minecraft.getInstance().getModelManager().getBlockStateModelSet();

        ps.pushPose();
        ps.translate(-cam.x, -cam.y, -cam.z);

        int drawn = 0;
        for (Map.Entry<BlockPos, BlockState> entry : state.ghostBlocks().entrySet()) {
            BlockState bs = entry.getValue();
            if (bs.isAir()) continue;
            BlockPos pos = entry.getKey();

            List<BlockStateModelPart> parts = new ArrayList<>();
            random.setSeed(pos.asLong());
            blockModelSet.get(bs).collectParts(random, parts);
            if (parts.isEmpty()) continue;

            QuadInstance qi = new QuadInstance();
            qi.setColor((blockAlpha(pos, state.chunkOrigin(), state.chunkSize()) << 24) | 0x00FFFFFF);
            qi.setLightCoords(FULL_BRIGHT);

            ps.pushPose();
            ps.translate(pos.getX(), pos.getY(), pos.getZ());

            collector.submitCustomGeometry(ps, RenderTypes.translucentMovingBlock(), (pose, vc) -> {
                for (BlockStateModelPart part : parts) {
                    for (Direction dir : Direction.values()) {
                        for (BakedQuad quad : part.getQuads(dir)) {
                            vc.putBakedQuad(pose, quad, qi);
                        }
                    }
                    for (BakedQuad quad : part.getQuads(null)) {
                        vc.putBakedQuad(pose, quad, qi);
                    }
                }
            });

            ps.popPose();
            drawn++;
        }

        // Bounding box outline
        if (state.chunkOrigin() != null) {
            int cs = state.chunkSize();
            BlockPos o = state.chunkOrigin();
            float x0 = o.getX(), y0 = o.getY(), z0 = o.getZ();
            float x1 = x0 + cs, y1 = y0 + cs, z1 = z0 + cs;
            collector.submitCustomGeometry(ps, RenderTypes.LINES, (pose, vc) ->
                emitBox(vc, pose, x0, y0, z0, x1, y1, z1));
        }

        ps.popPose();
        CopilotClient.LOGGER.debug("[Copilot] SuggestionRenderer: submitted {} ghost blocks", drawn);
    }

    private static int blockAlpha(BlockPos pos, BlockPos origin, int cs) {
        float half = cs / 2.0f;
        float dx = Math.abs((pos.getX() - origin.getX()) + 0.5f - half);
        float dy = Math.abs((pos.getY() - origin.getY()) + 0.5f - half);
        float dz = Math.abs((pos.getZ() - origin.getZ()) + 0.5f - half);
        float t = Math.min(Math.max(dx, Math.max(dy, dz)) / half, 1.0f);
        return Math.round(ALPHA_CENTER + (ALPHA_EDGE - ALPHA_CENTER) * t);
    }

    private static void emitBox(VertexConsumer vc, PoseStack.Pose pose,
                                float x0, float y0, float z0,
                                float x1, float y1, float z1) {
        // bottom
        emitLine(vc, pose, x0,y0,z0, x1,y0,z0);
        emitLine(vc, pose, x1,y0,z0, x1,y0,z1);
        emitLine(vc, pose, x1,y0,z1, x0,y0,z1);
        emitLine(vc, pose, x0,y0,z1, x0,y0,z0);
        // top
        emitLine(vc, pose, x0,y1,z0, x1,y1,z0);
        emitLine(vc, pose, x1,y1,z0, x1,y1,z1);
        emitLine(vc, pose, x1,y1,z1, x0,y1,z1);
        emitLine(vc, pose, x0,y1,z1, x0,y1,z0);
        // verticals
        emitLine(vc, pose, x0,y0,z0, x0,y1,z0);
        emitLine(vc, pose, x1,y0,z0, x1,y1,z0);
        emitLine(vc, pose, x1,y0,z1, x1,y1,z1);
        emitLine(vc, pose, x0,y0,z1, x0,y1,z1);
    }

    private static void emitLine(VertexConsumer vc, PoseStack.Pose pose,
                                 float x0, float y0, float z0,
                                 float x1, float y1, float z1) {
        float dx = x1 - x0, dy = y1 - y0, dz = z1 - z0;
        float len = (float) Math.sqrt(dx * dx + dy * dy + dz * dz);
        if (len < 1e-6f) return;
        dx /= len; dy /= len; dz /= len;
        vc.addVertex(pose, x0, y0, z0).setColor(255, 255, 0, 255).setNormal(pose, dx, dy, dz).setLineWidth(3.0f);
        vc.addVertex(pose, x1, y1, z1).setColor(255, 255, 0, 255).setNormal(pose, dx, dy, dz).setLineWidth(3.0f);
    }

    public void close() {}
}
