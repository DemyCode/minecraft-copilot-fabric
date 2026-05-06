package net.demycode.copilot.render;

import com.mojang.blaze3d.vertex.PoseStack;
import com.mojang.blaze3d.vertex.VertexConsumer;
import net.demycode.copilot.CopilotClient;
import net.fabricmc.fabric.api.client.rendering.v1.level.LevelRenderContext;
import net.fabricmc.fabric.api.client.rendering.v1.level.LevelRenderEvents;
import net.minecraft.client.Minecraft;
import net.minecraft.client.renderer.MultiBufferSource;
import net.minecraft.client.renderer.rendertype.RenderTypes;
import net.minecraft.core.BlockPos;
import net.minecraft.world.phys.Vec3;
import org.joml.Matrix4f;

import java.util.concurrent.atomic.AtomicReference;

public class SuggestionRenderer implements LevelRenderEvents.AfterTranslucentTerrain {

    public record SuggestionState(long[] blocks, boolean[] condMask, BlockPos origin, int chunkSize, int maskIdx) {}

    private final AtomicReference<SuggestionState> current = new AtomicReference<>();

    public void update(SuggestionState state) {
        CopilotClient.LOGGER.info("[Copilot] SuggestionRenderer.update called with state: {} blocks", state == null ? 0 : state.blocks().length);
        current.set(state);
    }
    public void clear() {
        CopilotClient.LOGGER.info("[Copilot] SuggestionRenderer.clear called");
        current.set(null);
    }

    @Override
    public void afterTranslucentTerrain(LevelRenderContext ctx) {
        SuggestionState state = current.get();
        if (state == null) {
            CopilotClient.LOGGER.info("[Copilot] SuggestionRenderer.afterTranslucentTerrain: state is null");
            return;
        }

        CopilotClient.LOGGER.info("[Copilot] SuggestionRenderer.afterTranslucentTerrain called with {} blocks", state.blocks().length);
        Vec3 cam = Minecraft.getInstance().gameRenderer.getMainCamera().position();
        PoseStack ps = ctx.poseStack();
        MultiBufferSource buffers = ctx.bufferSource();
        VertexConsumer vc = buffers.getBuffer(RenderTypes.LINES);

        int cs = state.chunkSize();
        long[] blocks = state.blocks();
        boolean[] condMask = state.condMask();
        BlockPos origin = state.origin();
        long maskIdx = state.maskIdx();

        ps.pushPose();
        ps.translate(origin.getX() - cam.x, origin.getY() - cam.y, origin.getZ() - cam.z);

        int drawn = 0;
        for (int y = 0; y < cs; y++) {
            for (int z = 0; z < cs; z++) {
                for (int x = 0; x < cs; x++) {
                    int idx = y * cs * cs + z * cs + x;
                    long blockId = blocks[idx];
                    if (blockId == 0 || blockId == maskIdx || condMask[idx]) continue;
                    float[] color = blockColor(blockId);
                    drawBox(ps, vc, x, y, z, color[0], color[1], color[2], 0.6f);
                    drawn++;
                }
            }
        }
        if (drawn > 0) {
            net.demycode.copilot.CopilotClient.LOGGER.info("[Copilot] Rendering {} suggestion boxes", drawn);
        } else {
            net.demycode.copilot.CopilotClient.LOGGER.info("[Copilot] No suggestion boxes to render (drawn=0)");
        }

        ps.popPose();
    }

    private void drawBox(PoseStack ps, VertexConsumer vc,
                         int x, int y, int z, float r, float g, float b, float a) {
        Matrix4f m = ps.last().pose();
        float x0 = x, y0 = y, z0 = z, x1 = x + 1, y1 = y + 1, z1 = z + 1;
        int[][] edges = {
            {0,0,0, 1,0,0}, {1,0,0, 1,1,0}, {1,1,0, 0,1,0}, {0,1,0, 0,0,0},
            {0,0,1, 1,0,1}, {1,0,1, 1,1,1}, {1,1,1, 0,1,1}, {0,1,1, 0,0,1},
            {0,0,0, 0,0,1}, {1,0,0, 1,0,1}, {1,1,0, 1,1,1}, {0,1,0, 0,1,1}
        };
        float[] xs = {x0, x1}, ys = {y0, y1}, zs = {z0, z1};
        for (int[] e : edges) {
            vc.addVertex(m, xs[e[0]], ys[e[1]], zs[e[2]]).setColor(r, g, b, a).setNormal(0, 1, 0).setLineWidth(2f);
            vc.addVertex(m, xs[e[3]], ys[e[4]], zs[e[5]]).setColor(r, g, b, a).setNormal(0, 1, 0).setLineWidth(2f);
        }
    }

    private float[] blockColor(long blockId) {
        int h = Long.hashCode(blockId);
        float r = ((h >> 16) & 0xFF) / 255f;
        float g = ((h >> 8)  & 0xFF) / 255f;
        float b = ( h        & 0xFF) / 255f;
        float max = Math.max(r, Math.max(g, b));
        if (max < 0.4f) { r += 0.4f; g += 0.4f; b += 0.4f; }
        return new float[]{Math.min(r, 1), Math.min(g, 1), Math.min(b, 1)};
    }
}
