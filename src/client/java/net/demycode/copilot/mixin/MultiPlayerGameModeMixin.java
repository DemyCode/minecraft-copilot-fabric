package net.demycode.copilot.mixin;

import net.demycode.copilot.CopilotClient;
import net.demycode.copilot.render.SuggestionRenderer;
import net.minecraft.client.multiplayer.MultiPlayerGameMode;
import net.minecraft.client.player.LocalPlayer;
import net.minecraft.core.BlockPos;
import net.minecraft.core.registries.BuiltInRegistries;
import net.minecraft.network.chat.Component;
import net.minecraft.resources.Identifier;
import net.minecraft.world.InteractionHand;
import net.minecraft.world.InteractionResult;
import net.minecraft.world.level.Level;
import net.minecraft.world.level.block.Blocks;
import net.minecraft.world.level.block.state.BlockState;
import net.minecraft.world.phys.BlockHitResult;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfoReturnable;

import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;

@Mixin(MultiPlayerGameMode.class)
public class MultiPlayerGameModeMixin {

    @Inject(method = "useItemOn", at = @At("RETURN"))
    private void onUseItemOn(LocalPlayer player, InteractionHand hand,
                              BlockHitResult hitResult,
                              CallbackInfoReturnable<InteractionResult> cir) {
        if (!cir.getReturnValue().consumesAction()) return;
        if (CopilotClient.sampler == null) {
            CopilotClient.LOGGER.info("[Copilot] Sampler is null, skipping inference");
            return;
        }

        BlockPos placedPos = hitResult.getBlockPos().relative(hitResult.getDirection());
        Level level = player.level();

        int cs = CopilotClient.blockMapper.getChunkSize();
        int half = cs / 2;
        BlockPos origin = placedPos.offset(-half, -half, -half);

        int n = cs * cs * cs;
        long[] blocks = new long[n];
        boolean[] condMask = new boolean[n];

        int unknownBlockCount = 0;
        for (int y = 0; y < cs; y++) {
            for (int z = 0; z < cs; z++) {
                for (int x = 0; x < cs; x++) {
                    int idx = y * cs * cs + z * cs + x;
                    var state = level.getBlockState(origin.offset(x, y, z));
                    String name = BuiltInRegistries.BLOCK.getKey(state.getBlock()).toString();
                    int blockIdx = CopilotClient.blockMapper.toIdx(name);
                    if (blockIdx == 0 && !state.isAir()) unknownBlockCount++;
                    blocks[idx] = blockIdx;
                    condMask[idx] = blockIdx != 0;
                }
            }
        }
        if (unknownBlockCount > 0)
            CopilotClient.LOGGER.warn("[Copilot] {} blocks not in vocab — treated as air (unconditioned)", unknownBlockCount);

        CopilotClient.LOGGER.info("[Copilot] Starting inference for {} blocks", n);
        player.sendSystemMessage(Component.literal("[Copilot] Running inference..."));

        // Move the bounding box to the new zone immediately, clearing old ghost blocks.
        final long[] blocksFinal = blocks;
        final boolean[] maskFinal = condMask;
        final BlockPos originFinal = origin;
        final int csFinal = cs;
        final int maskIdx = CopilotClient.blockMapper.getMaskIdx();

        CopilotClient.renderer.update(new SuggestionRenderer.SuggestionState(new HashMap<>(), originFinal, csFinal));

        AtomicBoolean loggedOnce = new AtomicBoolean(false);
        CopilotClient.sampler.submit(blocksFinal, maskFinal, stepBlocks -> {
            Map<BlockPos, BlockState> ghostMap = new HashMap<>();
            Map<String, Integer> blockFreq = loggedOnce.get() ? null : new HashMap<>();
            for (int y = 0; y < csFinal; y++) {
                for (int z = 0; z < csFinal; z++) {
                    for (int x = 0; x < csFinal; x++) {
                        int idx = y * csFinal * csFinal + z * csFinal + x;
                        if (maskFinal[idx]) continue;
                        long blockId = stepBlocks[idx];
                        if (blockId == 0 || blockId == maskIdx) continue;

                        String name = CopilotClient.blockMapper.toName((int) blockId);
                        if (blockFreq != null) blockFreq.merge(name, 1, Integer::sum);
                        BlockState state = BuiltInRegistries.BLOCK
                                .getOptional(Identifier.parse(name))
                                .orElse(Blocks.AIR)
                                .defaultBlockState();
                        if (state.isAir()) continue;

                        ghostMap.put(originFinal.offset(x, y, z), state);
                    }
                }
            }
            if (blockFreq != null && !blockFreq.isEmpty() && loggedOnce.compareAndSet(false, true)) {
                blockFreq.entrySet().stream()
                    .sorted(Map.Entry.<String, Integer>comparingByValue().reversed())
                    .limit(10)
                    .forEach(e -> CopilotClient.LOGGER.info("[Copilot] predicted block: {} x{}", e.getKey(), e.getValue()));
            }
            CopilotClient.renderer.update(new SuggestionRenderer.SuggestionState(ghostMap, originFinal, csFinal));
        });

        CopilotClient.LOGGER.info("[Copilot] Submit call completed");
    }
}
