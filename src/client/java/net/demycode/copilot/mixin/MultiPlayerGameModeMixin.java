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

        for (int y = 0; y < cs; y++) {
            for (int z = 0; z < cs; z++) {
                for (int x = 0; x < cs; x++) {
                    int idx = y * cs * cs + z * cs + x;
                    var state = level.getBlockState(origin.offset(x, y, z));
                    String name = BuiltInRegistries.BLOCK.getKey(state.getBlock()).toString();
                    int blockIdx = CopilotClient.blockMapper.toIdx(name);
                    blocks[idx] = blockIdx;
                    condMask[idx] = blockIdx != 0;
                }
            }
        }

        CopilotClient.LOGGER.info("[Copilot] Starting inference for {} blocks", n);
        player.sendSystemMessage(Component.literal("[Copilot] Running inference..."));

        final long[] blocksFinal = blocks;
        final boolean[] maskFinal = condMask;
        final BlockPos originFinal = origin;
        final int maskIdx = CopilotClient.blockMapper.getMaskIdx();

        CopilotClient.sampler.submit(blocksFinal, maskFinal, stepBlocks -> {
            Map<BlockPos, BlockState> ghostMap = new HashMap<>();
            for (int y = 0; y < cs; y++) {
                for (int z = 0; z < cs; z++) {
                    for (int x = 0; x < cs; x++) {
                        int idx = y * cs * cs + z * cs + x;
                        if (maskFinal[idx]) continue;
                        long blockId = stepBlocks[idx];
                        if (blockId == 0 || blockId == maskIdx) continue;

                        String name = CopilotClient.blockMapper.toName((int) blockId);
                        BlockState state = BuiltInRegistries.BLOCK
                                .getOptional(Identifier.parse(name))
                                .orElse(Blocks.AIR)
                                .defaultBlockState();
                        if (state.isAir()) continue;

                        ghostMap.put(originFinal.offset(x, y, z), state);
                    }
                }
            }
            CopilotClient.LOGGER.info("[Copilot] Step callback: {} ghost blocks", ghostMap.size());
            CopilotClient.renderer.update(new SuggestionRenderer.SuggestionState(ghostMap));
        });

        CopilotClient.LOGGER.info("[Copilot] Submit call completed");
    }
}
