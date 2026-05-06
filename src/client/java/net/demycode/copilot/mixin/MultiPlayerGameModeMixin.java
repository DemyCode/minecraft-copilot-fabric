package net.demycode.copilot.mixin;

import net.demycode.copilot.CopilotClient;
import net.demycode.copilot.render.SuggestionRenderer;
import net.minecraft.client.multiplayer.MultiPlayerGameMode;
import net.minecraft.client.player.LocalPlayer;
import net.minecraft.core.BlockPos;
import net.minecraft.core.registries.BuiltInRegistries;
import net.minecraft.network.chat.Component;
import net.minecraft.world.InteractionHand;
import net.minecraft.world.InteractionResult;
import net.minecraft.world.level.Level;
import net.minecraft.world.phys.BlockHitResult;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfoReturnable;

@Mixin(MultiPlayerGameMode.class)
public class MultiPlayerGameModeMixin {

    @Inject(method = "useItemOn", at = @At("RETURN"))
    private void onUseItemOn(LocalPlayer player, InteractionHand hand,
                             BlockHitResult hitResult,
                             CallbackInfoReturnable<InteractionResult> cir) {
        if (!cir.getReturnValue().consumesAction()) return;
        if (CopilotClient.sampler == null) return;

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

        player.sendSystemMessage(Component.literal("[Copilot] Running inference..."));

        final long[] blocksFinal = blocks;
        final boolean[] maskFinal = condMask;
        final BlockPos originFinal = origin;

        int maskIdx = CopilotClient.blockMapper.getMaskIdx();
        CopilotClient.sampler.submit(blocksFinal, maskFinal, stepBlocks -> {
            var state = new SuggestionRenderer.SuggestionState(stepBlocks, maskFinal, originFinal, cs, maskIdx);
            CopilotClient.renderer.update(state);
        });
    }
}
