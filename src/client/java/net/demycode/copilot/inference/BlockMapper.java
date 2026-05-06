package net.demycode.copilot.inference;

import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;

import java.io.IOException;
import java.io.Reader;
import java.lang.reflect.Type;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;

public class BlockMapper {

    private final Map<String, Integer> blockToIdx;
    private final Map<String, String> idxToBlock;
    private final int vocabSize;
    private final int maskIdx;
    private final int chunkSize;

    public BlockMapper(Path vocabPath, Path metaPath) throws IOException {
        Gson gson = new Gson();

        try (Reader r = Files.newBufferedReader(vocabPath)) {
            Type type = new TypeToken<Map<String, Object>>() {}.getType();
            Map<String, Object> root = gson.fromJson(r, type);
            Type strMap = new TypeToken<Map<String, String>>() {}.getType();
            Type intMap = new TypeToken<Map<String, Integer>>() {}.getType();
            idxToBlock = gson.fromJson(gson.toJson(root.get("idx_to_block")), strMap);
            blockToIdx = gson.fromJson(gson.toJson(root.get("block_to_idx")), intMap);
        }

        try (Reader r = Files.newBufferedReader(metaPath)) {
            Type type = new TypeToken<Map<String, Number>>() {}.getType();
            Map<String, Number> meta = gson.fromJson(r, type);
            vocabSize = meta.get("vocab_size").intValue();
            maskIdx = meta.get("mask_idx").intValue();
            chunkSize = meta.get("chunk_size").intValue();
        }
    }

    public int toIdx(String blockName) { return blockToIdx.getOrDefault(blockName, 0); }
    public String toName(int idx) { return idxToBlock.getOrDefault(String.valueOf(idx), "minecraft:air"); }
    public int getVocabSize() { return vocabSize; }
    public int getMaskIdx()   { return maskIdx; }
    public int getChunkSize() { return chunkSize; }
}
