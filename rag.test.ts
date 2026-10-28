import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { cosineSimilarity } from "ai";
import { afterEach, describe, expect, it, vi } from "vitest";
import {
  MAX_CHUNK_CHARS,
  answerQuestion,
  buildPrompt,
  chunkMarkdown,
  formatAnswer,
  formatCitation,
  indexDirectory,
  loadStore,
  refusalMessage,
  scoreChunks,
  takeRelevant,
  type CompletionInput,
  type Embedder,
  type RankedChunk,
  type Store,
} from "./rag.ts";

const tempDirs: string[] = [];

afterEach(async () => {
  await Promise.all(tempDirs.splice(0).map((dir) => rm(dir, { recursive: true, force: true })));
});

function deterministicEmbedding(text: string, dimensions = 48): number[] {
  const vector = new Array<number>(dimensions).fill(0);
  const tokens = text.toLowerCase().match(/[a-z0-9]+/g) ?? [];
  for (const token of tokens) {
    let hash = 2166136261;
    for (let i = 0; i < token.length; i++) {
      hash ^= token.charCodeAt(i);
      hash = Math.imul(hash, 16777619);
    }
    const bucket = Math.abs(hash) % dimensions;
    vector[bucket] += hash % 2 === 0 ? 1 : -1;
  }
  const norm = Math.hypot(...vector);
  if (norm === 0) return vector;
  return vector.map((value) => value / norm);
}

function deterministicEmbedder(): Embedder {
  return {
    model: "deterministic-hash",
    async embed(value) {
      return deterministicEmbedding(value);
    },
    async embedMany(values) {
      return values.map((value) => deterministicEmbedding(value));
    },
  };
}

async function storeFrom(
  chunks: Array<{ file: string; index: number; text: string }>,
): Promise<Store> {
  const embedder = deterministicEmbedder();
  const embeddings = await embedder.embedMany(chunks.map((chunk) => chunk.text));
  return {
    model: embedder.model,
    chunks: chunks.map((chunk, index) => ({
      ...chunk,
      embedding: embeddings[index] ?? [],
    })),
  };
}

describe("chunkMarkdown", () => {
  it("splits on headings and numbers chunks per file", () => {
    const chunks = chunkMarkdown(
      "guide.md",
      "# Alpha\r\n\r\nAlpha body.\r\n\r\n# Beta\r\n\r\nBeta body.\n",
    );

    expect(chunks.map((chunk) => chunk.index)).toEqual([0, 1]);
    expect(chunks[0]).toMatchObject({ file: "guide.md", text: "# Alpha\n\nAlpha body." });
    expect(chunks[1]?.text.startsWith("# Beta")).toBe(true);
  });

  it("packs a long section into pieces that stay within the character cap", () => {
    const body = Array.from({ length: 40 }, (_, index) => `Paragraph ${index} about export.`).join(
      "\n\n",
    );
    const chunks = chunkMarkdown("export.md", `# Export\n\n${body}`);

    expect(chunks.length).toBeGreaterThan(1);
    expect(chunks.every((chunk) => chunk.text.length <= MAX_CHUNK_CHARS)).toBe(true);
    expect(chunks[0]?.text.startsWith("# Export")).toBe(true);
    expect(chunks.at(-1)?.text).toContain("Paragraph 39");
  });

  it("returns nothing for an empty file", () => {
    expect(chunkMarkdown("empty.md", " \n\n ")).toEqual([]);
  });
});

describe("ranking", () => {
  it("orders chunks by cosine similarity and drops those under the floor", () => {
    const chunks = [
      { file: "a.md", index: 0, text: "same direction", embedding: [1, 0] },
      { file: "b.md", index: 0, text: "orthogonal", embedding: [0, 1] },
      { file: "c.md", index: 0, text: "nearby", embedding: [0.6, 0.8] },
    ];

    const ranked = scoreChunks([1, 0], chunks);
    expect(ranked.map((chunk) => chunk.file)).toEqual(["a.md", "c.md", "b.md"]);
    expect(ranked[0]?.similarity).toBeCloseTo(1);
    expect(ranked[2]?.similarity).toBeCloseTo(0);

    expect(takeRelevant(ranked, 4, 0.5).map((chunk) => chunk.file)).toEqual(["a.md", "c.md"]);
  });

  it("ranks a hashed chunk that shares the question's words first", async () => {
    const store = await storeFrom([
      {
        file: "sync.md",
        index: 0,
        text: "Lumen syncs notebooks through an encrypted relay called Harbor.",
      },
      {
        file: "shortcuts.md",
        index: 0,
        text: "Press Command-K to open the command palette.",
      },
    ]);
    const embedder = deterministicEmbedder();
    const query = await embedder.embed("How does Lumen sync notebooks through Harbor?");
    const ranked = scoreChunks(query, store.chunks);

    expect(ranked[0]?.file).toBe("sync.md");
    expect(ranked[0]?.similarity).toBeGreaterThan(ranked[1]?.similarity ?? 0);

    const related = deterministicEmbedding("Lumen syncs notebooks through Harbor");
    const unrelated = deterministicEmbedding("photosynthesis chlorophyll stomata");
    expect(cosineSimilarity(query, related)).toBeGreaterThan(cosineSimilarity(query, unrelated));
  });
});

describe("refusal", () => {
  it("does not call the model when every chunk is below the floor", async () => {
    const store = await storeFrom([
      {
        file: "pricing.md",
        index: 0,
        text: "Harbor sync costs four dollars a month.",
      },
    ]);
    const complete = vi.fn(async () => "should not run");
    const result = await answerQuestion(
      "photosynthesis chlorophyll stomata",
      store,
      deterministicEmbedder(),
      complete,
      { minSimilarity: 0.99 },
    );

    expect(complete).not.toHaveBeenCalled();
    expect(result).toMatch(
      /^Nothing in the indexed notes is relevant to that question \(best match 0\.\d{2}\), so I won't guess\.$/,
    );
    expect(result).toBe(refusalMessage(Number(result.match(/0\.\d{2}/)?.[0])));
    expect(result).not.toContain("Sources:");
  });

  it("refuses an empty index without embedding a question", async () => {
    const embed = vi.fn(async () => [1, 0]);
    const embedder: Embedder = {
      model: "unused",
      embed,
      async embedMany() {
        return [];
      },
    };
    const result = await answerQuestion(
      "anything",
      { model: "unused", chunks: [] },
      embedder,
      async () => "nope",
    );

    expect(embed).not.toHaveBeenCalled();
    expect(result).toBe("The index is empty, so I won't guess.");
  });
});

describe("citations", () => {
  it("formats file name, chunk index, and similarity", () => {
    const chunk: RankedChunk = {
      file: "sync.md",
      index: 1,
      text: "Harbor stores ciphertext.",
      similarity: 0.81234,
    };

    expect(formatCitation(chunk)).toBe("sync.md #1 (0.81)");
    expect(formatAnswer("Lumen keeps both versions.", [chunk])).toBe(
      "Lumen keeps both versions.\n\nSources:\n- sync.md #1 (0.81)",
    );
  });

  it("sends only the retrieved chunks to the model and appends their citations", async () => {
    const store: Store = {
      model: "hand-built",
      chunks: [
        { file: "sync.md", index: 0, text: "Harbor stores ciphertext.", embedding: [1, 0] },
        { file: "pricing.md", index: 2, text: "Sync costs four dollars.", embedding: [0, 1] },
      ],
    };
    const embedder: Embedder = {
      model: "hand-built",
      async embed() {
        return [1, 0];
      },
      async embedMany() {
        return [];
      },
    };
    const complete = vi.fn(async (_input: CompletionInput) => {
      return "Harbor stores ciphertext. [sync.md #0]";
    });

    const result = await answerQuestion("Where is the ciphertext?", store, embedder, complete, {
      k: 1,
      minSimilarity: 0.5,
    });

    expect(complete).toHaveBeenCalledOnce();
    const input = complete.mock.calls[0]?.[0];
    expect(input?.prompt).toContain("sync.md #0");
    expect(input?.prompt).toContain("Harbor stores ciphertext.");
    expect(input?.prompt).not.toContain("four dollars");
    const labeled = buildPrompt("Where is the ciphertext?", [
      {
        file: "sync.md",
        index: 0,
        text: "Harbor stores ciphertext.",
        similarity: 1,
      },
    ]);
    expect(labeled.prompt).toContain("Source: sync.md #0\nHarbor stores ciphertext.");
    expect(labeled.prompt).not.toContain("four dollars");
    expect(result).toBe(
      "Harbor stores ciphertext. [sync.md #0]\n\nSources:\n- sync.md #0 (1.00)",
    );
  });
});

describe("indexDirectory", () => {
  it("embeds every chunk with the provided model and reloads the store", async () => {
    const dir = await mkdtemp(join(tmpdir(), "markdown-rag-"));
    tempDirs.push(dir);
    await writeFile(
      join(dir, "notes.md"),
      "# One\n\nFirst note.\n\n# Two\n\nSecond note.\n",
    );
    const nested = join(dir, "nested");
    await mkdir(nested);
    await writeFile(join(nested, "more.md"), "# Nested\n\nNested note.\n");

    const store = await indexDirectory(dir, deterministicEmbedder());
    expect(store.model).toBe("deterministic-hash");
    expect(store.chunks.map((chunk) => `${chunk.file} #${chunk.index}`)).toEqual([
      "nested/more.md #0",
      "notes.md #0",
      "notes.md #1",
    ]);
    expect(store.chunks.every((chunk) => chunk.embedding.length === 48)).toBe(true);

    const path = join(dir, "embeddings.json");
    await writeFile(path, JSON.stringify(store));
    await expect(loadStore(path)).resolves.toEqual(store);
  });
});
