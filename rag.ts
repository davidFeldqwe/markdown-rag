import { readdir, readFile, stat } from "node:fs/promises";
import { relative, resolve, sep } from "node:path";
import { openai } from "@ai-sdk/openai";
import { cosineSimilarity, embed, embedMany, generateText } from "ai";
import { z } from "zod";

/** Chunks longer than this are split on paragraph boundaries, then on words. */
export const MAX_CHUNK_CHARS = 700;

/** How many retrieved chunks are passed to the model. */
export const TOP_K = 4;

/**
 * Cosine similarity floor for text-embedding-3-small.
 * Chunks under this are treated as not retrieved, and the question is refused
 * without calling the chat model.
 */
export const MIN_SIMILARITY = 0.42;

export const EMBEDDING_MODEL_ID = "text-embedding-3-small";
export const CHAT_MODEL_ID = "gpt-4o-mini";

const storedChunkSchema = z.object({
  file: z.string().min(1),
  index: z.number().int().nonnegative(),
  text: z.string().min(1),
  embedding: z.array(z.number()).min(1),
});

const storeSchema = z.object({
  model: z.string().min(1),
  chunks: z.array(storedChunkSchema),
});

export type StoredChunk = z.infer<typeof storedChunkSchema>;
export type Store = z.infer<typeof storeSchema>;

export type Chunk = {
  file: string;
  index: number;
  text: string;
};

export type RankedChunk = Chunk & {
  similarity: number;
};

export type Embedder = {
  /** Label stored next to the vectors so the file says what produced them. */
  model: string;
  embed(value: string): Promise<number[]>;
  embedMany(values: string[]): Promise<number[][]>;
};

export type CompletionInput = {
  instructions: string;
  prompt: string;
};

export function chunkMarkdown(file: string, markdown: string): Chunk[] {
  const normalized = markdown.replace(/\r\n/g, "\n").trim();
  if (!normalized) return [];

  const pieces: string[] = [];
  for (const section of splitSections(normalized)) {
    pieces.push(...packSection(section));
  }

  return pieces.map((text, index) => ({ file, index, text }));
}

function splitSections(markdown: string): string[] {
  const sections: string[] = [];
  let current: string[] = [];

  for (const line of markdown.split("\n")) {
    const startsHeading = /^#{1,6} /.test(line);
    if (startsHeading && current.some((entry) => entry.trim())) {
      sections.push(current.join("\n").trim());
      current = [];
    }
    current.push(line);
  }

  const tail = current.join("\n").trim();
  if (tail) sections.push(tail);
  return sections;
}

function packSection(section: string): string[] {
  if (section.length <= MAX_CHUNK_CHARS) return [section];

  const paragraphs = section
    .split(/\n{2,}/)
    .map((paragraph) => paragraph.trim())
    .filter(Boolean);
  const pieces: string[] = [];
  let current = "";

  const flush = () => {
    if (current) pieces.push(current);
    current = "";
  };

  for (const paragraph of paragraphs) {
    if (paragraph.length > MAX_CHUNK_CHARS) {
      flush();
      pieces.push(...splitLong(paragraph, MAX_CHUNK_CHARS));
      continue;
    }

    const next = current ? `${current}\n\n${paragraph}` : paragraph;
    if (next.length > MAX_CHUNK_CHARS) {
      flush();
      current = paragraph;
    } else {
      current = next;
    }
  }

  flush();
  return pieces;
}

function splitLong(text: string, max: number): string[] {
  const parts: string[] = [];
  let rest = text.trim();

  while (rest.length > max) {
    let cut = rest.lastIndexOf(" ", max);
    if (cut < max * 0.5) cut = max;
    const part = rest.slice(0, cut).trim();
    if (part) parts.push(part);
    rest = rest.slice(cut).trim();
  }

  if (rest) parts.push(rest);
  return parts;
}

export async function indexDirectory(dir: string, embedder: Embedder): Promise<Store> {
  const root = resolve(dir);
  const info = await stat(root);
  if (!info.isDirectory()) {
    throw new Error(`${dir} is not a directory`);
  }

  const files = await collectMarkdown(root);
  const chunks: Chunk[] = [];
  for (const file of files) {
    const markdown = await readFile(file, "utf8");
    const name = relative(root, file).split(sep).join("/");
    chunks.push(...chunkMarkdown(name, markdown));
  }

  if (chunks.length === 0) {
    return { model: embedder.model, chunks: [] };
  }

  const embeddings = await embedder.embedMany(chunks.map((chunk) => chunk.text));
  if (embeddings.length !== chunks.length) {
    throw new Error(
      `embedMany returned ${embeddings.length} vectors for ${chunks.length} chunks`,
    );
  }

  return {
    model: embedder.model,
    chunks: chunks.map((chunk, index) => {
      const embedding = embeddings[index];
      if (!embedding?.length) {
        throw new Error(`Missing embedding for ${chunk.file} #${chunk.index}`);
      }
      return { ...chunk, embedding };
    }),
  };
}

async function collectMarkdown(dir: string): Promise<string[]> {
  const entries = await readdir(dir, { withFileTypes: true });
  const files: string[] = [];

  for (const entry of entries) {
    const full = resolve(dir, entry.name);
    if (entry.isDirectory()) {
      files.push(...(await collectMarkdown(full)));
    } else if (entry.isFile() && entry.name.endsWith(".md")) {
      files.push(full);
    }
  }

  return files.sort();
}

export async function loadStore(path: string): Promise<Store> {
  let raw: string;
  try {
    raw = await readFile(path, "utf8");
  } catch (error) {
    if (isErrno(error) && error.code === "ENOENT") {
      throw new Error("No embeddings.json found. Run: npm run index -- ./docs");
    }
    throw error;
  }

  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch {
    throw new Error("embeddings.json is not valid JSON. Run the index again.");
  }

  return storeSchema.parse(parsed);
}

function isErrno(error: unknown): error is NodeJS.ErrnoException {
  return error instanceof Error && "code" in error;
}

/** Same shape as the guide: score every stored vector, then sort high to low. */
export function scoreChunks(queryEmbedding: number[], chunks: StoredChunk[]): RankedChunk[] {
  return chunks
    .map((chunk) => ({
      file: chunk.file,
      index: chunk.index,
      text: chunk.text,
      similarity: cosineSimilarity(chunk.embedding, queryEmbedding),
    }))
    .sort(
      (a, b) =>
        b.similarity - a.similarity ||
        a.file.localeCompare(b.file) ||
        a.index - b.index,
    );
}

export function takeRelevant(
  ranked: RankedChunk[],
  k: number,
  minSimilarity: number,
): RankedChunk[] {
  return ranked.filter((chunk) => chunk.similarity >= minSimilarity).slice(0, k);
}

export function formatCitation(chunk: RankedChunk): string {
  return `${chunk.file} #${chunk.index} (${chunk.similarity.toFixed(2)})`;
}

export function formatAnswer(answer: string, chunks: RankedChunk[]): string {
  const sources = chunks.map((chunk) => `- ${formatCitation(chunk)}`).join("\n");
  return `${answer.trim()}\n\nSources:\n${sources}`;
}

export function refusalMessage(best: number | null): string {
  if (best === null) {
    return "The index is empty, so I won't guess.";
  }
  return `Nothing in the indexed notes is relevant to that question (best match ${best.toFixed(2)}), so I won't guess.`;
}

export function buildPrompt(question: string, chunks: RankedChunk[]): CompletionInput {
  const notes = chunks
    .map((chunk) => `Source: ${chunk.file} #${chunk.index}\n${chunk.text}`)
    .join("\n\n---\n\n");

  return {
    instructions: [
      "Answer using only the notes in the user message.",
      "End every factual sentence with a citation like [file #index], copying the source label exactly.",
      "If the notes do not contain the answer, reply with exactly: I don't know based on these notes.",
      "Do not add facts from outside the notes.",
    ].join(" "),
    prompt: `Notes:\n${notes}\n\nQuestion: ${question}`,
  };
}

export async function answerQuestion(
  question: string,
  store: Store,
  embedder: Embedder,
  complete: (input: CompletionInput) => Promise<string>,
  options?: { k?: number; minSimilarity?: number },
): Promise<string> {
  const k = options?.k ?? TOP_K;
  const minSimilarity = options?.minSimilarity ?? MIN_SIMILARITY;

  if (store.chunks.length === 0) {
    return refusalMessage(null);
  }

  const queryEmbedding = await embedder.embed(question);
  const ranked = scoreChunks(queryEmbedding, store.chunks);
  const hits = takeRelevant(ranked, k, minSimilarity);
  if (hits.length === 0) {
    return refusalMessage(ranked[0]?.similarity ?? null);
  }

  const text = await complete(buildPrompt(question, hits));
  return formatAnswer(text, hits);
}

export function openAiEmbedder(): Embedder {
  const model = openai.embedding(EMBEDDING_MODEL_ID);
  return {
    model: EMBEDDING_MODEL_ID,
    async embed(value) {
      const result = await embed({ model, value });
      return result.embedding;
    },
    async embedMany(values) {
      const result = await embedMany({ model, values });
      return result.embeddings;
    },
  };
}

export async function completeWithOpenAI(input: CompletionInput): Promise<string> {
  const { text } = await generateText({
    model: openai(CHAT_MODEL_ID),
    instructions: input.instructions,
    prompt: input.prompt,
  });
  return text;
}
