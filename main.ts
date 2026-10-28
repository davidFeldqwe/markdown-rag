import { existsSync, readFileSync } from "node:fs";
import { writeFile } from "node:fs/promises";
import { join } from "node:path";
import {
  answerQuestion,
  completeWithOpenAI,
  indexDirectory,
  loadStore,
  openAiEmbedder,
} from "./rag.ts";

const STORE_PATH = join(import.meta.dirname, "embeddings.json");

loadLocalEnv();

try {
  await run(process.argv.slice(2));
} catch (error) {
  const message = error instanceof Error ? error.message : String(error);
  console.error(message);
  process.exit(1);
}

async function run(argv: string[]): Promise<void> {
  const [command, ...rest] = argv;

  if (command === "index") {
    const dir = rest[0];
    if (!dir) {
      throw new Error("Usage: npm run index -- ./docs");
    }
    const store = await indexDirectory(dir, openAiEmbedder());
    await writeFile(STORE_PATH, `${JSON.stringify(store)}\n`);
    console.log(
      `Indexed ${store.chunks.length} chunks from ${dir} into embeddings.json`,
    );
    return;
  }

  if (command === "ask") {
    const question = rest.join(" ").trim();
    if (!question) {
      throw new Error('Usage: npm run ask -- "question"');
    }
    const store = await loadStore(STORE_PATH);
    const answer = await answerQuestion(
      question,
      store,
      openAiEmbedder(),
      completeWithOpenAI,
    );
    console.log(answer);
    return;
  }

  throw new Error('Usage: npm run index -- ./docs   |   npm run ask -- "question"');
}

/** Reads `.env` when present. Existing process env wins, so a shell export is kept. */
function loadLocalEnv(): void {
  const path = join(import.meta.dirname, ".env");
  if (!existsSync(path)) return;

  for (const line of readFileSync(path, "utf8").split("\n")) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith("#")) continue;
    const eq = trimmed.indexOf("=");
    if (eq <= 0) continue;
    const key = trimmed.slice(0, eq).trim();
    if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(key)) continue;
    if (process.env[key] !== undefined) continue;
    let value = trimmed.slice(eq + 1).trim();
    if (
      (value.startsWith('"') && value.endsWith('"')) ||
      (value.startsWith("'") && value.endsWith("'"))
    ) {
      value = value.slice(1, -1);
    }
    process.env[key] = value;
  }
}
