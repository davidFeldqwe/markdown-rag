# markdown-rag

A minimal retrieval-augmented generation demo in TypeScript, with no build step. It follows Matt Pocock's AI Hero guide [Create Embeddings With Vercel's AI SDK](https://www.aihero.dev/create-embeddings-with-vercel-ai-sdk): `embed`, `embedMany`, and `cosineSimilarity` from the `ai` package. Embeddings are stored in a local JSON file. There is no vector database.

The notes in `docs/` describe a fictional product, Lumen Notes. They are sample data, not a real service.

## Run it

```sh
npm install
cp .env.example .env   # set OPENAI_API_KEY in that file, or export it
npm run index -- ./docs
npm run ask -- "How does Lumen sync a notebook?"
```

`npm run index` chunks the Markdown, embeds the chunks with `openai.embedding("text-embedding-3-small")`, and writes `embeddings.json` (gitignored). `npm run ask` embeds the question, ranks chunks by cosine similarity, and calls `generateText` with only the top matches. Each answer ends with `file #chunk` citations. Chunks under a cosine similarity of 0.42 are dropped. If none remain, the command refuses and does not call the chat model.

```sh
npm test
```

Tests use a deterministic stand-in for the embedding model and do not need an API key.

## API notes

`embed`, `embedMany`, `cosineSimilarity`, and `openai.embedding("text-embedding-3-small")` match the guide on `ai` 7 and `@ai-sdk/openai` 4. The guide does not cover answering. `generateText` in this SDK version deprecates the `system` option, so the answer call passes the same text as `instructions`. The chat model is `gpt-4o-mini`.
