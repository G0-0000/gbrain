/**
 * resolveEmbedMaxChars — GBRAIN_EMBED_MAX_CHARS env override for per-text
 * truncation in embed() (src/core/ai/gateway.ts).
 *
 * Local / self-hosted embedding servers (llama.cpp GGUF such as
 * qwen3-embedding-4b) time out or overflow their context window on very long
 * single inputs; the operator must be able to bound single-input length to
 * their hardware without forking the gateway. Production-validated ~6 weeks
 * at 2000: full 17,106-chunk re-embed, zero timeouts. Complements the batch
 * token budget (#4650 / #3622): that knob bounds a batch's shape, this one
 * bounds a single text's length.
 *
 * Read from the configure-time cfg.env snapshot (same convention as
 * GBRAIN_EMBED_MAX_BATCH_TOKENS, #3622) — never process.env at call time;
 * buildGatewayConfig folds the operator's process env into the snapshot.
 * Invalid values (non-numeric, non-positive) fall back to EMBED_MAX_CHARS
 * (src/core/ai/embed-batch-plan.ts — the same constant upstream's shared
 * embed-batch-plan truncation defaults to).
 */

import { EMBED_MAX_CHARS } from './embed-batch-plan.ts';
import type { AIGatewayConfig } from './types.ts';

export function resolveEmbedMaxChars(cfg: Pick<AIGatewayConfig, 'env'>): number {
  const raw = parseInt(cfg.env?.GBRAIN_EMBED_MAX_CHARS ?? '', 10);
  return Number.isFinite(raw) && raw > 0 ? raw : EMBED_MAX_CHARS;
}
