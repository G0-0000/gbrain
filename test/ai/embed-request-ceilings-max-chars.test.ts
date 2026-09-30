/**
 * #5680 embedRequestCeilings × GBRAIN_EMBED_MAX_CHARS (local patch ⑤).
 *
 * #5680 predicts a migration's worst-case spend from `embedRequestCeilings`,
 * which truncated each text with the upstream default EMBED_MAX_CHARS (8000).
 * Patch ① made per-text truncation overridable (GBRAIN_EMBED_MAX_CHARS); this
 * pin makes the ceiling prediction use the SAME effective cap. A cap BELOW
 * 8000 previously over-estimated (safe); a cap ABOVE 8000 UNDER-estimated —
 * the real bug this guards.
 *
 * The preflight (embedding-migration-worst-case.ts#planMigrationWorstCase)
 * resolves the cap with resolveEmbedMaxChars(requireConfig()) — the same
 * configure-time cfg.env snapshot embed() truncates with — and passes it as
 * the fourth argument. These tests exercise that exact expression, hermetically
 * via configureGateway({env}) (no process.env writes), mirroring patch ①'s
 * test/ai/embed-env-max-chars.test.ts.
 *
 * Coverage:
 *  - no override → the 8000 default is byte-identical to upstream
 *  - override 2000 (below default) → the ceiling follows (was over-estimated)
 *  - override 12000 (above default) → the ceiling follows (was UNDER-estimated)
 *  - the cap comes from the cfg.env snapshot, invalid values fall back to 8000
 */

import { afterAll, beforeEach, describe, expect, test } from 'bun:test';
import { EMBED_MAX_CHARS, embedRequestCeilings } from '../../src/core/ai/embed-batch-plan.ts';
import { resolveEmbedMaxChars } from '../../src/core/ai/embed-env.ts';
import { configureGateway, requireConfig, resetGateway } from '../../src/core/ai/gateway.ts';

// Without a final reset, this file's last gateway config leaks into whichever
// test file the shard runs next (same rationale as the sibling embed tests).
afterAll(() => resetGateway());

const MODEL = 'ollama:nomic-embed-text';
/** ollama declares no max_batch_tokens / max_input_tokens, so a single text's
 *  ceiling is its UTF-8 byte length after truncation — easy to assert. */
const LONG = 'a'.repeat(9000);
/** The preflight's exact resolution: the configure-time env snapshot. */
const preflightMaxChars = () => resolveEmbedMaxChars(requireConfig());

function configureOllama(env: Record<string, string | undefined> = {}): void {
  configureGateway({ embedding_model: MODEL, embedding_dimensions: 768, env });
}

describe('embedRequestCeilings tracks the effective GBRAIN_EMBED_MAX_CHARS cap', () => {
  beforeEach(() => resetGateway());

  test('no override → ceiling truncates at the 8000 default (upstream-identical)', () => {
    expect(EMBED_MAX_CHARS).toBe(8000);
    expect(embedRequestCeilings([LONG], MODEL)).toEqual([8000]);
    // An explicit EMBED_MAX_CHARS is the same value the default applies.
    expect(embedRequestCeilings([LONG], MODEL, undefined, EMBED_MAX_CHARS)).toEqual([8000]);
  });

  test('override 2000 (below default) → ceiling follows the cap', () => {
    configureOllama({ GBRAIN_EMBED_MAX_CHARS: '2000' });
    expect(preflightMaxChars()).toBe(2000);
    expect(embedRequestCeilings([LONG], MODEL, undefined, preflightMaxChars())).toEqual([2000]);
  });

  test('override 12000 (above default) → ceiling follows the cap (previously under-estimated)', () => {
    configureOllama({ GBRAIN_EMBED_MAX_CHARS: '12000' });
    expect(preflightMaxChars()).toBe(12000);
    // 9000-char text is no longer truncated: the ceiling is the whole input,
    // strictly greater than the old 8000 prediction.
    expect(embedRequestCeilings([LONG], MODEL, undefined, preflightMaxChars())).toEqual([9000]);
  });

  test('ceiling stays additive over the texts of one request under the cap', () => {
    configureOllama({ GBRAIN_EMBED_MAX_CHARS: '2000' });
    // Both texts ride one request (ollama's no_batch_cap default is 16 items),
    // so the documented per-request ceiling is their truncated byte sum.
    expect(embedRequestCeilings(['a'.repeat(5000), 'b'.repeat(100)], MODEL, undefined, preflightMaxChars()))
      .toEqual([2100]);
  });

  test('the cap is the cfg.env snapshot, not process.env at call time', () => {
    // configureGateway({env}) carries the snapshot; an invalid value falls back
    // to EMBED_MAX_CHARS (8000) exactly as patch ①'s truncation would.
    configureOllama({ GBRAIN_EMBED_MAX_CHARS: 'not-a-number' });
    expect(preflightMaxChars()).toBe(EMBED_MAX_CHARS);
    expect(embedRequestCeilings([LONG], MODEL, undefined, preflightMaxChars())).toEqual([8000]);
  });
});
