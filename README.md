# LLM API Cost, Prompt Caching ROI & AI Developer Toolkit Notes

A practical engineering cheat sheet for estimating Large Language Model (LLM) API bills, modeling prompt caching break-even points, sizing GPU VRAM for self-hosted open-weight models, and structuring clean Model Context Protocol (`mcp.json`) configurations.

If you want to run these calculations interactively with live provider pricing without spreadsheets, see the free browser-based calculators and model directory at **[LLMBucket](https://llmbucket.com)**.

---

## 1. Prompt Caching ROI & Break-Even Math

Prompt caching is one of the highest-leverage ways to cut LLM API bills in agent loops, RAG pipelines, and coding assistants—provided your prefix stays byte-identical and your cache hit rate clears the write surcharge.

### Standard Pricing Multipliers (Anthropic-style Explicit Caching)

Let $P_{\text{in}}$ be the model's base uncached input token price (per 1M tokens):

| Operation | Multiplier | Cost Relative to Uncached Input |
| :--- | :--- | :--- |
| **Uncached Input** | $1.00\times$ | Baseline |
| **5-Minute Cache Write** | $1.25\times$ | +25% write surcharge |
| **1-Hour Cache Write** | $2.00\times$ | +100% write surcharge |
| **Cache Read (Hit)** | $0.10\times$ | 90% discount |

*(Note: Providers such as OpenAI and DeepSeek also offer automatic prompt caching with 0% write surcharge and 50%–90% read discounts, while explicit TTL caching like Anthropic's gives deterministic control over longer session windows.)*

### Exact Break-Even Formulas

Suppose a shared system prompt + tool schema + codebase context of $T$ tokens is written once and reused across $N$ subsequent requests before the TTL expires.

- **Uncached total cost** for $1 + N$ requests:
  $$\text{Cost}_{\text{uncached}} = (1 + N) \times T \times P_{\text{in}}$$

- **5-Minute TTL Cache ($1.25\times$ write, $0.1\times$ read):**
  $$\text{Cost}_{\text{5m}} = (1.25 + 0.1 N) \times T \times P_{\text{in}}$$
  Setting $\text{Cost}_{\text{5m}} \le \text{Cost}_{\text{uncached}}$:
  $$1.25 + 0.1 N \le 1 + N \implies 0.25 \le 0.9 N \implies N \ge 0.28$$
  **Takeaway:** With a 5-minute TTL, **1 single cache hit** ($N = 1$, 2 total calls) already yields a **32.5% net savings** ($1.35\times$ vs $2.0\times$).

- **1-Hour TTL Cache ($2.0\times$ write, $0.1\times$ read):**
  $$\text{Cost}_{\text{1h}} = (2.0 + 0.1 N) \times T \times P_{\text{in}}$$
  Setting $\text{Cost}_{\text{1h}} \le \text{Cost}_{\text{uncached}}$:
  $$2.0 + 0.1 N \le 1 + N \implies 1.0 \le 0.9 N \implies N \ge 1.12$$
  **Takeaway:** With a 1-hour TTL, **1 hit loses money** ($2.1\times$ vs $2.0\times$, a 5% loss). You need **at least 2 cache hits** ($N \ge 2$, 3 total calls) to break even ($2.2\times$ vs $3.0\times$, a **26.7% savings**).

### Production Rules of Thumb
1. **Put volatile data last:** Place timestamps, user IDs, and retrieved chunks *after* your static system prompt and tool definitions. Changing one character near the top invalidates the entire downstream prefix.
2. **Respect minimum token thresholds:** Most providers ignore cache markers below `1,024`–`4,096` tokens depending on the model family.
3. **Model your actual hit rate:** Use the interactive **[Prompt Caching Calculator](https://llmbucket.com/tools/prompt-caching-calculator)** to test 5-minute vs 1-hour TTLs against your traffic volume and prefix size.

---

## 2. Token Cost Estimation & Vision / PDF Tile Sizing

### Why Output Tokens Dominate Agent Bills
Many teams budget using input prices only, then get surprised when production bills are 3x–5x higher. Across frontier models, **output tokens typically cost 3x to 8x more than input tokens**—and agentic workflows amplify this in two ways:
- **Multi-turn context snowballing:** In a 10-step tool-calling loop, each turn re-sends the growing conversation history as input while generating fresh tool arguments and reasoning tokens as output.
- **Hidden reasoning tokens:** Models with extended thinking / reasoning modes bill internal chain-of-thought tokens at the output token rate, even when you only display the final answer.

Before deploying a multi-step agent, estimate full turn-by-turn spend using the **[LLM Cost Calculator](https://llmbucket.com/tools/llm-cost-calculator)** and compare up-to-date model rates on **[LLMBucket Models](https://llmbucket.com/models)**.

### Vision & Multi-Page PDF Token Traps
Images and PDFs are converted into fixed-size pixel tiles or per-page token blocks before hitting the model:
- **High-res tile explosion:** Under 512×512 tiling schemes (e.g., OpenAI high-detail mode), an image resized to fit within 2048×2048 is sliced into 512×512 tiles (`170 tokens/tile + 85 base tokens`). A single 1920×1080 screenshot costs ~`1,105` tokens (`6 tiles × 170 + 85`), whereas a cropped 800×600 UI element may only need ~`425` tokens (`2 tiles × 170 + 85`).
- **Pixel-area formulas:** Other APIs (such as Claude) compute tokens roughly as `(width × height) / 750`, capping longest edges at ~1568px (~`1,600` tokens max per image).
- **Multi-page PDFs:** Sending a 40-page scanned PDF with `high` detail can quietly add `45,000–75,000` input tokens per request. Downscale screenshots to the minimum legible DPI, crop whitespace, and extract plain text with local OCR whenever visual layout is unnecessary.

---

## 3. Quick GPU VRAM Estimation Formula (4-Bit / 8-Bit / FP16)

When evaluating whether to self-host open-weight models (Llama, Qwen, DeepSeek, Mistral) vs calling a managed API, total VRAM consumption equals **model weights + activation/runtime overhead + KV cache**:

$$\text{VRAM (GB)} \approx \left(\text{Params}_{\text{B}} \times \text{Bytes/Param} \times 1.2\right) + \text{KV Cache (GB)}$$

### Bytes per Parameter by Precision

| Precision / Quantization | Bytes / Param | Multiplier (with ~1.2× overhead) | 8B Model Base VRAM | 70B Model Base VRAM |
| :--- | :--- | :--- | :--- | :--- |
| **4-bit (INT4 / Q4_K_M / AWQ)** | ~0.55 GB / B | ~0.66 | ~5.3 GB | ~46.2 GB |
| **8-bit (FP8 / INT8 / Q8_0)** | ~1.00 GB / B | ~1.20 | ~9.6 GB | ~84.0 GB |
| **FP16 / BF16** | ~2.00 GB / B | ~2.40 | ~19.2 GB | ~168.0 GB |

*(The `1.2×` factor accounts for PyTorch/vLLM/llama.cpp CUDA kernels, activation buffers, and quantization group metadata.)*

### Estimating KV Cache Memory
Long context windows and concurrent requests rapidly expand KV cache usage:

$$\text{KV Cache (bytes)} = 2 \times L \times H_{\text{kv}} \times D_{\text{head}} \times B_{\text{kv}} \times S \times N_{\text{batch}}$$

Where:
- $L$ = number of transformer layers
- $H_{\text{kv}}$ = number of key-value heads (much smaller in Grouped-Query Attention models)
- $D_{\text{head}}$ = dimension per head (commonly `128`)
- $B_{\text{kv}}$ = bytes per KV element (`2` for FP16, `1` for FP8 KV cache)
- $S$ = sequence length (prompt + generation)
- $N_{\text{batch}}$ = concurrent sequences

For fast hardware sizing across RTX 4090 (24 GB), RTX 5090 (32 GB), Apple Silicon Unified Memory, and A100/H100 nodes, check the **[LLM VRAM Calculator](https://llmbucket.com/tools/llm-vram-calculator)**.

---

## 4. Clean `mcp.json` Snippet for Claude Desktop & Cursor

Hand-editing Model Context Protocol (`mcp.json`) files frequently breaks due to trailing commas, wrong `command` paths, or missing environment variable blocks. Below is a clean, production-ready template compatible with **Claude Desktop** (`claude_desktop_config.json`) and **Cursor** (`.cursor/mcp.json`):

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-filesystem",
        "/path/to/your/workspace"
      ]
    },
    "github": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-github"
      ],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "ghp_your_token_here"
      }
    },
    "fetch": {
      "command": "uvx",
      "args": [
        "mcp-server-fetch"
      ]
    }
  }
}
```

### Quick Tips
- Always verify `npx` or `uvx` is on your GUI app's `PATH` (on macOS/Linux, GUI apps launched outside a terminal sometimes need the full binary path such as `/usr/local/bin/npx` or `~/.local/bin/uvx`).
- Never commit real API tokens inside `.cursor/mcp.json`—use `.gitignore` or environment variable injection.
- Generate and validate multi-server configs cleanly with the **[MCP Config Generator](https://llmbucket.com/tools/mcp-config-generator)**.

---

## 5. Interactive Browser Tools & Live Pricing Reference

All tools below run directly in your browser with no signup or API key required:

- **[LLMBucket Home](https://llmbucket.com)** — Overview of live LLM pricing benchmarks, provider comparisons, and engineering calculators.
- **[Live LLM Models & Pricing Directory](https://llmbucket.com/models)** — Compare input, output, cached input, and context limits across OpenAI, Anthropic, Google Gemini, DeepSeek, and open-weight providers.
- **[Prompt Caching ROI Calculator](https://llmbucket.com/tools/prompt-caching-calculator)** — Model 5-minute vs 1-hour write surcharges, cache hit ratios, and monthly savings.
- **[LLM API Cost Calculator](https://llmbucket.com/tools/llm-cost-calculator)** — Estimate per-request and monthly production spend across input/output ratios and agent turns.
- **[LLM GPU VRAM Calculator](https://llmbucket.com/tools/llm-vram-calculator)** — Size model weights, KV cache, and quantization headroom for local GPUs and cloud instances.
- **[MCP Config Generator](https://llmbucket.com/tools/mcp-config-generator)** — Build validated `mcp.json` configurations for Claude Desktop, Cursor, and Windsurf.
- **[OpenRouter vs Direct API Comparison](https://llmbucket.com/providers/openrouter-vs-direct)** — Breakdown of gateway fee markups, latency trade-offs, prompt caching support, and failover routing.
