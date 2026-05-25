# DeFi Intelligence Engine — MCP Server

[![smithery badge](https://smithery.ai/badge/danteriva45/defi-yield-engine)](https://smithery.ai/servers/danteriva45/defi-yield-engine)

A **Model Context Protocol (MCP) server** providing complete DeFi intelligence — Yield, Liquid Staking, Restaking, RWA, Perpetuals, Gas Optimization, and Smart Contract Security. Risk-adjusted single recommendations with reasoning, 97% fewer tokens than raw data.

Built with the **MCP SDK** (FastMCP) · Transport: `streamable-http` · Compatible with Claude Desktop, Claude Code, OpenClaw, Cursor, Windsurf, and any MCP-compatible client.

---

## MCP Tools (paid · 0.05 USDC via x402 on Base)

| Tool | Description |
|---|---|
| `get_best_yield` | Best yield for USDC, USDT, ETH across 548 protocols |
| `get_optimal_allocation` | Multi-protocol capital split for max risk-adjusted yield |
| `explain_risk` | 9-signal risk breakdown for any DeFiLlama protocol |
| `compare_yields` | Side-by-side protocol comparison |
| `get_best_liquid_staking` | Best LST for ETH, SOL, BNB (Lido, Rocket Pool, Jito...) |
| `get_best_restaking` | Best restaking/LRT (EigenLayer, Ether.fi, Renzo, Puffer) |
| `get_best_rwa` | Best Real World Asset protocol (Ondo, BUIDL, Maple, Centrifuge) |
| `get_perps_overview` | Top perps by 24h volume (Hyperliquid, dYdX, GMX, Drift) |
| `compare_perps` | Side-by-side perpetuals comparison |
| `get_optimal_gas_window` | Predict best time window to transact for minimum gas cost |
| `score_contract` | Smart contract security risk score via GoPlus Security API |

## MCP Tools (free)

| Tool | Description |
|---|---|
| `get_gas_price` | Current gas prices (safe/standard/fast) for any EVM chain |
| `get_defi_overview` | Full DeFi market snapshot across all categories |
| `yield_alert_set` | Register APY threshold alert |
| `yield_alert_check` | Poll alert triggered/watching status |
| `yield_alert_delete` | Remove an alert by ID |
| `yield_alerts_list` | List all active alerts |
| `server_info` | Server metadata, all tools, resources, pricing |

## MCP Resources (free)

| URI | Description |
|---|---|
| `defi://market-overview` | Real-time top yields snapshot (~168 tokens) |
| `defi://risk-glossary` | Risk term definitions for output interpretation |

## MCP Prompts (free)

| Prompt | Usage |
|---|---|
| `/yield-check` | Find best yield for asset + amount |
| `/portfolio-optimize` | Optimize multi-asset DeFi allocation |
| `/daily-briefing` | Morning yield market summary |
| `/yield-watch` | Monitor APY threshold + act on trigger |

---

## Installation — Claude Desktop

```json
{
  "mcpServers": {
    "defi-intelligence-engine": {
      "url": "https://defi-yield-engine-production.up.railway.app/mcp"
    }
  }
}
```

## Installation — Any MCP Client

```
MCP Server URL : https://defi-yield-engine-production.up.railway.app/mcp
Transport      : streamable-http
Protocol       : Model Context Protocol (MCP)
```

---

## Payment — x402 Protocol

Paid tools require **0.05 USDC per call** on Base. No signup, no API key.

```
Recipient : 0x74E3ab71eC674D343aD481Ea20F489C720C11Ad4
Network   : Base (chain 8453)
Asset     : USDC
Protocol  : x402
```

---

## Discovery Endpoints

| Endpoint | Description |
|---|---|
| `/.well-known/x402.json` | x402 payment manifest (Coinbase Bazaar) |
| `/.well-known/agent.json` | A2A Agent Card (Google A2A Protocol) |
| `/health` | Health check + x402 status |

---

## Data Sources

- **DeFiLlama** — 13,800+ pools, 7,000+ protocols, 500+ chains (yield, staking, RWA, perps)
- **Etherscan Gas Oracle** — real-time gas prices + pattern-based prediction
- **GoPlus Security API** — smart contract honeypot + risk detection

No API keys required · No user data stored · Cache: 1-5 min

---

## Tech Stack

- **MCP SDK**: FastMCP (Python)
- **Transport**: Streamable HTTP (MCP standard)
- **Payment**: x402 (USDC on Base)
- **Hosting**: Railway
