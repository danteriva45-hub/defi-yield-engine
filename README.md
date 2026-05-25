# DeFi Intelligence Engine — MCP Server

[![smithery badge](https://smithery.ai/badge/danteriva45/defi-yield-engine)](https://smithery.ai/servers/danteriva45/defi-yield-engine)

A **Model Context Protocol (MCP) server** providing complete DeFi intelligence across Yield, Liquid Staking, Restaking, Real World Assets, and Perpetuals. Risk-adjusted single recommendations with reasoning — 97% fewer tokens than raw DeFiLlama data.

Built with the **MCP SDK** (FastMCP) using the `streamable-http` transport. Compatible with Claude Desktop, Claude Code, OpenClaw, Cursor, Windsurf, and any MCP-compatible client.

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
| `get_best_rwa` | Best Real World Asset protocol (Ondo, BUIDL, Maple) |
| `get_perps_overview` | Top perps by 24h volume (Hyperliquid, dYdX, GMX...) |
| `compare_perps` | Side-by-side perpetuals comparison |

## MCP Tools (free)

| Tool | Description |
|---|---|
| `get_defi_overview` | Full DeFi market snapshot across all categories |
| `yield_alert_set` | Register APY threshold alert |
| `yield_alert_check` | Poll alert status |
| `yield_alert_delete` | Remove an alert |
| `yield_alerts_list` | List all active alerts |
| `server_info` | Server metadata, capabilities and pricing |

## MCP Resources (free)

| Resource URI | Description |
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

Add to your `claude_desktop_config.json`:

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
Protocol       : MCP (Model Context Protocol)
```

---

## Payment

This MCP server uses the **x402 payment protocol** — paid tools require 0.05 USDC per call on the Base network. Free tools and resources require no payment.

```
Recipient : 0x74E3ab71eC674D343aD481Ea20F489C720C11Ad4
Network   : Base (chain 8453)
Asset     : USDC
```

---

## Data Source

**DeFiLlama** public API — 13,800+ pools, 7,000+ protocols, 500+ chains.
Cache: 5 minutes · No API key required · No user data stored.

---

## Discovery Endpoints

- `/.well-known/x402.json` — x402 payment manifest
- `/.well-known/agent.json` — A2A Agent Card (Google Agent-to-Agent Protocol)
- `/health` — Health check

---

## Tech Stack

- **MCP SDK**: FastMCP (Python)
- **Transport**: Streamable HTTP (MCP standard)
- **Payment**: x402 protocol (USDC on Base)
- **Data**: DeFiLlama public API
- **Hosting**: Railway
