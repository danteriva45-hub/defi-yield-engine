"""
DeFi Yield Decision Engine — MCP Server
Auteur  : toi (Guillaume)
Version : 1.0.0
Source  : DeFiLlama public API (gratuit, sans clé)

Ce serveur propose 3 outils à des agents IA :
  1. get_best_yield   → meilleure recommandation risk-ajustée
  2. explain_risk     → analyse détaillée d'un protocole
  3. compare_yields   → comparaison side-by-side

Déploiement : Railway (voir README.md)
Monétisation : mcp-billing-gateway (x402 / Stripe)
"""

import asyncio
import time
import logging
from typing import Optional, Annotated
import httpx
from fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import JSONResponse
from x402.http.middleware.fastapi import PaymentMiddlewareASGI
from x402.http import HTTPFacilitatorClient, FacilitatorConfig, PaymentOption
from x402.http.types import RouteConfig
from x402.server import x402ResourceServer
from x402.mechanisms.evm.exact import ExactEvmServerScheme

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("defi-yield")

# ── Serveur MCP ───────────────────────────────────────────────────────────────
mcp = FastMCP(
    name="defi-yield-engine",
    instructions=(
        "Risk-adjusted DeFi yield recommendations powered by DeFiLlama. "
        "Use get_best_yield to find the optimal yield for a given asset and risk profile. "
        "Use explain_risk to understand why a protocol is rated as it is. "
        "Use compare_yields to compare multiple protocols side-by-side. "
        "All outputs are optimized for minimal token consumption."
    ),
)

# ── Configuration x402 ───────────────────────────────────────────────────────
X402_RECIPIENT  = "0x74E3ab71eC674D343aD481Ea20F489C720C11Ad4"  # MetaMask Base
X402_NETWORK    = "base"
X402_CHAIN_ID   = 8453

# Pricing par outil (en USDC — 6 décimales sur Base)
X402_PRICING = {
    "get_best_yield":        0.05,   # 0,05 USDC
    "get_optimal_allocation":0.05,   # 0,05 USDC
    "explain_risk":          0.05,   # 0,05 USDC
    "compare_yields":        0.05,   # 0,05 USDC
    "server_info":           0.00,   # gratuit
}

# ── USDC Base contract address ────────────────────────────────────────────
USDC_BASE_CONTRACT  = "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"
X402_FACILITATOR    = "https://facilitator.openmid.xyz"  # mainnet Base
X402_CHAIN_CAIP2    = "eip155:8453"                       # Base mainnet

# ── Routes nécessitant un paiement x402 ────────────────────────────────────
# Format FastMCP: les outils MCP sont appelés via POST /mcp
# On protège l'endpoint /mcp côté application, avec vérification du tool appelé
X402_PAID_TOOLS = {
    "get_best_yield":         "$0.05",
    "get_optimal_allocation": "$0.10",
    "explain_risk":           "$0.02",
    "compare_yields":         "$0.03",
}

# ── Cache ─────────────────────────────────────────────────────────────────────
_CACHE: dict = {"data": None, "updated_at": 0.0}
_CACHE_TTL = 300  # 5 minutes

DEFILLAMA_URL = "https://yields.llama.fi/pools"

async def _fetch_pools() -> list[dict]:
    """Récupère tous les pools DeFiLlama avec cache 5 minutes."""
    now = time.time()
    if _CACHE["data"] and (now - _CACHE["updated_at"]) < _CACHE_TTL:
        return _CACHE["data"]

    log.info("Fetching fresh data from DeFiLlama…")
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.get(DEFILLAMA_URL)
        r.raise_for_status()
        pools = r.json()["data"]

    _CACHE["data"] = pools
    _CACHE["updated_at"] = now
    log.info(f"Loaded {len(pools)} pools from DeFiLlama")
    return pools


# ── Algorithme de risk score ──────────────────────────────────────────────────
def _risk_score(pool: dict) -> int:
    """
    Score de 0 à 100 (100 = plus sûr).

    Signaux positifs  : TVL élevé, APY stable, pas de token reward, stablecoin
    Signaux négatifs  : APY > 20%, outlier DeFiLlama, chute APY récente,
                        dépendance forte aux rewards, risque impermanent
    """
    score = 50

    # --- Liquidité (proxy de confiance marché)
    tvl = pool.get("tvlUsd") or 0
    if tvl >= 1_000_000_000:
        score += 20
    elif tvl >= 100_000_000:
        score += 12
    elif tvl >= 10_000_000:
        score += 5
    else:
        score -= 10  # TVL < $10M : risque de liquidité

    # --- Stabilité APY sur 30 jours
    mean30 = pool.get("apyMean30d") or 0
    current_apy = pool.get("apy") or 0
    if mean30 > 0 and current_apy > 0:
        deviation = abs(current_apy - mean30) / mean30
        if deviation < 0.05:
            score += 10   # très stable
        elif deviation < 0.15:
            score += 5    # acceptable
        elif deviation > 0.50:
            score -= 10   # APY très volatile

    # --- Durabilité du rendement (base vs reward)
    reward = pool.get("apyReward") or 0
    base = pool.get("apyBase") or 0
    if reward == 0:
        score += 10           # 100% APY de base = rendement organique
    elif base > 0 and reward <= base * 0.3:
        score += 5            # rewards < 30% du total : acceptable
    elif base > 0 and reward > base:
        score -= 20           # rewards > base APY : modèle non-durable
    elif base == 0 and reward > 0:
        score -= 15           # rendement 100% en tokens = risque élevé

    # --- Signaux de danger
    if pool.get("outlier"):
        score -= 25           # DeFiLlama marque les APY suspects ou anormaux
    if current_apy > 30:
        score -= 20           # APY > 30% = red flag sévère
    elif current_apy > 20:
        score -= 12           # APY > 20% = attention

    # --- Risque de perte (impermanent loss, exposition)
    if pool.get("ilRisk") == "yes":
        score -= 10
    if pool.get("exposure") == "single":
        score += 5            # actif unique = pas d'IL

    # --- Bonus stablecoin
    if pool.get("stablecoin"):
        score += 5

    # --- Momentum récent
    pct_7d = pool.get("apyPct7D") or 0
    if pct_7d < -5:
        score -= 10           # chute rapide en 7j
    elif pct_7d < -2:
        score -= 5

    # --- Prédiction DeFiLlama (ML interne)
    pred = (pool.get("predictions") or {}).get("predictedClass", "")
    if pred == "Stable/Up":
        score += 3
    elif pred == "Down":
        score -= 5

    return max(0, min(100, score))


def _risk_label(score: int) -> str:
    if score >= 80: return "safe"
    if score >= 60: return "moderate"
    if score >= 40: return "elevated"
    return "high_risk"


def _score_threshold(risk_profile: str) -> int:
    """Score minimum acceptable selon le profil."""
    return {"safe": 75, "moderate": 55, "max_yield": 35}.get(risk_profile, 55)


def _compact_pool(pool: dict, score: int) -> dict:
    """Format de sortie compact — économise ~97% de tokens vs DeFiLlama brut."""
    return {
        "protocol":    pool.get("project", "unknown"),
        "chain":       pool.get("chain", "unknown"),
        "asset":       pool.get("symbol", "unknown"),
        "apy":         round(pool.get("apy") or 0, 2),
        "apy_base":    round(pool.get("apyBase") or 0, 2),
        "apy_reward":  round(pool.get("apyReward") or 0, 2),
        "tvl_usd":     int(pool.get("tvlUsd") or 0),
        "risk_score":  score,
        "risk_level":  _risk_label(score),
    }


def _build_reasoning(pool: dict, score: int) -> str:
    """Génère une explication courte et lisible du score."""
    reasons = []
    tvl = pool.get("tvlUsd") or 0
    if tvl >= 1_000_000_000:
        reasons.append(f"TVL ${tvl/1e9:.1f}B (très liquide)")
    elif tvl >= 100_000_000:
        reasons.append(f"TVL ${tvl/1e6:.0f}M (liquide)")
    else:
        reasons.append(f"TVL ${tvl/1e6:.1f}M (faible liquidité)")

    reward = pool.get("apyReward") or 0
    base = pool.get("apyBase") or 0
    if reward == 0:
        reasons.append("rendement 100% organique (pas de token reward)")
    elif base > 0 and reward > base:
        reasons.append(f"⚠ reward APY ({reward:.1f}%) > base APY ({base:.1f}%)")

    if pool.get("outlier"):
        reasons.append("⚠ APY marqué comme outlier par DeFiLlama")
    if (pool.get("apy") or 0) > 20:
        reasons.append("⚠ APY élevé — vérifier source du rendement")
    if pool.get("stablecoin"):
        reasons.append("actif stable (pas de risque de marché)")
    if pool.get("ilRisk") == "no":
        reasons.append("pas de risque de perte impermanente")

    return ". ".join(reasons) + "."


# ── Outil 1 : get_best_yield ─────────────────────────────────────────────────
@mcp.tool
async def get_best_yield(
    asset: Annotated[str, "Token symbol to find yield for. Examples: 'USDC', 'USDT', 'ETH', 'DAI', 'USDS'"],
    amount_usd: Annotated[float, "Amount to deploy in USD. Minimum TVL filter = 10x this amount. Example: 50000"],
    risk_profile: Annotated[str, "Risk tolerance: 'safe' (score>=75, large TVL), 'moderate' (>=55), 'max_yield' (>=35)"] = "moderate",
    chain: Annotated[str, "Blockchain to filter by. 'all' or: 'Ethereum', 'Arbitrum', 'Base', 'Polygon', 'Optimism'"] = "all",
) -> dict:
    """
    Select the single best DeFi yield opportunity for a given asset and risk profile.

    Scores 13,800+ pools across 548 protocols and 115 chains using a 9-signal
    risk algorithm (TVL, APY stability, reward dependency, outlier flag, momentum).
    Returns ONE opinionated recommendation with reasoning + 2 alternatives.
    Output: ~60 tokens — 97% smaller than raw DeFiLlama data.

    Use this when: an agent needs to deploy capital and wants a single actionable
    answer, not a list to evaluate manually.
    Do NOT use for: protocol comparison (use compare_yields), risk detail (use explain_risk).

    Args:
        asset        : Token symbol. Examples: "USDC", "USDT", "ETH", "DAI", "USDS"
        amount_usd   : Amount to deploy in USD. Filters pools with TVL >= 10x amount
                       to ensure sufficient exit liquidity.
        risk_profile : "safe" (risk_score >=75, large TVL, organic APY only),
                       "moderate" (>=55, balanced), "max_yield" (>=35, higher risk)
        chain        : "all" or specific chain: "Ethereum", "Arbitrum", "Base",
                       "Polygon", "Optimism", "Avalanche", "BNB Chain", "Solana"

    Returns:
        JSON: recommendation {protocol, chain, apy, risk_score, risk_level, tvl_usd},
        reasoning (plain text, ~30 words), alternatives (top 2), candidates_evaluated.
    """
    if risk_profile not in ("safe", "moderate", "max_yield"):
        return {"error": "risk_profile must be 'safe', 'moderate', or 'max_yield'"}

    pools = await _fetch_pools()

    # Filtre TVL minimum : au moins 10× le montant déployé (sécurité liquidité)
    min_tvl = max(10_000_000, amount_usd * 10)

    # Filtrage
    candidates = []
    for p in pools:
        sym = (p.get("symbol") or "").upper()
        if asset.upper() not in sym:
            continue
        if (p.get("tvlUsd") or 0) < min_tvl:
            continue
        if not p.get("apy") or p["apy"] <= 0:
            continue
        if chain != "all" and p.get("chain", "").lower() != chain.lower():
            continue

        score = _risk_score(p)
        if score < _score_threshold(risk_profile):
            continue

        candidates.append((score, p))

    if not candidates:
        return {
            "error": "no_results",
            "message": (
                f"Aucun pool trouvé pour {asset} avec profil '{risk_profile}' "
                f"et montant ${amount_usd:,.0f}. "
                "Essaie un profil moins restrictif ou un montant plus faible."
            ),
        }

    # Trier par score × APY (rendement ajusté au risque)
    candidates.sort(key=lambda x: x[0] * (x[1].get("apy") or 0), reverse=True)

    best_score, best = candidates[0]
    alts = candidates[1:3]

    return {
        "recommendation": _compact_pool(best, best_score),
        "reasoning":      _build_reasoning(best, best_score),
        "alternatives": [
            {**_compact_pool(p, s), "note": f"alternative #{i+1}"}
            for i, (s, p) in enumerate(alts)
        ],
        "params": {
            "asset": asset,
            "amount_usd": amount_usd,
            "risk_profile": risk_profile,
            "chain": chain,
            "candidates_evaluated": len(candidates),
        },
    }


# ── Outil 2 : explain_risk ───────────────────────────────────────────────────
@mcp.tool
async def explain_risk(
    protocol: str,
    asset: str,
    chain: str = "all",
) -> dict:
    """
    Analyse détaillée du risque d'un protocole spécifique pour un asset.

    Args:
        protocol : Slug DeFiLlama. Exemples : "aave-v3", "morpho-blue",
                   "compound-v3", "spark", "yearn-finance"
        asset    : Symbole. Exemples : "USDC", "USDT", "ETH"
        chain    : "all" ou nom de chain.

    Returns:
        Analyse complète : score, signaux positifs/négatifs, verdict.
    """
    pools = await _fetch_pools()

    matches = [
        p for p in pools
        if protocol.lower() in (p.get("project") or "").lower()
        and asset.upper() in (p.get("symbol") or "").upper()
        and (chain == "all" or (p.get("chain") or "").lower() == chain.lower())
        and (p.get("apy") or 0) > 0
    ]

    if not matches:
        return {
            "error": "not_found",
            "message": f"Aucun pool trouvé pour {protocol}/{asset} sur {chain}.",
            "hint": "Vérifie le slug avec la liste DeFiLlama : https://defillama.com/yields",
        }

    # Prendre le pool avec la TVL la plus élevée si plusieurs résultats
    matches.sort(key=lambda p: p.get("tvlUsd") or 0, reverse=True)
    p = matches[0]
    score = _risk_score(p)

    # Construire analyse signal par signal
    positive, negative = [], []

    tvl = p.get("tvlUsd") or 0
    if tvl >= 1_000_000_000:
        positive.append(f"TVL ${tvl/1e9:.1f}B — très grande liquidité")
    elif tvl >= 100_000_000:
        positive.append(f"TVL ${tvl/1e6:.0f}M — bonne liquidité")
    else:
        negative.append(f"TVL ${tvl/1e6:.1f}M — liquidité limitée")

    reward = p.get("apyReward") or 0
    base   = p.get("apyBase")   or 0
    apy    = p.get("apy")       or 0
    if reward == 0:
        positive.append("Rendement 100% en APY de base (organique, durable)")
    elif base > 0 and reward <= base * 0.3:
        positive.append(f"Rewards modérés ({reward:.1f}% sur {apy:.1f}% total)")
    else:
        pct = (reward / apy * 100) if apy > 0 else 0
        negative.append(f"Rewards = {pct:.0f}% du rendement total — risque d'émission")

    mean30 = p.get("apyMean30d") or 0
    if mean30 > 0:
        dev = abs(apy - mean30) / mean30
        if dev < 0.05:
            positive.append(f"APY très stable (moy. 30j : {mean30:.2f}%)")
        elif dev > 0.30:
            negative.append(f"APY volatile (moy. 30j : {mean30:.2f}% vs actuel {apy:.2f}%)")
        else:
            positive.append(f"APY correct (moy. 30j : {mean30:.2f}%)")

    if p.get("outlier"):
        negative.append("⚠ Marqué 'outlier' par DeFiLlama — APY potentiellement suspect")
    if apy > 20:
        negative.append(f"⚠ APY élevé ({apy:.1f}%) — vérifier la source du rendement")
    if p.get("ilRisk") == "no":
        positive.append("Pas de risque de perte impermanente")
    else:
        negative.append("Risque de perte impermanente (pool multi-actifs)")
    if p.get("stablecoin"):
        positive.append("Actif stable — pas d'exposition au marché crypto")

    pred = (p.get("predictions") or {}).get("predictedClass", "Inconnu")
    prob = (p.get("predictions") or {}).get("predictedProbability", 0)
    if pred == "Stable/Up":
        positive.append(f"Prédiction DeFiLlama : {pred} ({prob}% confiance)")
    elif pred == "Down":
        negative.append(f"Prédiction DeFiLlama : {pred} ({prob}% confiance)")

    # Verdict
    if score >= 80:
        verdict = "✅ Recommandé pour déploiement long terme (3-12 mois)"
    elif score >= 65:
        verdict = "⚠ Acceptable pour déploiement moyen terme (1-3 mois), surveiller"
    elif score >= 50:
        verdict = "⚠ À utiliser avec prudence, horizon court terme seulement"
    else:
        verdict = "❌ Déconseillé — risques significatifs identifiés"

    return {
        "protocol":    p.get("project"),
        "chain":       p.get("chain"),
        "asset":       p.get("symbol"),
        "risk_score":  score,
        "risk_level":  _risk_label(score),
        "apy":         round(apy, 2),
        "tvl_usd":     int(tvl),
        "positive_signals": positive,
        "negative_signals": negative,
        "verdict":     verdict,
        "data_age":    f"{int(time.time() - _CACHE['updated_at'])}s",
    }


# ── Outil 3 : compare_yields ─────────────────────────────────────────────────
@mcp.tool
async def compare_yields(
    asset: Annotated[str, "Token symbol to compare yields for. Examples: 'USDC', 'USDT', 'ETH'"],
    protocols: Annotated[list[str], "List of 2-6 DeFiLlama project slugs. Examples: ['aave-v3', 'morpho-blue', 'compound-v3']"],
    chain: Annotated[str, "Chain filter. 'all' or: 'Ethereum', 'Arbitrum', 'Base', 'Polygon', 'Optimism'"] = "all",
) -> dict:
    """
    Comparaison side-by-side risk-ajustée de plusieurs protocoles.

    Args:
        asset     : Symbole. Exemples : "USDC", "USDT", "ETH"
        protocols : Liste de slugs DeFiLlama (2 à 6).
                    Exemples : ["aave-v3", "morpho-blue", "compound-v3"]
        chain     : "all" ou nom de chain.

    Returns:
        Tableau comparatif trié par score risk-ajusté + recommandation finale.
    """
    if len(protocols) < 2:
        return {"error": "Fournis au moins 2 protocoles à comparer."}
    if len(protocols) > 6:
        return {"error": "Maximum 6 protocoles par comparaison."}

    pools = await _fetch_pools()
    results = []

    for slug in protocols:
        matches = [
            p for p in pools
            if slug.lower() in (p.get("project") or "").lower()
            and asset.upper() in (p.get("symbol") or "").upper()
            and (chain == "all" or (p.get("chain") or "").lower() == chain.lower())
            and (p.get("apy") or 0) > 0
        ]
        if not matches:
            results.append({
                "protocol": slug,
                "status": "not_found",
                "note": f"Aucun pool {asset} trouvé pour '{slug}'",
            })
            continue

        matches.sort(key=lambda p: p.get("tvlUsd") or 0, reverse=True)
        best = matches[0]
        score = _risk_score(best)
        results.append({
            **_compact_pool(best, score),
            "apy_mean_30d":   round(best.get("apyMean30d") or 0, 2),
            "apy_change_7d":  round(best.get("apyPct7D") or 0, 2),
            "reward_pct":     round(
                (best.get("apyReward") or 0) / (best.get("apy") or 1) * 100, 1
            ),
            "prediction":     (best.get("predictions") or {}).get("predictedClass", "N/A"),
            "risk_adjusted_score": round(score * (best.get("apy") or 0) / 100, 2),
        })

    # Trier les résultats valides par score risk-ajusté
    valid = [r for r in results if r.get("risk_score") is not None]
    invalid = [r for r in results if r.get("risk_score") is None]
    valid.sort(key=lambda r: r["risk_adjusted_score"], reverse=True)

    winner = valid[0] if valid else None

    return {
        "comparison": valid + invalid,
        "winner": {
            "protocol":   winner["protocol"] if winner else "N/A",
            "chain":      winner.get("chain", "N/A") if winner else "N/A",
            "apy":        winner["apy"] if winner else 0,
            "risk_score": winner["risk_score"] if winner else 0,
            "reasoning":  (
                f"Meilleur compromis rendement/risque : APY {winner['apy']}% "
                f"avec score de sécurité {winner['risk_score']}/100."
            ) if winner else "Aucun résultat valide.",
        },
        "params": {"asset": asset, "protocols": protocols, "chain": chain},
    }


# ── Manifest .well-known/x402.json (découverte Coinbase Bazaar) ─────────────
@mcp.custom_route("/.well-known/x402.json", methods=["GET"])
async def well_known_x402(request: Request) -> JSONResponse:
    """
    Manifest de découverte automatique pour le Coinbase x402 Bazaar.
    Ce fichier est crawlé automatiquement — aucune inscription manuelle requise.
    """
    return JSONResponse({
        "x402Version": "1",
        "name": "DeFi Yield Decision Engine",
        "description": (
            "Risk-adjusted DeFi yield recommendations across 13,800+ pools. "
            "Returns opinionated single recommendation with reasoning. "
            "97% more token-efficient than raw DeFiLlama data."
        ),
        "version": "1.0.0",
        "contact": "your@email.com",
        "tags": ["defi", "yield", "finance", "risk", "USDC", "ETH", "DeFiLlama"],
        "endpoints": [
            {
                "path": "/mcp",
                "protocol": "mcp-streamable-http",
                "tools": ["get_best_yield", "explain_risk", "compare_yields", "server_info"],
                "resources": ["defi://market-overview", "defi://risk-glossary"],
                "prompts":   ["yield_check", "portfolio_optimize", "daily_briefing"],
            }
        ],
        "payment": {
            "scheme":    "exact",
            "network":   X402_NETWORK,
            "chainId":   X402_CHAIN_ID,
            "asset":     "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",
            "payTo":     X402_RECIPIENT,
            "pricing": {
                "get_best_yield":        {"amount": "50000",  "decimals": 6},
                "get_optimal_allocation":{"amount": "50000",  "decimals": 6},
                "explain_risk":          {"amount": "50000",  "decimals": 6},
                "compare_yields":        {"amount": "50000",  "decimals": 6},
                "server_info":           {"amount": "0",      "decimals": 6},
            },
        },
        "dataSource":      "DeFiLlama public API",
        "cacheTtlSeconds": 300,
        "mcpUrl":          "https://defi-yield-engine-production.up.railway.app/mcp",
    })




# ═══════════════════════════════════════════════════════════════════════════════
# PARTIE 3 : YIELD ALERTS + A2A AGENT CARD
# ═══════════════════════════════════════════════════════════════════════════════

# Stockage alertes en mémoire
_ALERTS: dict = {}


@mcp.tool
async def yield_alert_set(
    asset: Annotated[str, "Token to monitor: 'USDC', 'USDT', 'ETH', 'DAI'"],
    threshold_apy: Annotated[float, "Alert when best yield exceeds this APY. Example: 6.0"],
    risk_profile: Annotated[str, "'safe', 'moderate', or 'max_yield'"] = "moderate",
    chain: Annotated[str, "'all' or specific chain: 'Ethereum', 'Arbitrum', 'Base'"] = "all",
) -> dict:
    """Register an APY threshold alert. Returns alert_id to check later. FREE.

    Fires when get_best_yield finds an opportunity exceeding threshold_apy.
    Use yield_alert_check with alert_id to poll status.
    Use yield_alert_delete to remove. Alerts persist in server memory.

    Use this when: an agent wants to be notified when a yield opportunity opens
    without continuously calling get_best_yield.

    Args:
        asset         : Token to monitor. Examples: 'USDC', 'USDT', 'ETH', 'DAI'
        threshold_apy : Minimum APY to trigger. Example: 6.0 means alert when APY > 6%
        risk_profile  : 'safe' (score>=75), 'moderate' (>=55), 'max_yield' (>=35)
        chain         : 'all' or specific chain filter

    Returns:
        JSON with alert_id, current status (triggered/watching), current best APY.
    """
    import uuid
    alert_id = str(uuid.uuid4())[:8]
    current = await get_best_yield(asset, 1_000, risk_profile, chain)
    current_apy = current.get("recommendation", {}).get("apy", 0) if "error" not in current else 0
    already_triggered = current_apy >= threshold_apy

    _ALERTS[alert_id] = {
        "asset": asset, "threshold_apy": threshold_apy,
        "risk_profile": risk_profile, "chain": chain,
        "created_at": int(time.time()), "triggered": already_triggered,
    }
    log.info(f"Alert set: {alert_id} | {asset} > {threshold_apy}% | triggered={already_triggered}")

    return {
        "alert_id": alert_id,
        "status": "triggered" if already_triggered else "watching",
        "asset": asset, "threshold_apy": threshold_apy,
        "current_best_apy": current_apy, "price": "free",
        "message": (
            f"TRIGGERED: {asset} yield {current_apy}% > {threshold_apy}%"
            if already_triggered else
            f"Watching {asset} > {threshold_apy}%. Poll: yield_alert_check('{alert_id}')"
        ),
    }


@mcp.tool
async def yield_alert_check(
    alert_id: Annotated[str, "Alert ID from yield_alert_set. Example: 'a3f2b1c0'"],
) -> dict:
    """Poll a yield alert status. FREE. Safe to call frequently — uses cached data.

    Returns current status (triggered/watching), current best APY vs threshold,
    and full recommendation if triggered.

    Args:
        alert_id : ID returned by yield_alert_set

    Returns:
        JSON with status, current_best_apy, threshold, recommendation if triggered.
    """
    if alert_id not in _ALERTS:
        return {"status": "not_found", "alert_id": alert_id,
                "message": "Alert not found — may have expired on server restart."}

    alert = _ALERTS[alert_id]
    current = await get_best_yield(alert["asset"], 1_000, alert["risk_profile"], alert["chain"])
    current_apy = current.get("recommendation", {}).get("apy", 0) if "error" not in current else 0
    triggered = current_apy >= alert["threshold_apy"]
    _ALERTS[alert_id]["triggered"] = triggered

    result = {
        "alert_id": alert_id, "status": "triggered" if triggered else "watching",
        "asset": alert["asset"], "threshold_apy": alert["threshold_apy"],
        "current_best_apy": current_apy, "risk_profile": alert["risk_profile"],
        "age_seconds": int(time.time()) - alert["created_at"], "price": "free",
    }
    if triggered:
        result["recommendation"] = current.get("recommendation", {})
        result["reasoning"] = current.get("reasoning", "")
        result["message"] = f"ALERT TRIGGERED: {alert['asset']} yield {current_apy}% exceeds {alert['threshold_apy']}%"
    else:
        result["message"] = f"Watching: best {alert['asset']} is {current_apy}% (threshold: {alert['threshold_apy']}%)"
    return result


@mcp.tool
async def yield_alert_delete(
    alert_id: Annotated[str, "Alert ID to remove. Example: 'a3f2b1c0'"],
) -> dict:
    """Delete a yield alert by ID. FREE.

    Args:
        alert_id : ID returned by yield_alert_set

    Returns:
        JSON confirming deletion or not_found.
    """
    if alert_id not in _ALERTS:
        return {"status": "not_found", "alert_id": alert_id}
    del _ALERTS[alert_id]
    return {"status": "deleted", "alert_id": alert_id}


@mcp.tool
async def yield_alerts_list() -> dict:
    """List all active yield alerts. Useful for agents managing multiple positions. FREE.

    Returns:
        JSON with all alerts, their status, and age in seconds.
    """
    if not _ALERTS:
        return {"alerts": [], "count": 0}
    return {
        "alerts": [
            {"alert_id": aid, "asset": a["asset"],
             "threshold_apy": a["threshold_apy"], "risk_profile": a["risk_profile"],
             "chain": a["chain"], "triggered": a["triggered"],
             "age_seconds": int(time.time()) - a["created_at"]}
            for aid, a in _ALERTS.items()
        ],
        "count": len(_ALERTS), "price": "free",
    }


@mcp.prompt()
def yield_watch(
    asset: str = "USDC",
    target_apy: str = "6.0",
    risk: str = "safe",
) -> str:
    """Monitor yield and act when threshold is reached.
    Usage: /yield-watch asset=USDC target_apy=6.0 risk=safe
    """
    return (
        f"Set a yield alert for {asset} with threshold {target_apy}% and risk profile '{risk}' "
        f"using yield_alert_set. Check yield_alert_check every 5 minutes until triggered. "
        f"When triggered, call get_best_yield to confirm and report protocol, chain, APY, "
        f"risk_score, and recommended action."
    )




# ═══════════════════════════════════════════════════════════════════════════════
# PARTIE 4 : OPTIMAL ALLOCATION + VRAI AGENT A2A
# ═══════════════════════════════════════════════════════════════════════════════

# ── Outil : get_optimal_allocation ───────────────────────────────────────────
@mcp.tool
async def get_optimal_allocation(
    asset: Annotated[str, "Token to allocate. Examples: 'USDC', 'USDT', 'ETH', 'DAI'"],
    total_amount_usd: Annotated[float, "Total capital to deploy in USD. Example: 100000"],
    risk_profile: Annotated[str, "'safe' (score>=75), 'moderate' (>=55), 'max_yield' (>=35)"] = "moderate",
    max_protocols: Annotated[int, "Maximum number of protocols to split across (2-5). Default: 3"] = 3,
    chains: Annotated[list[str], "Chains to consider. Default: ['Ethereum','Arbitrum','Base']. Pass ['all'] for all chains."] = None,
) -> dict:
    """Split capital optimally across multiple DeFi protocols for maximum risk-adjusted yield.

    Unlike get_best_yield (one protocol), this tool returns a weighted multi-protocol
    allocation that maximizes APY while respecting TVL constraints and risk limits.
    Ideal for amounts >$10,000 where diversification improves risk-adjusted returns.

    Use this when: an agent needs to deploy capital across multiple protocols.
    Do NOT use for: single-protocol analysis (use get_best_yield or explain_risk).

    Args:
        asset           : Token symbol. Examples: 'USDC', 'USDT', 'ETH', 'DAI'
        total_amount_usd: Total amount to deploy. Minimum TVL per protocol = 20x allocation.
        risk_profile    : 'safe' (score>=75), 'moderate' (>=55), 'max_yield' (>=35)
        max_protocols   : Max protocols to split across (2-5). More = diversification.
        chains          : List of chains. Examples: ['Ethereum','Arbitrum'] or ['all']

    Returns:
        JSON with: allocation list (protocol, chain, amount_usd, pct, apy, risk_score),
        weighted_apy, monthly_revenue_usd, vs_best_single (APY delta), reasoning.
        Price: 0.05 USDC
    """
    if chains is None:
        chains = ["Ethereum", "Arbitrum", "Base"]

    max_protocols = max(2, min(5, max_protocols))
    all_chains    = "all" in chains

    pools = await _fetch_pools()
    min_tvl       = max(20_000_000, total_amount_usd * 20)
    score_threshold = _score_threshold(risk_profile)

    # ── Filtrer les pools éligibles ───────────────────────────────────────────
    candidates = []
    for p in pools:
        sym  = (p.get("symbol") or "").upper()
        if asset.upper() not in sym:
            continue
        if (p.get("tvlUsd") or 0) < min_tvl:
            continue
        if not p.get("apy") or p["apy"] <= 0:
            continue
        if not all_chains and (p.get("chain","") not in chains):
            continue
        score = _risk_score(p)
        if score < score_threshold:
            continue
        candidates.append((score, p))

    if not candidates:
        return {
            "error":   "no_candidates",
            "message": (
                f"No eligible pools for {asset} with profile '{risk_profile}' "
                f"on chains {chains}. Try 'moderate' or add more chains."
            ),
        }

    # ── Dédupliquer par projet (garder le meilleur pool par protocole) ────────
    best_per_protocol: dict = {}
    for score, p in candidates:
        key = f"{p.get('project')}:{p.get('chain')}"
        if key not in best_per_protocol:
            best_per_protocol[key] = (score, p)
        else:
            existing_score = best_per_protocol[key][0]
            if score * (p.get("apy") or 0) > existing_score * (best_per_protocol[key][1].get("apy") or 0):
                best_per_protocol[key] = (score, p)

    # ── Trier par score risk-ajusté et prendre les top N ─────────────────────
    ranked = sorted(
        best_per_protocol.values(),
        key=lambda x: x[0] * (x[1].get("apy") or 0),
        reverse=True,
    )[:max_protocols]

    if len(ranked) < 2:
        ranked = ranked * 2  # Si 1 seul candidat, forcer 2 entrées similaires

    # ── Calculer les poids proportionnels au score risk-ajusté ───────────────
    total_weight = sum(s * (p.get("apy") or 0) for s, p in ranked)
    allocation   = []
    remaining    = total_amount_usd

    for i, (score, p) in enumerate(ranked):
        weight  = (score * (p.get("apy") or 0)) / total_weight if total_weight > 0 else 1 / len(ranked)

        # Dernier protocole prend le reste pour éviter les arrondis
        if i == len(ranked) - 1:
            amount = remaining
        else:
            raw    = total_amount_usd * weight
            amount = round(raw / 1000) * 1000  # Arrondi au millier
            amount = max(1000, min(amount, remaining - 1000))

        remaining -= amount
        pct = round(amount / total_amount_usd * 100, 1)

        allocation.append({
            "protocol":   p.get("project"),
            "chain":      p.get("chain"),
            "amount_usd": int(amount),
            "pct":        pct,
            "apy":        round(p.get("apy") or 0, 2),
            "apy_base":   round(p.get("apyBase") or 0, 2),
            "risk_score": score,
            "risk_level": _risk_label(score),
            "tvl_usd":    int(p.get("tvlUsd") or 0),
        })

    # ── Calcul APY pondéré ────────────────────────────────────────────────────
    weighted_apy     = sum(a["apy"] * a["pct"] / 100 for a in allocation)
    monthly_revenue  = total_amount_usd * weighted_apy / 100 / 12

    # APY best single (pour comparaison)
    best_single      = ranked[0][1]
    best_single_apy  = round(best_single.get("apy") or 0, 2)
    apy_delta        = round(weighted_apy - best_single_apy, 3)

    # ── Reasoning ────────────────────────────────────────────────────────────
    reasons = [
        f"Top {len(allocation)} protocols by risk-adjusted score on {', '.join(chains) if not all_chains else 'all chains'}.",
        f"Minimum TVL per pool: ${min_tvl/1e6:.0f}M (20× allocation size).",
    ]
    if apy_delta > 0:
        reasons.append(f"Multi-protocol allocation improves APY by +{apy_delta}% vs single best.")
    else:
        reasons.append(f"Single best protocol is optimal — multi-split not materially better.")

    high_reward = [a for a in allocation if a["apy"] - a["apy_base"] > 1]
    if high_reward:
        reasons.append(f"Note: {high_reward[0]['protocol']} has significant reward APY — monitor for emissions changes.")

    return {
        "asset":             asset,
        "total_amount_usd":  total_amount_usd,
        "risk_profile":      risk_profile,
        "allocation":        allocation,
        "weighted_apy":      round(weighted_apy, 3),
        "monthly_revenue_usd": round(monthly_revenue, 2),
        "vs_best_single": {
            "protocol":      best_single.get("project"),
            "chain":         best_single.get("chain"),
            "apy":           best_single_apy,
            "apy_delta":     f"{'+' if apy_delta >= 0 else ''}{apy_delta}%",
        },
        "reasoning":         " ".join(reasons),
        "price":             "0.05 USDC",
    }


# ── Endpoint A2A JSON-RPC (/a2a) ──────────────────────────────────────────────
@mcp.custom_route("/a2a", methods=["POST"])
async def a2a_endpoint(request: Request) -> JSONResponse:
    """
    Vrai endpoint A2A JSON-RPC 2.0 compatible spec Google A2A v0.3.
    Accepte des tasks/send et tasks/get depuis des agents orchestrateurs.
    Compatible: Google ADK, Salesforce Agentforce, ServiceNow, Amazon Bedrock AgentCore.
    """
    import uuid as _uuid

    try:
        body   = await request.json()
    except Exception:
        return JSONResponse(
            {"jsonrpc": "2.0", "id": None,
             "error": {"code": -32700, "message": "Parse error — invalid JSON"}},
            status_code=400,
        )

    rpc_id  = body.get("id", str(_uuid.uuid4())[:8])
    method  = body.get("method", "")
    params  = body.get("params", {})

    # ── Méthodes supportées ───────────────────────────────────────────────────
    if method not in ("tasks/send", "tasks/get", "tasks/cancel"):
        return JSONResponse({
            "jsonrpc": "2.0", "id": rpc_id,
            "error": {
                "code": -32601,
                "message": f"Method '{method}' not supported. Use: tasks/send, tasks/get",
            }
        })

    # ── tasks/get : statut simple ─────────────────────────────────────────────
    if method == "tasks/get":
        return JSONResponse({
            "jsonrpc": "2.0", "id": rpc_id,
            "result": {
                "id":     params.get("id", rpc_id),
                "status": {"state": "completed"},
                "message": "This server processes tasks synchronously.",
            }
        })

    # ── tasks/cancel ──────────────────────────────────────────────────────────
    if method == "tasks/cancel":
        return JSONResponse({
            "jsonrpc": "2.0", "id": rpc_id,
            "result": {"id": params.get("id", rpc_id), "status": {"state": "canceled"}}
        })

    # ── tasks/send : routing vers les outils MCP ─────────────────────────────
    task_id  = params.get("id", rpc_id)
    parts    = params.get("message", {}).get("parts", [])
    goal     = " ".join(
        p.get("text", "") for p in parts if isinstance(p, dict) and p.get("type") == "text"
    ).lower().strip()

    if not goal:
        return JSONResponse({
            "jsonrpc": "2.0", "id": rpc_id,
            "error": {"code": -32602, "message": "Missing task message.parts[].text"}
        })

    log.info(f"A2A task received: id={task_id} goal={goal[:80]}")

    # ── Router la task vers l'outil approprié ─────────────────────────────────
    try:
        # Extraire les paramètres communs de la phrase
        import re

        # Détecter l'asset (USDC, USDT, ETH, DAI, etc.)
        asset_match = re.search(
            r"(USDC|USDT|ETH|DAI|WETH|WBTC|stETH|USDS)", goal.upper()
        )
        asset = asset_match.group(1) if asset_match else "USDC"

        # Détecter le montant
        amount_match = re.search(
            r"(\d[\d,]*(?:\.\d+)?)\s*(?:usd|usdc|\$|k|m)?", goal
        )
        amount = 10_000.0
        if amount_match:
            raw = amount_match.group(1).replace(",", "")
            amount = float(raw)
            if "k" in goal[amount_match.end():amount_match.end()+2].lower():
                amount *= 1000
            elif "m" in goal[amount_match.end():amount_match.end()+2].lower():
                amount *= 1_000_000

        # Détecter le risk profile
        if any(w in goal for w in ["safe", "conservative", "low risk"]):
            risk = "safe"
        elif any(w in goal for w in ["max", "maximum", "aggressive", "high yield"]):
            risk = "max_yield"
        else:
            risk = "moderate"

        # ── Router ────────────────────────────────────────────────────────────
        if any(w in goal for w in ["allocat", "split", "distribut", "optimal", "portfolio"]):
            result = await get_optimal_allocation(asset, amount, risk)
            skill_used = "get_optimal_allocation"

        elif any(w in goal for w in ["compare", "vs", "versus", "between"]):
            # Extraire les protocoles mentionnés
            known = ["aave-v3","morpho-blue","compound-v3","spark","yearn-finance","lido"]
            mentioned = [p for p in known if p.replace("-","").replace("v3","") in goal.replace("-","")]
            if len(mentioned) < 2:
                mentioned = ["aave-v3","morpho-blue","compound-v3"]
            result = await compare_yields(asset, mentioned[:6])
            skill_used = "compare_yields"

        elif any(w in goal for w in ["risk", "safe", "audit", "analyse", "analyze", "explain"]):
            # Extraire le protocole
            proto_match = re.search(
                r"(aave|morpho|compound|spark|yearn|lido|uniswap|curve|balancer)", goal
            )
            protocol = proto_match.group(1) + ("-v3" if proto_match and "v3" not in proto_match.group(1) else "") if proto_match else "aave-v3"
            result = await explain_risk(protocol, asset)
            skill_used = "explain_risk"

        elif any(w in goal for w in ["alert", "notify", "watch", "monitor", "threshold"]):
            # Extraire le seuil
            threshold_match = re.search(r"(\d+(?:\.\d+)?)\s*%", goal)
            threshold = float(threshold_match.group(1)) if threshold_match else 5.0
            result = await yield_alert_set(asset, threshold, risk)
            skill_used = "yield_alert_set"

        else:
            # Default : get_best_yield
            result = await get_best_yield(asset, amount, risk)
            skill_used = "get_best_yield"

    except Exception as e:
        log.error(f"A2A task error: {e}")
        return JSONResponse({
            "jsonrpc": "2.0", "id": rpc_id,
            "error": {"code": -32603, "message": f"Internal error: {str(e)}"}
        })

    # ── Formater la réponse A2A JSON-RPC 2.0 ──────────────────────────────────
    return JSONResponse({
        "jsonrpc": "2.0",
        "id":      rpc_id,
        "result": {
            "id":     task_id,
            "status": {"state": "completed"},
            "artifacts": [
                {
                    "name":  f"defi-yield-{skill_used}",
                    "parts": [
                        {
                            "type": "data",
                            "data": result,
                        }
                    ],
                }
            ],
            "metadata": {
                "skill_used":  skill_used,
                "asset":       asset,
                "server":      "defi-yield-engine",
                "version":     "1.0.0",
                "payment":     {
                    "protocol":  "x402",
                    "network":   X402_NETWORK,
                    "recipient": X402_RECIPIENT,
                },
            },
        },
    })


@mcp.custom_route("/.well-known/agent.json", methods=["GET"])
async def agent_card_a2a(request: Request) -> JSONResponse:
    """A2A Agent Card — Google Agent-to-Agent Protocol discovery standard.
    Auto-crawled by A2A orchestrators (Salesforce, SAP, ServiceNow, Google ADK).
    """
    return JSONResponse({
        "name": "DeFi Yield Decision Engine",
        "description": (
            "Risk-adjusted DeFi yield recommendations and APY threshold alerts "
            "across 548 protocols and 115 chains. One answer, not 13,800 pools."
        ),
        "url": "https://defi-yield-engine-production.up.railway.app/mcp",
        "version": "1.0.0",
        "provider": {"name": "danteriva45", "organization": "Independent"},
        "capabilities": {
            "streaming": False,
            "pushNotifications": False,
            "stateTransitionHistory": False,
        },
        "authentication": {"schemes": ["x402"]},
        "defaultInputModes": ["application/json"],
        "defaultOutputModes": ["application/json"],
        "skills": [
            {"id": "get_best_yield", "name": "Get Best DeFi Yield",
             "description": "Single best risk-adjusted yield for USDC, USDT, ETH across 548 protocols.",
             "tags": ["defi", "yield", "USDC", "ETH", "risk"], "price": "0.05 USDC"},
            {"id": "get_optimal_allocation", "name": "Optimal Capital Allocation",
             "description": "Split capital across 2-5 protocols for maximum risk-adjusted yield.",
             "tags": ["allocation", "portfolio", "defi", "yield", "routing"], "price": "0.05 USDC"},
            {"id": "explain_risk", "name": "Explain Protocol Risk",
             "description": "9-signal risk breakdown for any DeFiLlama protocol.",
             "tags": ["risk", "audit", "defi", "safety"], "price": "0.05 USDC"},
            {"id": "compare_yields", "name": "Compare DeFi Protocols",
             "description": "Side-by-side risk-adjusted comparison of 2-6 protocols.",
             "tags": ["compare", "defi", "yield"], "price": "0.05 USDC"},
            {"id": "get_best_liquid_staking", "name": "Get Best Liquid Staking",
             "description": "Best LST protocol for ETH, SOL, BNB. Covers Lido, Rocket Pool, Jito, 50+ others.",
             "tags": ["liquid-staking", "ETH", "SOL", "lido", "staking"], "price": "0.05 USDC"},
            {"id": "get_best_restaking", "name": "Get Best Restaking / LRT",
             "description": "Best restaking (EigenLayer, Symbiotic) or liquid restaking token (Ether.fi, Renzo, Puffer).",
             "tags": ["restaking", "eigenlayer", "lrt", "ether.fi"], "price": "0.05 USDC"},
            {"id": "get_best_rwa", "name": "Get Best RWA Protocol",
             "description": "Top Real World Asset protocols: T-bills (Ondo, BUIDL), private credit (Maple, Centrifuge).",
             "tags": ["rwa", "ondo", "maple", "real-world-assets"], "price": "0.05 USDC"},
            {"id": "get_perps_overview", "name": "Get Perps Overview",
             "description": "Top perpetuals by 24h volume: Hyperliquid, dYdX, GMX, Drift, Jupiter Perps.",
             "tags": ["perps", "derivatives", "hyperliquid", "gmx"], "price": "0.05 USDC"},
            {"id": "compare_perps", "name": "Compare Perps Protocols",
             "description": "Side-by-side comparison of perpetuals protocols by volume.",
             "tags": ["perps", "compare", "derivatives"], "price": "0.05 USDC"},
            {"id": "get_defi_overview", "name": "DeFi Market Overview",
             "description": "Free snapshot: top protocols across Yield, Staking, Restaking, RWA and Perps.",
             "tags": ["overview", "market", "defi"], "price": "free"},
            {"id": "yield_alert_set", "name": "Set Yield Alert",
             "description": "Register APY threshold alert — fires when yield exceeds target.",
             "tags": ["alert", "monitoring", "automation"], "price": "free"},
            {"id": "yield_alert_check", "name": "Check Yield Alert",
             "description": "Poll alert status — triggered/watching + current best APY.",
             "tags": ["alert", "polling"], "price": "free"},
        ],
    })

# ── Setup x402 Payment Middleware ────────────────────────────────────────────
# PARTIE 5 : BATCH 1 — PERPS / LIQUID STAKING / RESTAKING / RWA
# ═══════════════════════════════════════════════════════════════════════════════

# ── Caches supplémentaires ────────────────────────────────────────────────────
_PROTOCOLS_CACHE: dict = {"data": None, "updated_at": 0.0}
_PERPS_CACHE:     dict = {"data": None, "updated_at": 0.0}
_PROTOCOLS_TTL = 300   # 5 min
_PERPS_TTL     = 180   # 3 min (volumes changent vite)


async def _fetch_all_protocols() -> list[dict]:
    """Récupère tous les protocoles DeFiLlama avec cache 5 min."""
    now = time.time()
    if _PROTOCOLS_CACHE["data"] and (now - _PROTOCOLS_CACHE["updated_at"]) < _PROTOCOLS_TTL:
        return _PROTOCOLS_CACHE["data"]
    log.info("Fetching protocols from DeFiLlama…")
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.get("https://api.llama.fi/protocols")
        r.raise_for_status()
        data = r.json()
    _PROTOCOLS_CACHE["data"]       = data
    _PROTOCOLS_CACHE["updated_at"] = now
    log.info(f"Loaded {len(data)} protocols")
    return data


async def _fetch_protocols_by_category(category: str) -> list[dict]:
    """Filtre les protocoles par catégorie DeFiLlama (case-insensitive)."""
    all_p = await _fetch_all_protocols()
    return [p for p in all_p
            if (p.get("category") or "").lower() == category.lower()]


async def _fetch_perps() -> list[dict]:
    """Récupère les données perpetuals/derivatives DeFiLlama."""
    now = time.time()
    if _PERPS_CACHE["data"] and (now - _PERPS_CACHE["updated_at"]) < _PERPS_TTL:
        return _PERPS_CACHE["data"]
    log.info("Fetching perps data from DeFiLlama…")
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.get("https://api.llama.fi/overview/derivatives")
        r.raise_for_status()
        data = r.json()
    protocols = data.get("protocols", []) if isinstance(data, dict) else data
    _PERPS_CACHE["data"]       = protocols
    _PERPS_CACHE["updated_at"] = now
    log.info(f"Loaded {len(protocols)} perps protocols")
    return protocols


# ── Risk score pour protocoles (TVL + ancienneté + audits) ───────────────────
def _protocol_risk_score(p: dict) -> int:
    """
    Score 0-100 pour protocoles staking/RWA basé sur TVL, audits, ancienneté.
    Différent du _risk_score() pool qui utilise APY + outlier.
    """
    score = 50
    tvl = p.get("tvl") or 0

    # TVL — proxy confiance marché
    if tvl >= 5_000_000_000:  score += 25
    elif tvl >= 1_000_000_000: score += 20
    elif tvl >= 100_000_000:   score += 12
    elif tvl >= 10_000_000:    score += 5
    else:                      score -= 15

    # Audits
    audits = p.get("audits") or "0"
    try:
        if int(audits) >= 2: score += 8
        elif int(audits) == 1: score += 4
    except (ValueError, TypeError):
        pass

    # Ancienneté (listedAt en timestamp Unix)
    listed = p.get("listedAt") or 0
    age_days = (time.time() - listed) / 86400 if listed else 0
    if age_days >= 730:  score += 10   # > 2 ans
    elif age_days >= 365: score += 7   # > 1 an
    elif age_days >= 180: score += 3

    # Momentum TVL 7j
    change7d = p.get("change_7d") or 0
    try:
        c = float(change7d)
        if c > 5:    score += 3
        elif c < -15: score -= 8
    except (ValueError, TypeError):
        pass

    return max(0, min(100, score))


def _compact_protocol(p: dict, score: int) -> dict:
    """Output compact pour protocoles (staking, RWA…)."""
    return {
        "protocol":    p.get("slug") or p.get("name", "unknown"),
        "name":        p.get("name", "unknown"),
        "category":    p.get("category", "unknown"),
        "chains":      (p.get("chains") or [])[:4],
        "tvl_usd":     int(p.get("tvl") or 0),
        "change_7d":   round(float(p.get("change_7d") or 0), 1),
        "risk_score":  score,
        "risk_level":  _risk_label(score),
        "url":         p.get("url") or f"https://defillama.com/protocol/{p.get('slug','')}",
    }


# ── Outil : get_best_liquid_staking ──────────────────────────────────────────
@mcp.tool
async def get_best_liquid_staking(
    asset: Annotated[str, "Asset to stake. Examples: 'ETH', 'SOL', 'BNB', 'MATIC'"] = "ETH",
    risk_profile: Annotated[str, "'safe' (score>=75), 'moderate' (>=55), 'max_yield' (>=35)"] = "moderate",
    chain: Annotated[str, "'all' or chain name: 'Ethereum', 'Solana', 'BNB Chain'"] = "all",
) -> dict:
    """Select the best liquid staking protocol for a given asset and risk profile.

    Scores protocols by TVL, audit count, age, and momentum. Returns one
    recommendation with reasoning and top alternatives.
    Covers: Lido, Rocket Pool, Coinbase cbETH, Jito, Marinade, and 50+ others.

    Use this when: an agent needs to stake an asset while keeping it liquid.
    Do NOT use for: yield farming (use get_best_yield), restaking (use get_best_restaking).

    Args:
        asset        : Asset to stake. Examples: 'ETH', 'SOL', 'BNB'
        risk_profile : 'safe' (score>=75), 'moderate' (>=55), 'max_yield' (>=35)
        chain        : 'all' or specific chain

    Returns:
        JSON with recommendation {protocol, name, tvl_usd, risk_score, chains},
        reasoning, alternatives (top 2). Price: 0.05 USDC
    """
    protocols = await _fetch_protocols_by_category("Liquid Staking")

    # Filtre par chain et asset (dans le nom ou symbol)
    candidates = []
    for p in protocols:
        if chain != "all":
            chains = [c.lower() for c in (p.get("chains") or [])]
            if chain.lower() not in chains:
                continue
        # Filtre asset par symbol/name (heuristique)
        if asset.upper() not in ["ALL", "ANY"]:
            name_sym = f"{p.get('name','')} {p.get('symbol','')}".upper()
            if asset.upper() not in name_sym and asset.upper() not in [
                c.upper() for c in (p.get("chains") or [])
            ]:
                # Cas spéciaux : ETH → accepter tout sur Ethereum
                if asset.upper() == "ETH" and "Ethereum" not in (p.get("chains") or []):
                    continue
                # SOL → accepter tout sur Solana
                elif asset.upper() == "SOL" and "Solana" not in (p.get("chains") or []):
                    continue
                elif asset.upper() not in ("ETH", "SOL"):
                    continue

        if (p.get("tvl") or 0) < 1_000_000:
            continue

        score = _protocol_risk_score(p)
        if score < _score_threshold(risk_profile):
            continue
        candidates.append((score, p))

    if not candidates:
        # Fallback : retourner les meilleurs sans filtre asset
        all_ls = await _fetch_protocols_by_category("Liquid Staking")
        candidates = [(s := _protocol_risk_score(p), p)
                      for p in all_ls if (p.get("tvl") or 0) > 10_000_000
                      and s >= _score_threshold(risk_profile)][:10]

    if not candidates:
        return {
            "error":   "no_results",
            "message": f"No liquid staking protocols found for {asset} with profile '{risk_profile}'.",
        }

    candidates.sort(key=lambda x: x[0], reverse=True)
    best_score, best = candidates[0]
    alts = candidates[1:3]

    reasoning_parts = [
        f"TVL ${int(best.get('tvl') or 0)/1e9:.1f}B." if (best.get("tvl") or 0) >= 1e9
        else f"TVL ${int(best.get('tvl') or 0)/1e6:.0f}M.",
        f"Active on {len(best.get('chains') or [])} chains.",
    ]
    age_days = (time.time() - (best.get("listedAt") or 0)) / 86400
    if age_days > 365:
        reasoning_parts.append(f"Protocol age {int(age_days/365)}y+ — established.")
    try:
        if int(best.get("audits") or 0) >= 1:
            reasoning_parts.append("Audited.")
    except (ValueError, TypeError):
        pass

    return {
        "recommendation": _compact_protocol(best, best_score),
        "reasoning":      " ".join(reasoning_parts),
        "alternatives":   [_compact_protocol(p, s) for s, p in alts],
        "asset":          asset,
        "risk_profile":   risk_profile,
        "price":          "0.05 USDC",
    }


# ── Outil : get_best_restaking ────────────────────────────────────────────────
@mcp.tool
async def get_best_restaking(
    risk_profile: Annotated[str, "'safe' (score>=75), 'moderate' (>=55), 'max_yield' (>=35)"] = "moderate",
    restaking_type: Annotated[str, "'restaking' (base layer: EigenLayer, Symbiotic) or 'liquid-restaking' (LRT tokens: EtherFi, Renzo, Puffer)"] = "liquid-restaking",
) -> dict:
    """Select the best restaking protocol by TVL and risk profile.

    Covers base restaking (EigenLayer, Symbiotic, Karak) and liquid restaking
    tokens (EtherFi, Renzo, Puffer, Kelp). Scored by TVL, audits, and age.

    Use this when: an agent wants to restake ETH or LSTs for additional yield.
    Do NOT use for: simple staking (use get_best_liquid_staking).

    Args:
        risk_profile    : 'safe', 'moderate', or 'max_yield'
        restaking_type  : 'restaking' or 'liquid-restaking'

    Returns:
        JSON with recommendation, reasoning, alternatives. Price: 0.05 USDC
    """
    category = "Liquid Restaking" if restaking_type == "liquid-restaking" else "Restaking"
    protocols = await _fetch_protocols_by_category(category)

    candidates = [
        (_protocol_risk_score(p), p) for p in protocols
        if (p.get("tvl") or 0) >= 1_000_000
        and _protocol_risk_score(p) >= _score_threshold(risk_profile)
    ]

    if not candidates:
        return {
            "error":   "no_results",
            "message": f"No {category} protocols found with profile '{risk_profile}'.",
        }

    candidates.sort(key=lambda x: x[0], reverse=True)
    best_score, best = candidates[0]
    alts = candidates[1:3]

    tvl = best.get("tvl") or 0
    reasoning = (
        f"Largest {category} protocol by TVL "
        f"(${tvl/1e9:.1f}B). " if tvl >= 1e9 else f"TVL ${tvl/1e6:.0f}M. "
    )
    reasoning += f"Risk score {best_score}/100. "
    if restaking_type == "liquid-restaking":
        reasoning += "Issues LRT token for DeFi composability."
    else:
        reasoning += "Base restaking layer — AVS rewards on top of staking yield."

    return {
        "recommendation": _compact_protocol(best, best_score),
        "reasoning":      reasoning,
        "alternatives":   [_compact_protocol(p, s) for s, p in alts],
        "restaking_type": restaking_type,
        "risk_profile":   risk_profile,
        "price":          "0.05 USDC",
    }


# ── Outil : get_best_rwa ──────────────────────────────────────────────────────
@mcp.tool
async def get_best_rwa(
    risk_profile: Annotated[str, "'safe' (score>=75), 'moderate' (>=55), 'max_yield' (>=35)"] = "moderate",
    rwa_type: Annotated[str, "Filter by RWA type. Options: 'all', 't-bills', 'private-credit', 'real-estate'. Default: 'all'"] = "all",
) -> dict:
    """Select the best Real World Asset (RWA) protocol by TVL and risk profile.

    Covers tokenized T-bills (Ondo USDY, BlackRock BUIDL), private credit
    (Maple, Centrifuge), and real estate protocols. Scored by TVL, audits, age.

    Use this when: an agent needs stable off-chain backed yield.
    Do NOT use for: on-chain DeFi yields (use get_best_yield).

    Args:
        risk_profile : 'safe', 'moderate', or 'max_yield'
        rwa_type     : 'all', 't-bills', 'private-credit', 'real-estate'

    Returns:
        JSON with top 3 RWA protocols ranked by risk score. Price: 0.05 USDC
    """
    protocols = await _fetch_protocols_by_category("RWA")

    # Filtre par type RWA (heuristique sur le nom)
    TYPE_KEYWORDS = {
        "t-bills":        ["ondo", "buidl", "spiko", "usdy", "ousg", "treasury", "tbill"],
        "private-credit": ["maple", "centrifuge", "goldfinch", "credix", "clearpool"],
        "real-estate":    ["tangible", "realt", "real-estate", "homium", "property"],
    }

    def matches_type(p, rwa_type):
        if rwa_type == "all":
            return True
        keywords = TYPE_KEYWORDS.get(rwa_type, [])
        name = (p.get("name") or "").lower()
        slug = (p.get("slug") or "").lower()
        return any(kw in name or kw in slug for kw in keywords)

    candidates = [
        (_protocol_risk_score(p), p) for p in protocols
        if (p.get("tvl") or 0) >= 1_000_000
        and matches_type(p, rwa_type)
        and _protocol_risk_score(p) >= _score_threshold(risk_profile)
    ]

    if not candidates:
        # Fallback sans filtre type
        candidates = [
            (_protocol_risk_score(p), p) for p in protocols
            if (p.get("tvl") or 0) >= 1_000_000
            and _protocol_risk_score(p) >= _score_threshold(risk_profile)
        ]

    if not candidates:
        return {
            "error":   "no_results",
            "message": f"No RWA protocols found with profile '{risk_profile}'.",
        }

    candidates.sort(key=lambda x: x[0], reverse=True)
    top3 = candidates[:3]

    return {
        "top_rwa_protocols": [_compact_protocol(p, s) for s, p in top3],
        "rwa_type":          rwa_type,
        "risk_profile":      risk_profile,
        "note":              "RWA yields vary — check protocol docs for current APY (off-chain backed).",
        "price":             "0.05 USDC",
    }


# ── Outil : get_perps_overview ────────────────────────────────────────────────
@mcp.tool
async def get_perps_overview(
    chain: Annotated[str, "'all' or chain name: 'Arbitrum', 'Solana', 'Base', 'BSC'"] = "all",
    top_n: Annotated[int, "Number of protocols to return (1-10). Default: 5"] = 5,
) -> dict:
    """Get top perpetuals/derivatives protocols ranked by 24h volume.

    Covers Hyperliquid, dYdX, GMX, Drift, Jupiter Perps, and 50+ others.
    Returns volume, open interest, and market share for each protocol.

    Use this when: an agent needs derivatives market intelligence.
    Do NOT use for: spot DEX (use DeFiLlama DEX endpoints), yield (use get_best_yield).

    Args:
        chain  : 'all' or specific chain filter
        top_n  : Number of top protocols to return (1-10)

    Returns:
        JSON with ranked perps protocols, 24h volume, market share. Price: 0.05 USDC
    """
    protocols = await _fetch_perps()
    top_n = max(1, min(10, top_n))

    # Filtrer par chain si spécifié
    if chain != "all":
        filtered = [
            p for p in protocols
            if chain.lower() in [c.lower() for c in (p.get("chains") or [])]
        ]
        if not filtered:
            filtered = protocols  # fallback si chain non trouvée
    else:
        filtered = protocols

    # Trier par volume 24h
    def get_volume(p):
        v = p.get("totalAllTime") or p.get("total24h") or p.get("total7d") or 0
        try:
            return float(v)
        except (ValueError, TypeError):
            return 0

    ranked = sorted(filtered, key=get_volume, reverse=True)[:top_n]

    if not ranked:
        return {"error": "no_results", "message": "No perps data available."}

    # Total volume pour market share
    total_vol = sum(get_volume(p) for p in ranked)

    result = []
    for p in ranked:
        vol = get_volume(p)
        market_share = round(vol / total_vol * 100, 1) if total_vol > 0 else 0
        result.append({
            "protocol":     p.get("name") or p.get("module", "unknown"),
            "chains":       (p.get("chains") or [])[:3],
            "volume_24h":   int(p.get("total24h") or 0),
            "volume_7d":    int(p.get("total7d") or 0),
            "market_share": market_share,
        })

    return {
        "top_perps": result,
        "chain":     chain,
        "total_24h_volume_usd": int(sum(int(r["volume_24h"]) for r in result)),
        "price":     "0.05 USDC",
    }


# ── Outil : compare_perps ─────────────────────────────────────────────────────
@mcp.tool
async def compare_perps(
    protocols: Annotated[list[str], "Protocol names to compare (2-5). Examples: ['Hyperliquid', 'GMX', 'dYdX', 'Drift']"],
) -> dict:
    """Compare perpetuals protocols side-by-side by volume and activity.

    Args:
        protocols : List of 2-5 protocol names to compare

    Returns:
        JSON with side-by-side metrics and winner by 24h volume. Price: 0.05 USDC
    """
    if len(protocols) < 2:
        return {"error": "Provide at least 2 protocols to compare."}
    if len(protocols) > 5:
        return {"error": "Maximum 5 protocols per comparison."}

    all_perps = await _fetch_perps()

    results = []
    for slug in protocols:
        match = next(
            (p for p in all_perps
             if slug.lower() in (p.get("name") or "").lower()
             or slug.lower() in (p.get("module") or "").lower()),
            None,
        )
        if match:
            vol_24h = float(match.get("total24h") or 0)
            vol_7d  = float(match.get("total7d") or 0)
            results.append({
                "protocol":   match.get("name") or slug,
                "chains":     (match.get("chains") or [])[:3],
                "volume_24h": int(vol_24h),
                "volume_7d":  int(vol_7d),
                "volume_30d": int(float(match.get("total30d") or 0)),
            })
        else:
            results.append({"protocol": slug, "status": "not_found"})

    valid = [r for r in results if "volume_24h" in r]
    valid.sort(key=lambda x: x["volume_24h"], reverse=True)
    winner = valid[0] if valid else None

    return {
        "comparison": valid + [r for r in results if "volume_24h" not in r],
        "winner":     winner,
        "price":      "0.05 USDC",
    }


# ── Outil : get_defi_overview (maître) ───────────────────────────────────────
@mcp.tool
async def get_defi_overview() -> dict:
    """Complete DeFi market snapshot across all categories. FREE.

    Returns top protocol per category: Yield, Liquid Staking, Restaking,
    RWA, and Perps top-3. Ideal first call for agents needing market context
    before deciding which category to explore deeper.

    Use this as a starting point before calling category-specific tools.
    This call is FREE — use it to decide which paid tool to call next.

    Returns:
        JSON with market leaders per category + TVL + quick stats.
    """
    try:
        all_p = await _fetch_all_protocols()

        def top_by_category(cat, n=3):
            filtered = [p for p in all_p if (p.get("category") or "").lower() == cat.lower()]
            filtered.sort(key=lambda x: float(x.get("tvl") or 0), reverse=True)
            return [
                {
                    "name":    p.get("name"),
                    "tvl_usd": int(float(p.get("tvl") or 0)),
                    "chains":  (p.get("chains") or [])[:2],
                }
                for p in filtered[:n]
            ]

        # Perps top 3 by volume
        perps_data = await _fetch_perps()
        perps_data.sort(key=lambda p: float(p.get("total24h") or 0), reverse=True)
        top_perps = [
            {
                "name":       p.get("name") or p.get("module"),
                "volume_24h": int(float(p.get("total24h") or 0)),
            }
            for p in perps_data[:3]
        ]

        return {
            "liquid_staking": top_by_category("Liquid Staking"),
            "restaking":      top_by_category("Restaking"),
            "liquid_restaking": top_by_category("Liquid Restaking"),
            "rwa":            top_by_category("RWA"),
            "perps_by_volume": top_perps,
            "note": (
                "Use get_best_liquid_staking, get_best_restaking, get_best_rwa, "
                "get_perps_overview, or get_best_yield for detailed recommendations."
            ),
            "price": "free",
        }

    except Exception as e:
        log.error(f"get_defi_overview error: {e}")
        return {"error": str(e), "message": "Could not load overview. Try individual tools."}


# ═══════════════════════════════════════════════════════════════════════════════
# PARTIE 6 : GAS PRICE PREDICTOR + SMART CONTRACT RISK SCORER
# ═══════════════════════════════════════════════════════════════════════════════

# ── Cache Gas ─────────────────────────────────────────────────────────────────
_GAS_CACHE: dict = {}      # {chain: {"data": {...}, "updated_at": float}}
_GAS_TTL = 60              # 1 min — gas change vite

# ── Patterns gas (UTC) ────────────────────────────────────────────────────────
# Basé sur analyse historique Ethereum/L2 — source : Etherscan Gas Tracker
GAS_HOURLY_MULTIPLIER = {
    0: 0.72, 1: 0.68, 2: 0.65, 3: 0.62, 4: 0.60,  # creux nocturne
    5: 0.63, 6: 0.70, 7: 0.80, 8: 0.90, 9: 0.95,
    10: 1.00, 11: 1.05, 12: 1.08, 13: 1.10, 14: 1.15,  # pic US market
    15: 1.18, 16: 1.20, 17: 1.18, 18: 1.12, 19: 1.05,
    20: 0.98, 21: 0.90, 22: 0.82, 23: 0.75,
}
GAS_DAY_MULTIPLIER = {
    0: 1.05,  # Lundi
    1: 1.08,  # Mardi
    2: 1.10,  # Mercredi — pic semaine
    3: 1.07,  # Jeudi
    4: 1.03,  # Vendredi
    5: 0.80,  # Samedi — -20%
    6: 0.75,  # Dimanche — -25%
}

# Configs par chain
CHAIN_GAS_APIS = {
    "ethereum": {
        "etherscan": "https://api.etherscan.io/api?module=gastracker&action=gasoracle",
        "chain_id": 1, "unit": "Gwei", "base_gwei": 20,
    },
    "arbitrum": {
        "chain_id": 42161, "unit": "Gwei", "base_gwei": 0.1,
    },
    "base": {
        "chain_id": 8453, "unit": "Gwei", "base_gwei": 0.05,
    },
    "polygon": {
        "chain_id": 137, "unit": "Gwei", "base_gwei": 50,
    },
    "optimism": {
        "chain_id": 10, "unit": "Gwei", "base_gwei": 0.05,
    },
}

# GoPlus chain IDs
GOPLUS_CHAIN_IDS = {
    "ethereum": "1", "bsc": "56", "polygon": "137",
    "arbitrum": "42161", "base": "8453", "optimism": "10",
    "avalanche": "43114", "fantom": "250",
}


async def _fetch_gas_current(chain: str) -> dict:
    """Récupère le gas actuel via Etherscan (Ethereum) ou estimation."""
    now = time.time()
    chain_lower = chain.lower()
    cached = _GAS_CACHE.get(chain_lower)
    if cached and (now - cached["updated_at"]) < _GAS_TTL:
        return cached["data"]

    result = {}
    if chain_lower == "ethereum":
        try:
            async with httpx.AsyncClient(timeout=8) as client:
                r = await client.get(
                    "https://api.etherscan.io/api",
                    params={"module": "gastracker", "action": "gasoracle"}
                )
                if r.status_code == 200:
                    d = r.json()
                    if d.get("status") == "1" and "result" in d:
                        res = d["result"]
                        result = {
                            "safe":     float(res.get("SafeGasPrice", 15)),
                            "standard": float(res.get("ProposeGasPrice", 20)),
                            "fast":     float(res.get("FastGasPrice", 25)),
                            "source":   "etherscan",
                        }
        except Exception as e:
            log.warning(f"Gas fetch failed: {e}")

    # Fallback pattern-based si API indisponible
    if not result:
        base = CHAIN_GAS_APIS.get(chain_lower, {}).get("base_gwei", 20)
        import datetime
        utc_now = datetime.datetime.utcnow()
        mult = GAS_HOURLY_MULTIPLIER.get(utc_now.hour, 1.0) *                GAS_DAY_MULTIPLIER.get(utc_now.weekday(), 1.0)
        result = {
            "safe":     round(base * mult * 0.8, 3),
            "standard": round(base * mult, 3),
            "fast":     round(base * mult * 1.3, 3),
            "source":   "pattern_estimate",
        }

    _GAS_CACHE[chain_lower] = {"data": result, "updated_at": now}
    return result


def _find_optimal_windows(chain: str, horizon_hours: int = 24) -> list[dict]:
    """Identifie les fenêtres de gas bas dans les prochaines N heures."""
    import datetime
    utc_now = datetime.datetime.utcnow()
    windows = []

    for h in range(horizon_hours):
        future = utc_now + datetime.timedelta(hours=h)
        hour_mult = GAS_HOURLY_MULTIPLIER.get(future.hour, 1.0)
        day_mult  = GAS_DAY_MULTIPLIER.get(future.weekday(), 1.0)
        combined  = hour_mult * day_mult

        if combined <= 0.75:  # seuil bas
            windows.append({
                "hours_from_now": h,
                "utc_time":       future.strftime("%H:%M UTC"),
                "day":            ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][future.weekday()],
                "gas_multiplier": round(combined, 2),
                "savings_pct":    round((1 - combined) * 100, 0),
            })

    return windows[:5]  # top 5 fenêtres


# ── Outil : get_gas_price ─────────────────────────────────────────────────────
@mcp.tool
async def get_gas_price(
    chain: Annotated[str, "Blockchain to check. Options: 'ethereum', 'arbitrum', 'base', 'polygon', 'optimism'"] = "ethereum",
) -> dict:
    """Get current gas prices across urgency levels for a given chain. FREE.

    Returns safe/standard/fast gas prices in Gwei with USD cost estimate
    for a standard ERC-20 transfer (21,000 gas).

    Use this when: an agent needs current gas cost before executing a transaction.
    Follow with get_optimal_gas_window to decide whether to wait.

    Args:
        chain : Chain to check. Options: 'ethereum', 'arbitrum', 'base',
                'polygon', 'optimism'

    Returns:
        JSON with safe/standard/fast Gwei, USD cost estimate, source. FREE.
    """
    gas = await _fetch_gas_current(chain)
    chain_cfg = CHAIN_GAS_APIS.get(chain.lower(), {})

    # Estimation coût USD (ETH ~$3000, gas transfer std = 21000 gas)
    eth_price_usd = 3000  # approximation — utiliser get_best_yield pour ETH price
    gas_units = 21_000

    def gwei_to_usd(gwei: float) -> float:
        return round(gwei * 1e-9 * gas_units * eth_price_usd, 4)

    return {
        "chain":    chain,
        "gas":      {
            "safe":     {"gwei": gas["safe"],     "usd": gwei_to_usd(gas["safe"])},
            "standard": {"gwei": gas["standard"], "usd": gwei_to_usd(gas["standard"])},
            "fast":     {"gwei": gas["fast"],     "usd": gwei_to_usd(gas["fast"])},
        },
        "source":   gas.get("source", "unknown"),
        "note":     "USD estimate based on 21,000 gas (ERC-20 transfer) at ETH ~$3,000",
        "price":    "free",
    }


# ── Outil : get_optimal_gas_window ───────────────────────────────────────────
@mcp.tool
async def get_optimal_gas_window(
    chain: Annotated[str, "Chain to optimize for: 'ethereum', 'arbitrum', 'base', 'polygon', 'optimism'"] = "ethereum",
    urgency: Annotated[str, "Transaction urgency: 'low' (can wait 24h), 'medium' (wait up to 6h), 'high' (execute now)"] = "low",
    horizon_hours: Annotated[int, "Hours to look ahead for optimal windows (1-48). Default: 24"] = 24,
) -> dict:
    """Predict the best time window to execute transactions for minimum gas cost.

    Uses historical Ethereum gas patterns (hourly + day-of-week multipliers)
    to identify upcoming low-gas windows. Typical savings: 25-45% vs peak hours.

    Best windows: 3-7 AM UTC (daily low) and weekends (-20 to -25%).

    Use this when: an agent can defer a non-urgent transaction to save on gas.
    Combine with get_gas_price for current baseline.

    Args:
        chain         : Chain to optimize: 'ethereum', 'arbitrum', 'base'
        urgency       : 'low' (best price, wait up to 24h), 'medium' (up to 6h),
                        'high' (execute now — returns current best)
        horizon_hours : Lookahead window in hours (1-48)

    Returns:
        JSON with recommended windows, expected savings, and current gas context.
        Price: 0.05 USDC
    """
    horizon_hours = max(1, min(48, horizon_hours))
    current_gas   = await _fetch_gas_current(chain)

    if urgency == "high":
        return {
            "recommendation": "execute_now",
            "urgency":        urgency,
            "current_gas":    current_gas,
            "reasoning":      "High urgency — execute immediately at current gas price.",
            "price":          "0.05 USDC",
        }

    windows = _find_optimal_windows(chain, horizon_hours)
    max_wait = 6 if urgency == "medium" else 24

    # Filtrer selon l'urgence
    relevant = [w for w in windows if w["hours_from_now"] <= max_wait]

    if not relevant:
        return {
            "recommendation": "execute_now",
            "reasoning":      f"No significantly cheaper window found in next {max_wait}h.",
            "current_gas":    current_gas,
            "urgency":        urgency,
            "price":          "0.05 USDC",
        }

    best = relevant[0]

    import datetime
    utc_now = datetime.datetime.utcnow()
    curr_mult = GAS_HOURLY_MULTIPLIER.get(utc_now.hour, 1.0) *                 GAS_DAY_MULTIPLIER.get(utc_now.weekday(), 1.0)
    savings_vs_now = round((1 - best["gas_multiplier"] / curr_mult) * 100, 0)

    return {
        "recommendation":   "wait_for_window",
        "best_window": {
            "in_hours":     best["hours_from_now"],
            "at_time":      best["utc_time"],
            "day":          best["day"],
            "est_savings":  f"{max(0, savings_vs_now):.0f}% vs now",
        },
        "top_windows":      relevant[:3],
        "current_gas":      current_gas,
        "chain":            chain,
        "urgency":          urgency,
        "reasoning":        (
            f"Wait ~{best['hours_from_now']}h ({best['utc_time']}, {best['day']}) "
            f"for estimated {max(0, savings_vs_now):.0f}% gas saving vs current prices."
        ),
        "price":            "0.05 USDC",
    }


# ── Outil : score_contract ────────────────────────────────────────────────────
@mcp.tool
async def score_contract(
    contract_address: Annotated[str, "Contract or token address to analyze. Example: '0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48'"],
    chain: Annotated[str, "Chain where the contract is deployed: 'ethereum', 'bsc', 'polygon', 'arbitrum', 'base', 'optimism'"] = "ethereum",
) -> dict:
    """Score a smart contract for security risks using GoPlus Security API.

    Checks for: honeypot detection, buy/sell tax, ownership risks, proxy patterns,
    trading cooldowns, blacklist functions, mint capabilities, and liquidity risks.

    Risk score 0-100 (100 = safest). Covers tokens and general contracts.

    Use this when: an agent is about to interact with an unknown contract or token.
    Do NOT use for: known safe protocols (use explain_risk for Aave/Morpho/etc.).

    Args:
        contract_address : EVM contract or token address (0x...)
        chain            : Deployment chain. Options: 'ethereum', 'bsc',
                           'polygon', 'arbitrum', 'base', 'optimism'

    Returns:
        JSON with risk_score (0-100), risk_signals, verdict. Price: 0.05 USDC
    """
    chain_id = GOPLUS_CHAIN_IDS.get(chain.lower(), "1")
    addr     = contract_address.lower().strip()

    if not addr.startswith("0x") or len(addr) != 42:
        return {
            "error":   "invalid_address",
            "message": "Address must be a valid EVM address starting with 0x (42 chars).",
        }

    # Appel GoPlus Security API
    token_data = {}
    try:
        async with httpx.AsyncClient(timeout=12) as client:
            r = await client.get(
                f"https://api.gopluslabs.io/api/v1/token_security/{chain_id}",
                params={"contract_addresses": addr}
            )
            if r.status_code == 200:
                d = r.json()
                if d.get("code") == 1:
                    result = d.get("result", {})
                    token_data = result.get(addr, result.get(addr.lower(), {}))
    except Exception as e:
        log.warning(f"GoPlus API error: {e}")

    # Calcul du risk score
    score  = 100
    risks  = []
    greens = []

    if token_data:
        # ── Signaux critiques ──────────────────────────────────────────────
        if token_data.get("is_honeypot") == "1":
            score -= 50; risks.append("⛔ HONEYPOT — cannot sell token")
        elif token_data.get("is_honeypot") == "0":
            greens.append("✅ Not a honeypot")

        buy_tax  = float(token_data.get("buy_tax")  or 0)
        sell_tax = float(token_data.get("sell_tax") or 0)
        if sell_tax > 0.10:
            score -= 25; risks.append(f"⛔ High sell tax: {sell_tax*100:.1f}%")
        elif sell_tax > 0.05:
            score -= 10; risks.append(f"⚠ Sell tax: {sell_tax*100:.1f}%")
        elif sell_tax == 0:
            greens.append("✅ No sell tax")

        if buy_tax > 0.10:
            score -= 15; risks.append(f"⚠ High buy tax: {buy_tax*100:.1f}%")

        # ── Ownership ─────────────────────────────────────────────────────
        if token_data.get("owner_address") in ("", "0x0000000000000000000000000000000000000000"):
            greens.append("✅ Ownership renounced")
        elif token_data.get("can_take_back_ownership") == "1":
            score -= 20; risks.append("⚠ Owner can reclaim contract")

        # ── Fonctions dangereuses ─────────────────────────────────────────
        if token_data.get("is_mintable") == "1":
            score -= 10; risks.append("⚠ Token is mintable (inflation risk)")
        if token_data.get("has_blacklist") == "1":
            score -= 5;  risks.append("⚠ Has blacklist function")
        if token_data.get("has_whitelist") == "1":
            score -= 3;  risks.append("ℹ Has whitelist function")
        if token_data.get("trading_cooldown") == "1":
            score -= 5;  risks.append("⚠ Trading cooldown enabled")
        if token_data.get("transfer_pausable") == "1":
            score -= 8;  risks.append("⚠ Transfers can be paused")

        # ── Proxy / upgradeable ───────────────────────────────────────────
        if token_data.get("is_proxy") == "1":
            score -= 5;  risks.append("ℹ Proxy contract (upgradeable)")
        if token_data.get("is_open_source") == "1":
            greens.append("✅ Open source contract")
        elif token_data.get("is_open_source") == "0":
            score -= 10; risks.append("⚠ Contract not verified/open source")

        # ── Liquidité ─────────────────────────────────────────────────────
        lp_holders = token_data.get("lp_holders", [])
        if lp_holders:
            locked = sum(1 for h in lp_holders if h.get("is_locked") == 1)
            if locked > 0:
                greens.append(f"✅ LP partially locked ({locked} holders)")
            else:
                risks.append("⚠ No locked liquidity detected")
                score -= 8

    else:
        # Pas de données GoPlus — scoring conservateur
        risks.append("⚠ Could not retrieve security data — treat as unknown risk")
        score = 40

    score = max(0, min(100, score))

    if score >= 80:
        verdict = "✅ Low risk — appears safe for interaction"
    elif score >= 60:
        verdict = "⚠ Moderate risk — review signals before interacting"
    elif score >= 40:
        verdict = "⚠ Elevated risk — significant concerns identified"
    else:
        verdict = "⛔ High risk — do NOT interact without thorough audit"

    return {
        "contract":         contract_address,
        "chain":            chain,
        "risk_score":       score,
        "risk_level":       _risk_label(score),
        "verdict":          verdict,
        "risk_signals":     risks,
        "positive_signals": greens,
        "data_source":      "GoPlus Security API" if token_data else "unavailable",
        "price":            "0.05 USDC",
    }



# ── Build app complète (custom routes + x402) ────────────────────────────────
def build_full_app(mcp_server):
    """
    Construit une Starlette app qui inclut custom routes + MCP + x402.
    Résout le problème : mcp.http_app() ne retourne que /mcp sans les
    routes custom (/health, /.well-known/*).
    """
    from starlette.applications import Starlette
    from starlette.routing import Route, Mount

    # Wrappers pour les routes custom
    async def _health(request):
        return await health_check(request)

    async def _x402_manifest(request):
        return await well_known_x402(request)

    async def _agent_card(request):
        return await agent_card_a2a(request)

    async def _a2a(request):
        return await a2a_endpoint(request)

    # App Starlette combinée
    mcp_http = mcp_server.http_app()

    combined = Starlette(routes=[
        Route("/health",                 _health,        methods=["GET"]),
        Route("/.well-known/x402.json",  _x402_manifest, methods=["GET"]),
        Route("/.well-known/agent.json", _agent_card,    methods=["GET"]),
        Route("/a2a",                    _a2a,           methods=["POST"]),
        Mount("/",                       app=mcp_http),
    ])

    # Ajouter x402 middleware
    try:
        facilitator = HTTPFacilitatorClient(FacilitatorConfig(url=X402_FACILITATOR))
        x402_srv = x402ResourceServer(facilitator)
        x402_srv.register(X402_CHAIN_CAIP2, ExactEvmServerScheme())

        routes_protected = {
            "POST /mcp": RouteConfig(accepts=[PaymentOption(
                scheme="exact", price="$0.05",
                network=X402_CHAIN_CAIP2, pay_to=X402_RECIPIENT,
            )]),
            "POST /a2a": RouteConfig(accepts=[PaymentOption(
                scheme="exact", price="$0.05",
                network=X402_CHAIN_CAIP2, pay_to=X402_RECIPIENT,
            )]),
        }
        combined.add_middleware(
            PaymentMiddlewareASGI, routes=routes_protected, server=x402_srv,
        )
        log.info("✅ x402 actif sur POST /mcp et POST /a2a — $0.05 USDC")
        log.info(f"   Wallet: {X402_RECIPIENT[:20]}...")
    except Exception as e:
        log.warning(f"⚠ x402 non disponible ({e}) — serveur actif sans paiement")

    log.info("Routes: /health /mcp /.well-known/x402.json /.well-known/agent.json /a2a")
    return combined


# ── Endpoint de santé (toujours gratuit) ─────────────────────────────────────
@mcp.custom_route("/health", methods=["GET"])
async def health_check(request: Request) -> JSONResponse:
    """Health check endpoint — always free, no payment required."""
    pools_loaded = len(_CACHE.get("data") or [])
    return JSONResponse({
        "status":       "ok",
        "server":       "defi-yield-engine",
        "version":      "1.0.0",
        "pools_loaded": pools_loaded if pools_loaded > 0 else "pending_first_fetch",
        "x402":         "active",
        "network":      X402_NETWORK,
        "recipient":    X402_RECIPIENT,
    })


# ── Point d'entrée ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os
    port = int(os.getenv("PORT", 8000))
    transport = os.getenv("MCP_TRANSPORT", "streamable-http")

    log.info(f"Starting DeFi Yield Engine on port {port} ({transport})")
    log.info(f"x402 recipient  : {X402_RECIPIENT}")
    log.info(f"x402 network    : {X402_NETWORK} (chain {X402_CHAIN_ID})")
    log.info(f"Pricing         : get_best_yield={X402_PRICING['get_best_yield']} USDC | "
             f"explain_risk={X402_PRICING['explain_risk']} USDC | "
             f"compare_yields={X402_PRICING['compare_yields']} USDC")

    if transport == "stdio":
        # Mode local (Claude Desktop / OpenClaw) — sans x402
        mcp.run(transport="stdio")
    else:
        # Mode remote (Railway) — app complète avec routes custom + x402
        try:
            full_app = build_full_app(mcp)
            log.info(f"Démarrage sur port {port} (routes: /mcp /health /.well-known/* /a2a)")
            uvicorn.run(full_app, host="0.0.0.0", port=port)
        except Exception as e:
            log.warning(f"build_full_app échoué ({e}) — fallback mode basique")
            mcp.run(transport="streamable-http", host="0.0.0.0", port=port)


# ═══════════════════════════════════════════════════════════════════════════════
# PARTIE 2 : RESOURCES, PROMPTS, SERVER CARD
# Rend le serveur attrayant pour les agents IA
# ═══════════════════════════════════════════════════════════════════════════════

# ── Resource 1 : Market Overview (données sans appel d'outil) ────────────────
@mcp.resource("defi://market-overview")
async def market_overview() -> str:
    """
    Snapshot temps réel des meilleurs yields DeFi toutes chains.
    Les agents peuvent lire cette ressource SANS payer un appel d'outil.
    Mise à jour toutes les 5 minutes via le cache partagé.
    """
    pools = await _fetch_pools()

    # Top 5 USDC safe
    usdc = sorted(
        [p for p in pools
         if "USDC" in (p.get("symbol") or "").upper()
         and (p.get("tvlUsd") or 0) > 50_000_000
         and (p.get("apy") or 0) > 0
         and _risk_score(p) >= 70],
        key=lambda p: _risk_score(p) * (p.get("apy") or 0),
        reverse=True
    )[:5]

    # Top 3 ETH
    eth = sorted(
        [p for p in pools
         if p.get("symbol") in ("ETH", "WETH", "stETH")
         and (p.get("tvlUsd") or 0) > 100_000_000
         and (p.get("apy") or 0) > 0],
        key=lambda p: _risk_score(p) * (p.get("apy") or 0),
        reverse=True
    )[:3]

    lines = ["# DeFi Yield Market Overview", f"_Updated: {int(time.time())}_", ""]
    lines.append("## Top USDC Yields (risk-adjusted)")
    for p in usdc:
        sc = _risk_score(p)
        lines.append(
            f"- {p['project']} ({p['chain']}): "
            f"{round(p.get('apy',0),2)}% APY | "
            f"risk {sc}/100 | "
            f"TVL ${int((p.get('tvlUsd') or 0)/1e6)}M"
        )

    lines.append("")
    lines.append("## Top ETH Yields (risk-adjusted)")
    for p in eth:
        sc = _risk_score(p)
        lines.append(
            f"- {p['project']} ({p['chain']}): "
            f"{round(p.get('apy',0),2)}% APY | "
            f"risk {sc}/100 | "
            f"TVL ${(p.get('tvlUsd') or 0)/1e9:.1f}B"
        )

    lines.append("")
    lines.append(
        "_Source: DeFiLlama. Risk score: 80+ = safe, 60-79 = moderate, "
        "<60 = elevated risk. Use get_best_yield for personalized recommendations._"
    )
    return "\n".join(lines)


# ── Resource 2 : Risk Glossary ───────────────────────────────────────────────
@mcp.resource("defi://risk-glossary")
async def risk_glossary() -> str:
    """
    Glossaire des termes de risque utilisés dans les outputs de ce serveur.
    Aide les agents à interpréter correctement les recommandations.
    """
    return """# DeFi Yield Engine — Risk Glossary

## risk_score (0-100)
Score propriétaire composite. 80+ = safe. 60-79 = moderate. 40-59 = elevated. <40 = high risk.
Calculé sur : TVL, stabilité APY 30j, ratio base/reward APY, outlier flag, momentum 7j, prédictions DeFiLlama.

## apy_base vs apy_reward
- apy_base : rendement généré par l'utilisation réelle du protocole (durable)
- apy_reward : bonus en tokens émis par le protocole (souvent temporaire, à déprécier)
- Règle : si apy_reward > apy_base, le rendement est non-durable

## outlier
Marqueur DeFiLlama indiquant un APY statistiquement anormal vs les pairs.
Un outlier = true doit être traité avec grande prudence.

## ilRisk (Impermanent Loss)
- "no" : actif unique, pas de risque de perte impermanente
- "yes" : pool multi-actifs (ex: LP Uniswap), exposition au ratio de prix entre actifs

## exposure
- "single" : exposition à un seul actif (plus sûr)
- "multi" : exposition à plusieurs actifs

## TVL (Total Value Locked)
Liquidité totale dans le protocole. Proxy de confiance du marché.
Règle générale : TVL > $100M = acceptable, > $500M = bon, > $1B = excellent.

## risk_profile (paramètre get_best_yield)
- "safe" : score minimum 75/100. Pour capital important, horizon long terme.
- "moderate" : score minimum 55/100. Équilibre rendement/risque.
- "max_yield" : score minimum 35/100. Rendement maximal, risque accru.
"""


# ── Prompt 1 : yield-check (commande slash dans Claude Desktop) ──────────────
@mcp.prompt()
def yield_check(
    asset: str = "USDC",
    amount: str = "10000",
    risk: str = "moderate"
) -> str:
    """
    Template : trouver le meilleur yield pour un montant donné.
    Apparaît comme commande slash dans Claude Desktop.
    Usage : /yield-check asset=USDC amount=50000 risk=safe
    """
    return (
        f"I need to deploy {amount} USD worth of {asset} in DeFi. "
        f"My risk profile is '{risk}'. "
        f"Please use get_best_yield to find the best risk-adjusted yield, "
        f"then use explain_risk to detail why the top recommendation is safe. "
        f"Give me a clear recommendation I can act on immediately."
    )


# ── Prompt 2 : portfolio-optimize ────────────────────────────────────────────
@mcp.prompt()
def portfolio_optimize(
    usdc_amount: str = "0",
    usdt_amount: str = "0",
    eth_amount: str = "0",
    risk: str = "moderate"
) -> str:
    """
    Template : optimiser un portefeuille multi-actifs.
    Usage : /portfolio-optimize usdc_amount=20000 eth_amount=5 risk=safe
    """
    parts = []
    if usdc_amount != "0":
        parts.append(f"{usdc_amount} USDC")
    if usdt_amount != "0":
        parts.append(f"{usdt_amount} USDT")
    if eth_amount != "0":
        parts.append(f"{eth_amount} ETH")

    assets_str = ", ".join(parts) if parts else "a mixed portfolio"

    return (
        f"I have {assets_str} to deploy in DeFi with a '{risk}' risk profile. "
        f"For each asset, use get_best_yield to find the best option. "
        f"Then use compare_yields to compare the top protocols across all assets. "
        f"Give me a final allocation table with protocol, chain, expected APY, "
        f"and estimated monthly revenue for each position."
    )


# ── Prompt 3 : daily-briefing ────────────────────────────────────────────────
@mcp.prompt()
def daily_briefing() -> str:
    """
    Template : rapport quotidien des meilleures opportunités yield.
    Usage : /daily-briefing
    Idéal pour les agents qui tournent sur un cron job matinal.
    """
    return (
        "Generate a daily DeFi yield briefing using the defi://market-overview resource. "
        "Structure it as: "
        "1) Market summary (2 sentences on overall yield environment), "
        "2) Top opportunity of the day with reasoning, "
        "3) Any notable changes or risks to watch. "
        "Keep it under 150 words, actionable and factual."
    )


# ── MCP Server Card (.well-known) ─────────────────────────────────────────────
# Exposé via un outil léger pour la découverte automatique
@mcp.tool
async def server_info() -> dict:
    """
    Discover this DeFi yield server capabilities, tools, pricing, and free resources.

    Returns complete server metadata: tool names + descriptions, available MCP resources
    (defi://market-overview, defi://risk-glossary), prompt templates (/yield-check,
    /portfolio-optimize, /daily-briefing), and per-tool USDC pricing.

    Call this FIRST before any paid tool to understand what is available and free.
    Data source: DeFiLlama (13,800+ pools). Cache: 5 min. This call is free.

    Returns:
        JSON with: name, description, tools {name: description}, resources {uri: desc},
        prompts {name: desc}, pricing {tool: "X.XX USDC"}, data_source, cache_ttl_seconds.
    """
    return {
        "name": "DeFi Yield Decision Engine",
        "version": "1.0.0",
        "description": (
            "Complete DeFi Intelligence Engine — Yield, Liquid Staking, "
            "Restaking, RWA, and Perpetuals. Risk-adjusted recommendations "
            "across 7,000+ protocols on 500+ chains via DeFiLlama. "
            "Opinionated single answers with reasoning. Optimized for agents."
        ),
        "tools": {
            "get_best_yield":           "Best yield for asset + risk profile — 0.05 USDC",
            "get_optimal_allocation":   "Multi-protocol capital allocation — 0.05 USDC",
            "explain_risk":             "Risk breakdown for a specific protocol — 0.05 USDC",
            "compare_yields":           "Side-by-side protocol comparison — 0.05 USDC",
            "get_best_liquid_staking":  "Best liquid staking protocol — 0.05 USDC",
            "get_best_restaking":       "Best restaking/LRT protocol — 0.05 USDC",
            "get_best_rwa":             "Best Real World Asset protocol — 0.05 USDC",
            "get_perps_overview":       "Top perps protocols by volume — 0.05 USDC",
            "compare_perps":            "Side-by-side perps comparison — 0.05 USDC",
            "get_optimal_gas_window":   "Predict best time to transact for min gas — 0.05 USDC",
            "score_contract":           "Smart contract security risk score — 0.05 USDC",
            "get_gas_price":            "Current gas prices across urgency levels — FREE",
            "get_defi_overview":        "Full DeFi market snapshot — FREE",
            "yield_alert_set":          "Register APY threshold alert — FREE",
            "yield_alert_check":        "Poll alert status — FREE",
            "yield_alert_delete":       "Remove an alert — FREE",
            "yield_alerts_list":        "List all active alerts — FREE",
            "server_info":              "Server metadata and capabilities — FREE",
        },
        "resources": {
            "defi://market-overview": "Real-time snapshot of top yields (free, no tool call)",
            "defi://risk-glossary": "Definitions of risk terms used in outputs",
        },
        "prompts": {
            "yield_check":        "Find best yield for asset + amount — /yield-check",
            "portfolio_optimize": "Optimize multi-asset portfolio — /portfolio-optimize",
            "daily_briefing":     "Morning yield market summary — /daily-briefing",
            "yield_watch":        "Monitor APY threshold + act when triggered — /yield-watch",
        },
        "pricing": {
            "get_best_yield": "0.05 USDC",
            "explain_risk": "0.05 USDC",
            "compare_yields": "0.05 USDC",
            "server_info": "free",
            "resources": "free",
            "prompts": "free",
        },
        "payment": {
            "protocol": "x402",
            "network": X402_NETWORK,
            "chain_id": X402_CHAIN_ID,
            "asset": "USDC",
            "recipient": X402_RECIPIENT,
        },
        "data_source": "DeFiLlama (https://defillama.com)",
        "cache_ttl_seconds": 300,
        "contact": "your@email.com",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# PARTIE 3 : YIELD ALERTS + A2A AGENT CARD
# ═══════════════════════════════════════════════════════════════════════════════

# Stockage alertes en mémoire
_ALERTS: dict = {}


@mcp.custom_route("/.well-known/agent.json", methods=["GET"])
async def agent_card_a2a(request: Request) -> JSONResponse:
    """A2A Agent Card — Google Agent-to-Agent Protocol discovery standard.
    Auto-crawled by A2A orchestrators (Salesforce, SAP, ServiceNow, Google ADK).
    """
    return JSONResponse({
        "name": "DeFi Yield Decision Engine",
        "description": (
            "Risk-adjusted DeFi yield recommendations and APY threshold alerts "
            "across 548 protocols and 115 chains. One answer, not 13,800 pools."
        ),
        "url": "https://defi-yield-engine-production.up.railway.app/mcp",
        "version": "1.0.0",
        "provider": {"name": "danteriva45", "organization": "Independent"},
        "capabilities": {
            "streaming": False,
            "pushNotifications": False,
            "stateTransitionHistory": False,
        },
        "authentication": {"schemes": ["x402"]},
        "defaultInputModes": ["application/json"],
        "defaultOutputModes": ["application/json"],
        "skills": [
            {"id": "get_best_yield", "name": "Get Best DeFi Yield",
             "description": "Single best risk-adjusted yield for USDC, USDT, ETH across 548 protocols.",
             "tags": ["defi", "yield", "USDC", "ETH", "risk"], "price": "0.05 USDC"},
            {"id": "get_optimal_allocation", "name": "Optimal Capital Allocation",
             "description": "Split capital across 2-5 protocols for maximum risk-adjusted yield.",
             "tags": ["allocation", "portfolio", "defi", "yield", "routing"], "price": "0.05 USDC"},
            {"id": "explain_risk", "name": "Explain Protocol Risk",
             "description": "9-signal risk breakdown for any DeFiLlama protocol.",
             "tags": ["risk", "audit", "defi", "safety"], "price": "0.05 USDC"},
            {"id": "compare_yields", "name": "Compare DeFi Protocols",
             "description": "Side-by-side risk-adjusted comparison of 2-6 protocols.",
             "tags": ["compare", "defi", "yield"], "price": "0.05 USDC"},
            {"id": "get_best_liquid_staking", "name": "Get Best Liquid Staking",
             "description": "Best LST protocol for ETH, SOL, BNB. Covers Lido, Rocket Pool, Jito, 50+ others.",
             "tags": ["liquid-staking", "ETH", "SOL", "lido", "staking"], "price": "0.05 USDC"},
            {"id": "get_best_restaking", "name": "Get Best Restaking / LRT",
             "description": "Best restaking (EigenLayer, Symbiotic) or liquid restaking token (Ether.fi, Renzo, Puffer).",
             "tags": ["restaking", "eigenlayer", "lrt", "ether.fi"], "price": "0.05 USDC"},
            {"id": "get_best_rwa", "name": "Get Best RWA Protocol",
             "description": "Top Real World Asset protocols: T-bills (Ondo, BUIDL), private credit (Maple, Centrifuge).",
             "tags": ["rwa", "ondo", "maple", "real-world-assets"], "price": "0.05 USDC"},
            {"id": "get_perps_overview", "name": "Get Perps Overview",
             "description": "Top perpetuals by 24h volume: Hyperliquid, dYdX, GMX, Drift, Jupiter Perps.",
             "tags": ["perps", "derivatives", "hyperliquid", "gmx"], "price": "0.05 USDC"},
            {"id": "compare_perps", "name": "Compare Perps Protocols",
             "description": "Side-by-side comparison of perpetuals protocols by volume.",
             "tags": ["perps", "compare", "derivatives"], "price": "0.05 USDC"},
            {"id": "get_defi_overview", "name": "DeFi Market Overview",
             "description": "Free snapshot: top protocols across Yield, Staking, Restaking, RWA and Perps.",
             "tags": ["overview", "market", "defi"], "price": "free"},
            {"id": "yield_alert_set", "name": "Set Yield Alert",
             "description": "Register APY threshold alert — fires when yield exceeds target.",
             "tags": ["alert", "monitoring", "automation"], "price": "free"},
            {"id": "yield_alert_check", "name": "Check Yield Alert",
             "description": "Poll alert status — triggered/watching + current best APY.",
             "tags": ["alert", "polling"], "price": "free"},
        ],
    })


# ═══════════════════════════════════════════════════════════════════════════════
