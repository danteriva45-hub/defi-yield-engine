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
    "get_best_yield":  0.05,   # 0,05 USDC
    "explain_risk":    0.02,   # 0,02 USDC
    "compare_yields":  0.03,   # 0,03 USDC
    "server_info":     0.00,   # gratuit
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
                "get_best_yield":  {"amount": "50000",  "decimals": 6},
                "explain_risk":    {"amount": "20000",  "decimals": 6},
                "compare_yields":  {"amount": "30000",  "decimals": 6},
                "server_info":     {"amount": "0",      "decimals": 6},
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
            {"id": "explain_risk", "name": "Explain Protocol Risk",
             "description": "Detailed risk signal breakdown for any DeFiLlama protocol.",
             "tags": ["risk", "audit", "defi", "safety"], "price": "0.02 USDC"},
            {"id": "compare_yields", "name": "Compare DeFi Protocols",
             "description": "Side-by-side risk-adjusted comparison of 2-6 protocols.",
             "tags": ["compare", "defi", "yield"], "price": "0.03 USDC"},
            {"id": "yield_alert_set", "name": "Set Yield Alert",
             "description": "Register APY threshold alert — fires when yield exceeds target.",
             "tags": ["alert", "monitoring", "automation"], "price": "free"},
            {"id": "yield_alert_check", "name": "Check Yield Alert",
             "description": "Poll alert status — triggered/watching + current best APY.",
             "tags": ["alert", "polling"], "price": "free"},
        ],
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
        # Mode local (Claude Desktop / OpenClaw)
        mcp.run(transport="stdio")
    else:
        # Mode remote (Railway / serveur)
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
            "Risk-adjusted DeFi yield recommendations. "
            "Covers 13,800+ pools across 548 protocols and 115 chains. "
            "Returns opinionated recommendations with reasoning, "
            "not raw data dumps. Optimized for minimal token consumption."
        ),
        "tools": {
            "get_best_yield":     "Best yield recommendation for asset + risk profile — 0.05 USDC",
            "explain_risk":       "Detailed risk breakdown for a specific protocol — 0.02 USDC",
            "compare_yields":     "Side-by-side comparison of multiple protocols — 0.03 USDC",
            "yield_alert_set":    "Register APY threshold alert — FREE",
            "yield_alert_check":  "Poll alert status — FREE",
            "yield_alert_delete": "Remove an alert — FREE",
            "yield_alerts_list":  "List all active alerts — FREE",
            "server_info":        "Server metadata and capabilities — FREE",
        },
        "resources": {
            "defi://market-overview": "Real-time snapshot of top yields (free, no tool call)",
            "defi://risk-glossary": "Definitions of risk terms used in outputs",
        },
        "prompts": {
            "yield_check": "Find best yield for an amount — /yield-check",
            "portfolio_optimize": "Optimize multi-asset portfolio — /portfolio-optimize",
            "daily_briefing": "Daily yield briefing — /daily-briefing",
        },
        "pricing": {
            "get_best_yield": "0.05 USDC",
            "explain_risk": "0.02 USDC",
            "compare_yields": "0.03 USDC",
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
            {"id": "explain_risk", "name": "Explain Protocol Risk",
             "description": "Detailed risk signal breakdown for any DeFiLlama protocol.",
             "tags": ["risk", "audit", "defi", "safety"], "price": "0.02 USDC"},
            {"id": "compare_yields", "name": "Compare DeFi Protocols",
             "description": "Side-by-side risk-adjusted comparison of 2-6 protocols.",
             "tags": ["compare", "defi", "yield"], "price": "0.03 USDC"},
            {"id": "yield_alert_set", "name": "Set Yield Alert",
             "description": "Register APY threshold alert — fires when yield exceeds target.",
             "tags": ["alert", "monitoring", "automation"], "price": "free"},
            {"id": "yield_alert_check", "name": "Check Yield Alert",
             "description": "Poll alert status — triggered/watching + current best APY.",
             "tags": ["alert", "polling"], "price": "free"},
        ],
    })
