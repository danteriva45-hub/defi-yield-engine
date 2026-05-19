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
from typing import Optional
import httpx
from fastmcp import FastMCP

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
    asset: str,
    amount_usd: float,
    risk_profile: str = "moderate",
    chain: str = "all",
) -> dict:
    """
    Recommandation de yield risk-ajustée pour un asset et un profil donné.

    Args:
        asset        : Symbole de l'actif. Exemples : "USDC", "USDT", "ETH", "DAI"
        amount_usd   : Montant à déployer en USD. Utilisé pour filtrer les pools
                       avec TVL suffisant (min TVL = 10× le montant).
        risk_profile : "safe" (score ≥75), "moderate" (≥55), "max_yield" (≥35)
        chain        : "all" ou nom de chain — "Ethereum", "Arbitrum", "Base",
                       "Polygon", "Optimism", "Avalanche", "BNB Chain", "Solana"

    Returns:
        Objet JSON compact avec recommandation principale + 2 alternatives.
        Format optimisé pour agents IA (< 200 tokens).
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
    asset: str,
    protocols: list[str],
    chain: str = "all",
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


# ── Point d'entrée ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os
    port = int(os.getenv("PORT", 8000))
    transport = os.getenv("MCP_TRANSPORT", "streamable-http")

    log.info(f"Starting DeFi Yield Engine on port {port} ({transport})")

    if transport == "stdio":
        # Mode local (Claude Desktop / OpenClaw)
        mcp.run(transport="stdio")
    else:
        # Mode remote (Railway / serveur)
        mcp.run(transport="streamable-http", host="0.0.0.0", port=port)
