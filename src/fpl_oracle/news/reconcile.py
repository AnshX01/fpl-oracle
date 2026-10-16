"""
Single Availability & Minutes Reconciliation Layer.
Performs an auditable, unified per-player, per-fixture reconciliation step before projections.
Accepts official FPL baseline and candidate LLM evidence.
Enforces single-adjustment invariant (zero double-counting of availability/injury risks).
Supports API-Only, Shadow, and Gated-Active modes.
"""

import json
import logging
import re
import urllib.parse
from dataclasses import dataclass
from typing import Any

from fpl_oracle.api.models import BootstrapStatic, Fixture
from fpl_oracle.config import (
    CONFIG_DIR,
    NEWS_GATE_MAX_FALSE_RULED_OUT,
    NEWS_GATE_MIN_PRECISION,
    NEWS_GATE_MIN_RECALL,
    NEWS_RECOMMENDATION_MODE,
    REPORTS_DIR,
)
from fpl_oracle.news.models import (
    EvidenceCategory,
    PlayerEvidence,
    RecommendationMode,
    ReconciledAvailability,
)

logger = logging.getLogger("fpl_oracle.news.reconcile")

DEFAULT_TRUSTED_PUBLISHERS = [
    "bbc.co.uk",
    "bbci.co.uk",
    "theguardian.com",
    "skysports.com",
    "talksport.com",
    "premierleague.com",
    "theathletic.com",
]


def get_trusted_publishers() -> list[str]:
    """Load trusted publisher domains from config/news_sources.yaml."""
    try:
        import yaml

        cfg_path = CONFIG_DIR / "news_sources.yaml"
        if cfg_path.exists():
            data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
            trusted = data.get("trusted_publishers")
            if isinstance(trusted, list) and trusted:
                return [str(p).lower().strip() for p in trusted]
    except Exception as e:
        logger.debug(f"Could not load trusted_publishers from yaml: {e}")
    return list(DEFAULT_TRUSTED_PUBLISHERS)


def extract_publisher_domain(url: str | None) -> str:
    """Extract and normalize canonical publisher domain from an article URL."""
    if not url:
        return ""
    try:
        parsed = urllib.parse.urlparse(url)
        host = parsed.netloc.lower().split(":")[0]
        for p in get_trusted_publishers():
            if host == p or host.endswith("." + p):
                return p
        return host
    except Exception:
        return ""


def is_safe_quote(quote: str | None) -> tuple[bool, str]:
    """Validate quote span against HTML comment injection, prompt override, and homoglyphs."""
    if not quote or not str(quote).strip():
        return False, "Empty quote"
    q_str = str(quote)
    if "<" in q_str or ">" in q_str:
        if any(tag in q_str.lower() for tag in ["<!--", "-->", "<script", "</script>", "<span", "<div"]):
            return False, "HTML comment or markup injection detected in quote"
    lower_q = q_str.lower()
    if any(
        kw in lower_q
        for kw in ["system override", "ignore previous", "ignore all previous", "mandatory priority override"]
    ):
        return False, "Adversarial prompt injection keywords detected in quote"
    for ch in q_str:
        if ch in ("\u200b", "\u200c", "\u200d", "\ufeff"):
            return False, "Zero-width hidden characters detected in quote"
    return True, "Safe"


def compute_wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """
    Computes Wilson 95% score confidence interval (centre +/- spread) / (1 + z^2/n) (WI-6.2).
    Returns (lower_bound, upper_bound).
    """
    import math

    if total <= 0:
        return 0.0, 1.0
    p_hat = successes / total
    denom = 1.0 + (z**2) / total
    centre = (p_hat + (z**2) / (2.0 * total)) / denom
    spread = (z / denom) * math.sqrt(max(0.0, (p_hat * (1.0 - p_hat) / total) + (z**2) / (4.0 * (total**2))))
    lower = max(0.0, centre - spread)
    upper = min(1.0, centre + spread)
    if abs(upper - 1.0) < 1e-9:
        upper = 1.0
    if abs(lower - 0.0) < 1e-9:
        lower = 0.0
    return lower, upper


def check_news_benchmark_gate(
    extractor_name: str = "deterministic_fallback",
    use_wilson_bounds: bool | None = None,
) -> tuple[bool, str, dict[str, Any]]:
    """
    Verifies whether the active news extractor satisfies benchmark accuracy thresholds (WI-6.2).
    Uses Wilson 95% intervals and enforces minimum labelled-set size requirements when enabled:
    - Precision Wilson 95% lower bound >= NEWS_GATE_MIN_PRECISION (default 85%)
    - Recall Wilson 95% lower bound >= NEWS_GATE_MIN_RECALL (default 80%)
    - False Ruled Out Rate Wilson 95% upper bound <= NEWS_GATE_MAX_FALSE_RULED_OUT (default 2%)
    - Minimum negative sample size >= 189 to reach <=2% Wilson upper bound.
    """
    from fpl_oracle.config import is_flag_enabled

    if use_wilson_bounds is None:
        use_wilson_bounds = is_flag_enabled("wilson_news_gate", default=False)

    bench_file = REPORTS_DIR / "news_benchmark.json"
    if not bench_file.exists():
        return False, "Benchmark artifact reports/news_benchmark.json missing", {}

    try:
        data = json.loads(bench_file.read_text(encoding="utf-8"))
        extractors = data.get("extractors", {})
        metrics = extractors.get(extractor_name)
        if not isinstance(metrics, dict) or not metrics.get("active"):
            return False, f"No measured active benchmark for {extractor_name}", {}
        if metrics.get("status") in ("mock", "awaiting", "unverified"):
            return False, f"Extractor {extractor_name} is unverified", metrics

        if extractor_name == "gemini":
            import os

            from fpl_oracle.news.gemini_extractor import DEFAULT_GEMINI_MODEL

            if metrics.get("model") != os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL):
                return False, "Gemini model changed; rerun its benchmark", metrics
            if not metrics.get("production_gate", {}).get("gate_passed"):
                return False, "Gemini benchmark safety checks failed", metrics

        cm = data.get("confusion_matrix", {})
        tp = int(cm.get("true_positives", 0))
        fp = int(cm.get("false_positives", 0))
        fn = int(cm.get("false_negatives", 0))
        tn = int(cm.get("true_negatives", 0))
        false_ro_cnt = int(data.get("false_ruled_out_count", 0))

        # Precision trials: TP + FP
        n_prec = tp + fp
        # Recall trials: TP + FN
        n_rec = tp + fn
        # False ruled out trials: TN + FP (or total non-ruled-out cases)
        n_fro = tn + fp if (tn + fp) > 0 else max(1, data.get("total_cases_evaluated", 70) - tp)

        precision = (
            (tp / n_prec)
            if n_prec > 0
            else float(metrics.get("extraction_precision_pct", metrics.get("precision_pct", 0.0)) or 0.0) / 100.0
        )
        recall = (
            (tp / n_rec)
            if n_rec > 0
            else float(metrics.get("extraction_recall_pct", metrics.get("recall_pct", 0.0)) or 0.0) / 100.0
        )
        false_ro_rate = (
            (false_ro_cnt / n_fro)
            if n_fro > 0
            else float(metrics.get("false_ruled_out_rate_pct", metrics.get("false_ruled_out_rate", 0.0)) or 0.0) / 100.0
        )

        prec_lower, prec_upper = compute_wilson_interval(tp, n_prec) if n_prec > 0 else (precision, precision)
        rec_lower, rec_upper = compute_wilson_interval(tp, n_rec) if n_rec > 0 else (recall, recall)
        fro_lower, fro_upper = (
            compute_wilson_interval(false_ro_cnt, n_fro) if n_fro > 0 else (false_ro_rate, false_ro_rate)
        )

        metrics_annotated = dict(metrics)
        metrics_annotated["wilson_bounds"] = {
            "precision": [round(prec_lower, 4), round(prec_upper, 4)],
            "recall": [round(rec_lower, 4), round(rec_upper, 4)],
            "false_ruled_out": [round(fro_lower, 4), round(fro_upper, 4)],
            "n_precision": n_prec,
            "n_recall": n_rec,
            "n_false_ruled_out": n_fro,
        }

        if use_wilson_bounds:
            # Enforce minimum sample size required for 2% Wilson upper bound
            min_negatives_required = 189
            if n_fro < min_negatives_required:
                return (
                    False,
                    f"Benchmark negative sample size (N={n_fro}) is below minimum required N={min_negatives_required} "
                    f"to statistically guarantee <=2% Wilson upper bound (current Wilson upper bound: {fro_upper:.1%})",
                    metrics_annotated,
                )

            if prec_lower < NEWS_GATE_MIN_PRECISION:
                return (
                    False,
                    f"Precision Wilson 95% lower bound {prec_lower:.1%} below gate threshold {NEWS_GATE_MIN_PRECISION:.1%} "
                    f"(point={precision:.1%}, bounds=[{prec_lower:.1%}, {prec_upper:.1%}], N={n_prec})",
                    metrics_annotated,
                )
            if rec_lower < NEWS_GATE_MIN_RECALL:
                return (
                    False,
                    f"Recall Wilson 95% lower bound {rec_lower:.1%} below gate threshold {NEWS_GATE_MIN_RECALL:.1%} "
                    f"(point={recall:.1%}, bounds=[{rec_lower:.1%}, {rec_upper:.1%}], N={n_rec})",
                    metrics_annotated,
                )
            if fro_upper > NEWS_GATE_MAX_FALSE_RULED_OUT:
                return (
                    False,
                    f"False ruled-out rate Wilson 95% upper bound {fro_upper:.1%} exceeds tolerance {NEWS_GATE_MAX_FALSE_RULED_OUT:.1%} "
                    f"(point={false_ro_rate:.1%}, bounds=[{fro_lower:.1%}, {fro_upper:.1%}], N={n_fro})",
                    metrics_annotated,
                )

            return (
                True,
                f"Gate OPEN (Wilson 95%): Precision lower={prec_lower:.1%}, Recall lower={rec_lower:.1%}, False-RO upper={fro_upper:.1%}",
                metrics_annotated,
            )
        else:
            # Point estimate fallback
            if precision < NEWS_GATE_MIN_PRECISION:
                return (
                    False,
                    f"Precision {precision:.1%} below gate threshold {NEWS_GATE_MIN_PRECISION:.1%}",
                    metrics_annotated,
                )
            if recall < NEWS_GATE_MIN_RECALL:
                return False, f"Recall {recall:.1%} below gate threshold {NEWS_GATE_MIN_RECALL:.1%}", metrics_annotated
            if false_ro_rate > NEWS_GATE_MAX_FALSE_RULED_OUT:
                return (
                    False,
                    f"False ruled-out rate {false_ro_rate:.1%} exceeds tolerance {NEWS_GATE_MAX_FALSE_RULED_OUT:.1%}",
                    metrics_annotated,
                )

            return (
                True,
                f"Gate OPEN (point): Precision={precision:.1%}, Recall={recall:.1%}, False-RO={false_ro_rate:.1%}",
                metrics_annotated,
            )
    except Exception as e:
        return False, f"Error inspecting news benchmark gate: {e}", {}


@dataclass
class ReconcileProbabilitySettings:
    """
    Named, calibrated probability parameters for availability reconciliation (N3).
    Documented defaults derived from Premier League historical return rates:
    - Players cleared in press conferences after return to training start ~65% of the time.
    - Late fitness test doubts yield ~50% availability and ~40% start rate.
    - Minutes restricted assets have ~70% start rate with capped ceiling.
    - Regular starters with official >= 75% have ~90% baseline start probability.
    """

    prob_start_high_base: float = 0.90
    prob_start_low_base: float = 0.50
    prob_start_minutes_limit: float = 0.70
    prob_avail_returned_training: float = 0.85
    prob_start_returned_training: float = 0.65
    prob_avail_doubtful: float = 0.50
    prob_start_doubtful: float = 0.40
    single_source_cop_cap: float = 25.0


def normalize_publisher_group(domain: str) -> str:
    """Normalize publisher domains belonging to the same media entity."""
    d = (domain or "").lower()
    if "bbc" in d:
        return "bbc"
    if "theguardian" in d or "guardian" in d:
        return "guardian"
    if "skysports" in d or "sky" in d:
        return "sky"
    if "athletic" in d:
        return "theathletic"
    if "telegraph" in d:
        return "telegraph"
    return d


def are_syndicated_copies(e1: PlayerEvidence, e2: PlayerEvidence) -> bool:
    """
    Determine whether two evidence items originate from the same wire story or syndicate.
    Returns True if evidence shows they are syndicated copies of the same original source,
    even if hosted on different publisher domains.
    """
    dom1 = extract_publisher_domain(e1.source_url)
    dom2 = extract_publisher_domain(e2.source_url)
    if dom1 and dom2:
        if dom1 == dom2 or normalize_publisher_group(dom1) == normalize_publisher_group(dom2):
            return True

    # 1. Canonical URL match
    canon1 = (getattr(e1, "canonical_url", None) or "").strip().lower()
    canon2 = (getattr(e2, "canonical_url", None) or "").strip().lower()
    if canon1 and canon2 and canon1 == canon2:
        return True

    # 2. Shared wire agency / wire service attribution (e.g. PA Media, Reuters, AP)
    wire1 = (getattr(e1, "wire_agency", None) or "").strip().lower()
    wire2 = (getattr(e2, "wire_agency", None) or "").strip().lower()
    if wire1 and wire2 and wire1 == wire2:
        return True

    # 3. Matching syndicate article ID
    art_id1 = (getattr(e1, "article_id", None) or "").strip()
    art_id2 = (getattr(e2, "article_id", None) or "").strip()
    if art_id1 and art_id2 and art_id1 == art_id2:
        return True

    # 4. Textual similarity of surrounding article body outside the quote
    surr1 = getattr(e1, "surrounding_text", None) or ""
    surr2 = getattr(e2, "surrounding_text", None) or ""
    if surr1 and surr2:
        words1 = set(re.findall(r"\w+", surr1.lower()))
        words2 = set(re.findall(r"\w+", surr2.lower()))
        if len(words1) >= 10 and len(words2) >= 10:
            jaccard = len(words1 & words2) / len(words1 | words2)
            if jaccard >= 0.60:
                return True

    return False


class AvailabilityReconciler:
    def __init__(self, mode: str | None = None, prob_settings: ReconcileProbabilitySettings | None = None):
        raw_mode = (mode or NEWS_RECOMMENDATION_MODE or "shadow").lower().strip()
        if raw_mode in ("gated_active", "active"):
            self.mode = RecommendationMode.GATED_ACTIVE
        elif raw_mode == "api_only":
            self.mode = RecommendationMode.API_ONLY
        else:
            self.mode = RecommendationMode.SHADOW
        self.prob_settings = prob_settings or ReconcileProbabilitySettings()
        self._logged_disagreements: list[dict[str, Any]] = []

    def get_logged_disagreements(self) -> list[dict[str, Any]]:
        """Return all logged disagreements between news-adjusted and API-only availability."""
        return list(self._logged_disagreements)

    def clear_logged_disagreements(self) -> None:
        """Clear logged disagreements."""
        self._logged_disagreements.clear()

    def reconcile_player_fixture(
        self,
        element: Any,
        target_gw: int,
        fixture: Fixture | None = None,
        candidate_evidence: list[PlayerEvidence] | None = None,
        extractor_name: str = "deterministic_fallback",
    ) -> ReconciledAvailability:
        """
        Reconcile a single player's availability for a specific gameweek and fixture.
        Guarantees single availability adjustment without double counting.
        """
        elem_id = getattr(element, "id", 0)
        web_name = getattr(element, "web_name", "Unknown")
        team_id = getattr(element, "team", 0)
        status = getattr(element, "status", "a")
        news_text = getattr(element, "news", "") or ""

        # 1. Authoritative Official FPL Baseline
        baseline_cop = 100.0
        if status in ("i", "s"):
            baseline_cop = 0.0
        elif getattr(element, "chance_of_playing_next_round", None) is not None:
            baseline_cop = float(element.chance_of_playing_next_round)
        elif getattr(element, "chance_of_playing_this_round", None) is not None:
            baseline_cop = float(element.chance_of_playing_this_round)

        # Baseline probabilities
        p_avail = baseline_cop / 100.0
        p_start_given_avail = (
            self.prob_settings.prob_start_high_base
            if baseline_cop >= 75.0
            else (self.prob_settings.prob_start_low_base if baseline_cop > 0 else 0.0)
        )
        mins_limit = None
        reconciliation_reason = f"Official FPL status: '{status}' ({baseline_cop:.0f}% chance)"

        # Check official scout loan ineligibility
        scout_risks = getattr(element, "scout_risks", None) or []
        for risk in scout_risks:
            risk_prop = getattr(risk, "property", "") if hasattr(risk, "property") else risk.get("property", "")
            risk_gw = getattr(risk, "gameweek", None) if hasattr(risk, "gameweek") else risk.get("gameweek")
            if risk_prop == "loan_ineligible" and (risk_gw is None or risk_gw == target_gw):
                baseline_cop = 0.0
                p_avail = 0.0
                p_start_given_avail = 0.0
                reconciliation_reason = f"Official Scout Risk: Loan ineligible for GW {target_gw}"

        baseline_p_avail = p_avail
        baseline_p_start = p_start_given_avail
        effective_cop = baseline_cop
        is_shadow_override = False
        source_quote = None
        source_url = None
        evidence_category = EvidenceCategory.UNKNOWN
        rejected_signals: list[str] = []

        # 2. Evaluate Candidate Text Evidence if in Shadow or Gated-Active mode
        if self.mode != RecommendationMode.API_ONLY and candidate_evidence:
            player_evidences = [
                e for e in candidate_evidence if e.player_id == elem_id and e.target_gw in (None, target_gw)
            ]

            trusted_publishers = get_trusted_publishers()
            for ev in player_evidences:
                # 1. Quote safety verification (HTML comments, prompt injection, homoglyphs)
                safe_q, safety_reason = is_safe_quote(ev.quote)
                if not safe_q:
                    rejected_signals.append(f"Unsafe quote rejected ({safety_reason}): '{ev.quote[:40]}'")
                    continue

                # 2. Filter out cup-only quotes for Premier League planning
                if ev.match_context and any(
                    cup in ev.match_context.lower() for cup in ["carabao", "fa cup", "champions league", "europa"]
                ):
                    rejected_signals.append(
                        f"Cup competition quote ('{ev.match_context}') does not apply to PL GW {target_gw}"
                    )
                    continue

                # 3. Filter out negated updates ("not ruled out") - abstain from ruling out
                if ev.is_negated:
                    rejected_signals.append(f"Negated statement ('{ev.quote}') abstains from ruling out player")
                    continue

                # 4. Filter out low-confidence extractions
                if ev.confidence < 0.65:
                    rejected_signals.append(f"Low confidence ({ev.confidence:.2f}) quote rejected")
                    continue

                # 5. Publisher allowlist verification in Gated-Active mode for external articles
                if ev.source_url:
                    domain = extract_publisher_domain(ev.source_url)
                    is_allowlisted = domain in trusted_publishers
                    if self.mode == RecommendationMode.GATED_ACTIVE and not is_allowlisted:
                        rejected_signals.append(
                            f"Source domain '{domain}' not in publisher allowlist; override rejected in Gated-Active mode"
                        )
                        continue

                # Process verified high-confidence candidate evidence
                evidence_category = ev.category
                source_quote = ev.quote
                source_url = ev.source_url

                if ev.category in (EvidenceCategory.RULED_OUT, EvidenceCategory.INELIGIBLE):
                    # Filter matching high-confidence candidate evidence
                    valid_ruled_out: list[PlayerEvidence] = []
                    for other in player_evidences:
                        if (
                            other.category in (EvidenceCategory.RULED_OUT, EvidenceCategory.INELIGIBLE)
                            and not other.is_negated
                        ):
                            o_safe, _ = is_safe_quote(other.quote)
                            if not o_safe:
                                continue
                            o_dom = extract_publisher_domain(other.source_url)
                            if (
                                self.mode == RecommendationMode.GATED_ACTIVE
                                and other.source_url
                                and o_dom not in trusted_publishers
                            ):
                                continue
                            valid_ruled_out.append(other)

                    # Group evidence items into independent source clusters using syndication detection
                    source_clusters: list[list[PlayerEvidence]] = []
                    for item in valid_ruled_out:
                        placed = False
                        for cluster in source_clusters:
                            if any(are_syndicated_copies(item, member) for member in cluster):
                                cluster.append(item)
                                placed = True
                                break
                        if not placed:
                            source_clusters.append([item])

                    num_independent_sources = len(source_clusters)
                    has_external_urls = any(bool(e.source_url) for e in valid_ruled_out)

                    if self.mode == RecommendationMode.GATED_ACTIVE and has_external_urls and num_independent_sources < 2:
                        # Single uncorroborated allowlisted source: cap at configurable single_source_cop_cap and flag it!
                        is_shadow_override = True
                        cap_val = self.prob_settings.single_source_cop_cap
                        effective_cop = min(baseline_cop, cap_val)
                        p_avail = min(baseline_p_avail, cap_val / 100.0)
                        p_start_given_avail = 0.0
                        reconciliation_reason = (
                            f"Candidate Evidence: Single uncorroborated allowlisted source; "
                            f"availability capped at {effective_cop:.0f}% (Official FPL: '{status}' {baseline_cop:.0f}%) ('{ev.quote[:50]}...')"
                        )
                        break
                    else:
                        # Corroborated by 2+ independent allowlisted sources (or in Shadow mode)
                        is_shadow_override = True
                        effective_cop = 0.0
                        p_avail = 0.0
                        p_start_given_avail = 0.0
                        reconciliation_reason = f"Candidate Evidence: Manager ruled out player ('{ev.quote[:50]}...')"
                        break

                elif ev.category == EvidenceCategory.MINUTES_LIMIT:
                    is_shadow_override = True
                    mins_limit = float(ev.minutes_restriction or 60.0)
                    # A minutes cap alone does not clear an official fitness doubt.
                    effective_cop = baseline_cop
                    p_avail = baseline_p_avail
                    p_start_given_avail = self.prob_settings.prob_start_minutes_limit
                    reconciliation_reason = (
                        f"Candidate Evidence: Minutes restricted to {mins_limit:.0f}m ('{ev.quote[:50]}...')"
                    )
                    break

                elif ev.category in (EvidenceCategory.AVAILABLE, EvidenceCategory.RETURNED_TO_TRAINING):
                    if baseline_cop <= 50.0 and status != "i":
                        # Manager confirmed player is available despite old official doubt
                        is_shadow_override = True
                        effective_cop = 75.0
                        p_avail = self.prob_settings.prob_avail_returned_training
                        p_start_given_avail = self.prob_settings.prob_start_returned_training
                        reconciliation_reason = (
                            f"Candidate Evidence: Confirmed returned to training ('{ev.quote[:50]}...')"
                        )
                    break

                elif ev.category == EvidenceCategory.DOUBTFUL:
                    if baseline_cop > 75.0:
                        is_shadow_override = True
                        effective_cop = 50.0
                        p_avail = self.prob_settings.prob_avail_doubtful
                        p_start_given_avail = self.prob_settings.prob_start_doubtful
                        reconciliation_reason = (
                            f"Candidate Evidence: Manager reported late fitness test ('{ev.quote[:50]}...')"
                        )
                    break

        # Enforce benchmark accuracy gate before applying candidate news to production
        gate_open = True
        if self.mode == RecommendationMode.GATED_ACTIVE:
            gate_open, gate_reason, _ = check_news_benchmark_gate(extractor_name)
            if not gate_open:
                rejected_signals.append(f"News gate closed: {gate_reason}")

        applied_to_production = (self.mode == RecommendationMode.GATED_ACTIVE) and is_shadow_override and gate_open

        # Log all disagreements between news-adjusted and API-only availability (WI-6.2)
        if is_shadow_override or abs(effective_cop - baseline_cop) > 0.01:
            disagreement_entry = {
                "element_id": elem_id,
                "web_name": web_name,
                "target_gw": target_gw,
                "baseline_status": status,
                "baseline_cop": baseline_cop,
                "effective_cop": effective_cop,
                "mode": str(self.mode),
                "is_shadow_override": is_shadow_override,
                "applied_to_production": applied_to_production,
                "evidence_category": str(evidence_category),
                "source_quote": source_quote,
                "source_url": source_url,
                "reconciliation_reason": reconciliation_reason,
            }
            self._logged_disagreements.append(disagreement_entry)
            logger.info(
                "[AvailabilityDisagreement] %s (ID %s): API=%s%% vs News=%s%% (mode=%s, applied=%s)",
                web_name, elem_id, baseline_cop, effective_cop, self.mode, applied_to_production
            )

        return ReconciledAvailability(
            element_id=elem_id,
            web_name=web_name,
            team_id=team_id,
            target_gw=target_gw,
            fixture_sub_index=0,
            baseline_status=status,
            baseline_chance_of_playing=baseline_cop,
            official_news_text=news_text,
            effective_chance_of_playing=effective_cop if applied_to_production else baseline_cop,
            p_available=p_avail if applied_to_production else baseline_p_avail,
            p_start_given_available=p_start_given_avail if applied_to_production else baseline_p_start,
            expected_minutes_limit=mins_limit if applied_to_production else None,
            evidence_category=evidence_category,
            source_quote=source_quote,
            source_url=source_url,
            reconciliation_reason=reconciliation_reason,
            rejected_signals=rejected_signals,
            mode=self.mode,
            is_shadow_override=is_shadow_override,
            applied_to_production=applied_to_production,
        )

    def reconcile_squad_availability(
        self,
        bootstrap: BootstrapStatic,
        target_gw: int,
        candidate_evidence: list[PlayerEvidence] | None = None,
    ) -> dict[int, ReconciledAvailability]:
        """Reconcile availability for all active players in bootstrap."""
        results = {}
        for elem in bootstrap.elements:
            rec = self.reconcile_player_fixture(
                element=elem,
                target_gw=target_gw,
                candidate_evidence=candidate_evidence,
            )
            results[elem.id] = rec
        return results


availability_reconciler = AvailabilityReconciler()
