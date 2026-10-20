"""ROI calculations based on held-out uplift scores; assumptions remain explicit."""
from __future__ import annotations
import math
import pandas as pd

def uplift_campaign_roi(scored: pd.DataFrame, targeted_fraction: float, discount_per_conversion: float,
                        contribution_per_conversion: float, contact_cost_per_customer: float = 0.0,
                        contribution_margin: float = 1.0) -> dict:
    """Estimate campaign economics for the top predicted-uplift fraction.

    The offer discount is charged only on expected conversions under treatment.
    Contribution is the expected contribution before discount per incremental
    conversion. Since the experiment measures conversion uplift (not retention),
    this is campaign ROI, not churn-retention ROI.
    """
    if not 0 < targeted_fraction <= 1:
        raise ValueError("targeted_fraction must be in (0, 1]")
    if min(discount_per_conversion, contribution_per_conversion, contact_cost_per_customer) < 0:
        raise ValueError("Costs and contribution must be non-negative")
    if not 0 <= contribution_margin <= 1:
        raise ValueError("contribution_margin must be between 0 and 1")
    if "predicted_uplift" not in scored or scored.empty:
        raise ValueError("No scored uplift customers available")
    n = max(1, int(math.ceil(len(scored) * targeted_fraction)))
    target = scored.sort_values("predicted_uplift", ascending=False).head(n)
    uplift = target["predicted_uplift"].clip(lower=0).astype(float)
    expected_incremental_conversions = float(uplift.sum())
    if "p_treat" in target:
        expected_treated_conversions = float(target["p_treat"].clip(0, 1).sum())
    else:
        expected_treated_conversions = float((target["p_control"].clip(0, 1) + uplift).clip(0, 1).sum())
    contact_cost = n * contact_cost_per_customer
    discount_cost = expected_treated_conversions * discount_per_conversion
    total_cost = contact_cost + discount_cost
    incremental_value = expected_incremental_conversions * contribution_per_conversion * contribution_margin
    net_benefit = incremental_value - total_cost
    return {"targeted_fraction": float(targeted_fraction), "scored_customers": int(len(scored)),
        "targeted_customers": int(n), "mean_predicted_uplift": float(target["predicted_uplift"].mean()),
        "expected_incremental_conversions": expected_incremental_conversions,
        "expected_treated_conversions": expected_treated_conversions,
        "contact_cost": float(contact_cost), "discount_cost": float(discount_cost),
        "total_offer_cost": float(total_cost), "expected_incremental_contribution": float(incremental_value),
        "net_benefit": float(net_benefit), "roi_pct": float(net_benefit / total_cost * 100) if total_cost > 0 else None,
        "assumptions": {"uplift_scores_are_estimates": True, "negative_uplift_clipped_for_value": True,
            "discount_applies_to_expected_treated_conversions": True, "contribution_margin": float(contribution_margin),
            "interpretation": "Scenario estimate based on held-out Hillstrom campaign scores; validate with a randomized campaign and uncertainty intervals."}}
