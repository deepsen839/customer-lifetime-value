"""CausalML T-learner with a tested scikit-learn fallback and uplift evaluation."""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import train_test_split


def _uplift_summary(scored, treatment_col, outcome_col, method):
    scored = scored.copy()
    scored["uplift_decile"] = pd_qcut(scored["predicted_uplift"], 10)
    decile_effects = {}
    for decile, group in scored.groupby("uplift_decile", observed=True):
        treated = group.loc[group[treatment_col] == 1, outcome_col]
        control = group.loc[group[treatment_col] == 0, outcome_col]
        decile_effects[str(decile)] = float(treated.mean() - control.mean()) if len(treated) and len(control) else None
    sorted_scored = scored.sort_values("predicted_uplift", ascending=False)
    cumulative = []
    for frac in (0.1, 0.2, 0.3, 0.5, 0.75, 1.0):
        group = sorted_scored.head(max(1, int(np.ceil(len(sorted_scored) * frac))))
        treated = group.loc[group[treatment_col] == 1, outcome_col]
        control = group.loc[group[treatment_col] == 0, outcome_col]
        cumulative.append({"targeted_fraction": frac, "targeted_rows": int(len(group)),
            "treated_rows": int(len(treated)), "control_rows": int(len(control)),
            "observed_incremental_response_rate": float(treated.mean() - control.mean()) if len(treated) and len(control) else None,
            "mean_predicted_uplift": float(group["predicted_uplift"].mean())})
    tr = scored.loc[scored[treatment_col] == 1, outcome_col]
    co = scored.loc[scored[treatment_col] == 0, outcome_col]
    summary = {"method": method, "test_rows": int(len(scored)),
        "mean_predicted_uplift": float(scored["predicted_uplift"].mean()),
        "observed_test_treatment_rate": float(tr.mean()) if len(tr) else None,
        "observed_test_control_rate": float(co.mean()) if len(co) else None,
        "observed_test_difference": float(tr.mean() - co.mean()) if len(tr) and len(co) else None,
        "observed_uplift_by_decile": decile_effects, "cumulative_uplift_curve": cumulative,
        "interpretation": "Held-out randomized-arm differences are noisy estimates, not individual causal effects; validate with experiment and confidence intervals."}
    return scored, summary


def pd_qcut(values, q):
    unique = int(pd.Series(values).nunique(dropna=True))
    if unique <= 1:
        return pd.Series(["all_scores_equal"] * len(values), index=getattr(values, "index", None), dtype="object")
    return pd.qcut(values, q=min(q, unique), duplicates="drop")


def fit_uplift(df, feature_cols, treatment_col, outcome_col, estimator, seed=42):
    """Train a two-model T-learner and score a held-out split."""
    from sklearn.base import clone
    d = df[list(feature_cols) + [treatment_col, outcome_col]].dropna().copy()
    d[treatment_col] = pd.to_numeric(d[treatment_col], errors="raise").astype(int)
    d[outcome_col] = pd.to_numeric(d[outcome_col], errors="raise").astype(int)
    if d[treatment_col].nunique() != 2 or d[outcome_col].nunique() != 2:
        raise ValueError("Need binary treatment and outcome with both classes")
    # Stratify on treatment/outcome joint strata when every stratum is sufficiently populated.
    joint = d[treatment_col].astype(str) + "_" + d[outcome_col].astype(str)
    stratify = joint if joint.value_counts().min() >= 2 else d[treatment_col]
    tr, te = train_test_split(d, test_size=.25, random_state=seed, stratify=stratify)
    models = {}
    for arm in (0, 1):
        part = tr[tr[treatment_col] == arm]
        if part[outcome_col].nunique() != 2:
            raise ValueError(f"Training arm {arm} must contain both outcome classes")
        models[arm] = clone(estimator).fit(part[feature_cols], part[outcome_col])
    out = te.copy()
    out["p_treat"] = models[1].predict_proba(te[feature_cols])[:, 1]
    out["p_control"] = models[0].predict_proba(te[feature_cols])[:, 1]
    out["predicted_uplift"] = out["p_treat"] - out["p_control"]
    out, summary = _uplift_summary(out, treatment_col, outcome_col, "sklearn T-learner")
    return models, out, summary


def fit_causalml_t_learner(df, feature_cols, treatment_col, outcome_col, seed=42):
    """Use CausalML BaseTClassifier if importable, otherwise transparently fall back."""
    from sklearn.ensemble import RandomForestClassifier
    d = df[list(feature_cols) + [treatment_col, outcome_col]].dropna().copy()
    d[treatment_col] = pd.to_numeric(d[treatment_col], errors="raise").astype(int)
    d[outcome_col] = pd.to_numeric(d[outcome_col], errors="raise").astype(int)
    if d[treatment_col].nunique() != 2 or d[outcome_col].nunique() != 2:
        raise ValueError("Need binary treatment and outcome with both classes")
    joint = d[treatment_col].astype(str) + "_" + d[outcome_col].astype(str)
    stratify = joint if joint.value_counts().min() >= 2 else d[treatment_col]
    train, test = train_test_split(d, test_size=.25, random_state=seed, stratify=stratify)
    Xtr = train[feature_cols].to_numpy(dtype=float); Xte = test[feature_cols].to_numpy(dtype=float)
    t = train[treatment_col].to_numpy(dtype=int); y = train[outcome_col].to_numpy(dtype=int)
    base = RandomForestClassifier(n_estimators=160, min_samples_leaf=20, random_state=seed, n_jobs=-1)
    try:
        from causalml.inference.meta import BaseTClassifier
        model = BaseTClassifier(learner=base, control_name=0)
        model.fit(X=Xtr, treatment=t, y=y)
        raw = np.asarray(model.predict(Xte))
        uplift = raw.reshape(len(test), -1)[:, 0]
        scored = test.copy(); scored["predicted_uplift"] = uplift
        scored, summary = _uplift_summary(scored, treatment_col, outcome_col, "CausalML BaseTClassifier")
        # Refit a deployable final model on all labelled data after keeping the
        # held-out scores strictly separate for evaluation.
        final_model = BaseTClassifier(learner=RandomForestClassifier(n_estimators=160, min_samples_leaf=20, random_state=seed, n_jobs=-1), control_name=0)
        final_model.fit(X=d[feature_cols].to_numpy(dtype=float), treatment=d[treatment_col].to_numpy(dtype=int), y=d[outcome_col].to_numpy(dtype=int))
        return final_model, scored, summary
    except Exception as exc:
        # Keep an actual sklearn fallback, and record why CausalML did not run.
        _, scored, summary = fit_uplift(d, feature_cols, treatment_col, outcome_col, base, seed)
        final_models = {}
        for arm in (0, 1):
            part = d[d[treatment_col] == arm]
            if part[outcome_col].nunique() != 2:
                raise ValueError(f"Full-data arm {arm} must contain both outcome classes")
            final_models[arm] = clone(base).fit(part[feature_cols], part[outcome_col])
        summary["method"] = "sklearn T-learner fallback"
        summary["causalml_fallback_reason"] = f"{type(exc).__name__}: {str(exc)[:240]}"
        return final_models, scored, summary


def score_uplift_customers(model, customers, feature_cols):
    """Score a full eligible customer frame using the refitted final uplift model."""
    X = customers[feature_cols].to_numpy(dtype=float)
    if isinstance(model, dict):
        p1 = model[1].predict_proba(customers[feature_cols])[:, 1]
        p0 = model[0].predict_proba(customers[feature_cols])[:, 1]
        uplift = p1 - p0
    else:
        uplift = np.asarray(model.predict(X)).reshape(len(customers), -1)[:, 0]
    out = customers.copy()
    out["predicted_uplift"] = np.asarray(uplift, dtype=float)
    if isinstance(model, dict):
        out["p_treat"] = model[1].predict_proba(customers[feature_cols])[:, 1]
        out["p_control"] = model[0].predict_proba(customers[feature_cols])[:, 1]
    else:
        # BaseTClassifier returns treatment effect. For ROI only, estimate the
        # treated response probability by adding it to the observed control-arm
        # population rate; this is an explicit aggregate baseline, not individual
        # control probabilities. The scikit-learn fallback has individual p0/p1.
        if "treatment" in customers and "conversion" in customers:
            control = customers.loc[customers["treatment"].astype(int) == 0, "conversion"]
            baseline = float(pd.to_numeric(control, errors="coerce").mean()) if len(control) else 0.0
        else:
            baseline = 0.0
        out["p_control"] = baseline
        out["p_treat"] = (out["p_control"] + out["predicted_uplift"]).clip(0, 1)
    return out.sort_values("predicted_uplift", ascending=False).reset_index(drop=True)
