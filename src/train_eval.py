"""Train CNN-BiLSTM-Attention on 24 wind-speed lags, report MAE/RMSE/MAPE/R2,
then explain it with SHAP, LIME and attention weights and compare rankings (Spearman)."""
import argparse, json, os
import numpy as np, torch
from scipy.stats import spearmanr
from .data import load_scada_csv, synthetic_scada, make_windows, chrono_split, Scaler
from .metrics import regression_metrics
from .model import CNNBiLSTMAttention
from .trainutil import fit, predict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/synthetic_scada.csv" if os.path.exists("data/synthetic_scada.csv") else None, help="SCADA-style CSV (default: committed synthetic CSV)"); ap.add_argument("--lags", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=15); ap.add_argument("--quick", action="store_true")
    ap.add_argument("--n_background", type=int, default=300); ap.add_argument("--n_explain", type=int, default=500)
    ap.add_argument("--n_lime", type=int, default=50); ap.add_argument("--out", default="results")
    a = ap.parse_args()
    if a.quick: a.epochs, a.n_background, a.n_explain, a.n_lime = 5, 100, 100, 10
    df = load_scada_csv(a.csv) if a.csv else synthetic_scada()
    df = df[["wind_speed"]]          # paper: lagged wind speed as the 24 input features
    X, y = make_windows(df, lookback=a.lags)
    tr, va, te = chrono_split(len(X))
    sc = Scaler().fit(X[tr]); Xs = sc.transform(X).astype("float32")
    ym, ys = y[tr].mean(), y[tr].std(); yn = ((y - ym) / ys).astype("float32")
    model = fit(CNNBiLSTMAttention(1), Xs[tr], yn[tr], Xs[va], yn[va], epochs=a.epochs)
    pred = predict(model, Xs[te]) * ys + ym
    res = {"metrics_test": regression_metrics(y[te], pred),
           "persistence_test": regression_metrics(y[te], X[te][:, -1, 0])}
    names = [f"t-{a.lags - i}" for i in range(a.lags)]   # index i -> lag

    # --- attention ---
    predict(model, Xs[te][: a.n_explain]); att = model.last_attention.numpy().mean(0)
    # --- SHAP (GradientExplainer: expected gradients) ---
    import shap
    rng = np.random.default_rng(0)
    bg = torch.tensor(Xs[tr][rng.choice(tr.stop, a.n_background, replace=False)])
    ex = torch.tensor(Xs[te][: a.n_explain])
    wrapped = torch.nn.Sequential(model, torch.nn.Unflatten(0, (-1, 1)))
    sv = shap.GradientExplainer(wrapped, bg).shap_values(ex)
    sv = np.asarray(sv).reshape(len(ex), a.lags)
    shap_imp = np.abs(sv).mean(0)
    # --- LIME ---
    from lime.lime_tabular import LimeTabularExplainer
    f2 = lambda Z: predict(model, Z.reshape(-1, a.lags, 1).astype("float32"))
    lx = LimeTabularExplainer(Xs[tr].reshape(-1, a.lags), feature_names=names, mode="regression",
                              discretize_continuous=False, random_state=0)
    lime_imp = np.zeros(a.lags)
    for i in range(a.n_lime):
        e = lx.explain_instance(Xs[te][i].ravel(), f2, num_features=a.lags, num_samples=1000)
        for j, w in e.as_map()[0]: lime_imp[j] += abs(w)
    lime_imp /= a.n_lime
    nrm = lambda v: (v / v.max()).round(4).tolist()
    res["importance_normalised"] = {"lags": names, "shap": nrm(shap_imp), "lime": nrm(lime_imp), "attention": nrm(att)}
    res["spearman"] = {"shap_vs_lime": float(spearmanr(shap_imp, lime_imp)[0]),
                       "shap_vs_attention": float(spearmanr(shap_imp, att)[0]),
                       "lime_vs_attention": float(spearmanr(lime_imp, att)[0])}
    res["top3"] = {k: [names[i] for i in np.argsort(-v)[:3]] for k, v in
                   {"shap": shap_imp, "lime": lime_imp, "attention": att}.items()}
    os.makedirs(a.out, exist_ok=True); json.dump(res, open(f"{a.out}/results.json", "w"), indent=2)
    print(json.dumps({k: res[k] for k in ("metrics_test", "persistence_test", "spearman", "top3")}, indent=2))


if __name__ == "__main__":
    main()
