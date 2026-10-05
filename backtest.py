import numpy as np
import pandas as pd
from scipy.stats import chi2
from .models import forecasts
from .stats import kupiec_pof, christoffersen_ind, traffic_light


def run(r, alpha=0.01, window=250) -> tuple[pd.DataFrame, dict]:
    r = np.asarray(r, float); f = forecasts(r, alpha, window); rows = []
    for name, v in f.items():
        ok = ~np.isnan(v); b = (r[ok] < v[ok]).astype(int)
        lr_u, p_u = kupiec_pof(b, alpha); lr_i, p_i = christoffersen_ind(b)
        rows.append(dict(model=name, days=len(b), breaches=int(b.sum()), expected=round(alpha * len(b), 1),
                         kupiec_p=p_u, independence_p=p_i, cond_coverage_p=float(chi2.sf(lr_u + lr_i, 2)),
                         basel_zone=traffic_light(b[-250:]) if alpha == 0.01 else "n/a", avg_var=float(-v[ok].mean())))
    df = pd.DataFrame(rows); df["passes"] = (df.kupiec_p > 0.05) & (df.independence_p > 0.05)
    return df, f
