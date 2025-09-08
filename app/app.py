# avant
base, top3, top6 = risk_scores(hour, fc)
# après
base, top3, top6 = risk_scores(hour, fc, horizon_h=(3,6), low_thr=0.20, high_thr=0.80)
