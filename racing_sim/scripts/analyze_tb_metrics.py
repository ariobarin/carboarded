import os, sys
from collections import OrderedDict
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import numpy as np

RUNS = OrderedDict([
    ('A1: simple', 'logs/ppo_20260206_110420/PPO_1'),
    ('A2: supersimple', 'logs/ppo_20260206_110422/PPO_1'),
    ('A3: square_test', 'logs/ppo_20260206_112640/PPO_1'),
    ('A4: track4', 'logs/ppo_20260206_112641/PPO_1'),
    ('A5: track1', 'logs/ppo_20260206_114728/PPO_1'),
    ('A6: track2', 'logs/ppo_20260206_114730/PPO_1'),
    ('A7: track3', 'logs/ppo_20260206_121919/PPO_1'),
])

KL_SPIKE = 0.05

def load_scalars(logdir, tag):
    ea = EventAccumulator(logdir)
    ea.Reload()
    avail = ea.Tags().get('scalars', [])
    if tag not in avail: return None
    events = ea.Scalars(tag)
    return np.array([e.step for e in events]), np.array([e.value for e in events])

def split_el(vals, frac=0.25):
    n = len(vals)
    cutoff = max(1, int(n * frac))
    return vals[:cutoff], vals[-cutoff:]

def analyze(logdir):
    r = {}
    for tag_name, key, extras in [
        ('train/approx_kl', 'approx_kl', True),
        ('train/entropy_loss', 'entropy', True),
        ('train/value_loss', 'vloss', True),
        ('train/clip_fraction', 'clip', False),
        ('train/explained_variance', 'ev', True),
        ('train/policy_gradient_loss', 'pgloss', False),
        ('train/grad_norm', 'grad', False),
    ]:
        d = load_scalars(logdir, tag_name)
        if d is None: continue
        s, v = d
        info = dict(mean=float(np.mean(v)), mx=float(np.max(v)), mn=float(np.min(v)), n=len(v))
        if extras:
            e, l = split_el(v)
            info['early'] = float(np.mean(e))
            info['late'] = float(np.mean(l))
        if key == 'approx_kl':
            sp = v[v > KL_SPIKE]
            info['spikes'] = int(len(sp))
            info['spike_pct'] = float(len(sp)/len(v)*100) if len(v)>0 else 0
        if key == 'entropy':
            e2, l2 = split_el(v)
            info['start'] = float(np.mean(e2))
            info['end'] = float(np.mean(l2))
            info['delta'] = float(np.mean(l2) - np.mean(e2))
        r[key] = info
    ea2 = EventAccumulator(logdir)
    ea2.Reload()
    r['_tags'] = ea2.Tags().get('scalars', [])
    return r

def fmt(val, p=4):
    if val is None: return 'N/A'
    if abs(val) >= 100: return f'{val:.1f}'
    return f'{val:.{p}f}'

def main():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if not os.path.isdir(os.path.join(base, 'logs')):
        base = os.getcwd()
    S = '=' * 140
    D = '-' * 140
    print(S)
    print('PPO TRAINING METRICS ANALYSIS -- 7 runs (2026-02-06)')
    print(S); print()
    ar = {}
    for lab, rp in RUNS.items():
        ld = os.path.join(base, rp)
        if not os.path.isdir(ld):
            print(f'  WARNING: {ld} not found, skipping {lab}'); continue
        print(f'  Loading {lab} ... ', end='', flush=True)
        res = analyze(ld)
        ar[lab] = res
        n = res.get('approx_kl', {}).get('n', 0)
        print(f'done ({n} training updates)')
    print()
    tags = ar[list(ar.keys())[0]].get('_tags', [])
    print('Available TensorBoard scalar tags (first run):')
    for t in sorted(tags): print(f'  {t}')
    print()

    # Helpers for compact table
    def g(res, key, field):
        m = res.get(key)
        if m is None: return None
        return m.get(field)

    print(S)
    print('TABLE 1: approx_kl')
    print(S)
    hdr = f'{"Run":<20} {"Mean":>8} {"Max":>8} {"Early":>8} {"Late":>8} {"Spikes>0.05":>12} {"Spike%":>8} {"Trend":>12}'
    print(hdr); print(D)
    for lab, res in ar.items():
        m = res.get('approx_kl')
        if not m: print(f'{lab:<20} N/A'); continue
        tr = 'RISING' if m['late']>m['early']*1.2 else ('FALLING' if m['late']<m['early']*0.8 else 'STABLE')
        sf = ' *** ALERT' if m['spikes']>0 else ''
        mn_s, mx_s, ea_s, la_s = fmt(m["mean"]), fmt(m["mx"]), fmt(m["early"]), fmt(m["late"])
        sp_s = fmt(m["spike_pct"], 1)
        print(f'{lab:<20} {mn_s:>8} {mx_s:>8} {ea_s:>8} {la_s:>8} {m["spikes"]:>12} {sp_s:>8}% {tr:>12}{sf}')
    print()
    print(S)
    print('TABLE 2: entropy_loss (start vs end)')
    print(S)
    print(f'{"Run":<20} {"Mean":>10} {"Start":>10} {"End":>10} {"Delta":>10} {"Trend":>12}'); print(D)
    for lab, res in ar.items():
        m = res.get('entropy')
        if not m: print(f'{lab:<20} N/A'); continue
        tr = 'COLLAPSING' if m['end']>m['start']*0.7 and m['delta']>0 else ('HEALTHY' if m['delta']<0 else 'STABLE')
        print(f'{lab:<20} {fmt(m["mean"]):>10} {fmt(m["start"]):>10} {fmt(m["end"]):>10} {fmt(m["delta"]):>10} {tr:>12}')
    print()
    print(S)
    print('TABLE 3: value_loss')
    print(S)
    print(f'{"Run":<20} {"Mean":>10} {"Max":>10} {"Early":>10} {"Late":>10} {"Trend":>12}'); print(D)
    for lab, res in ar.items():
        m = res.get('vloss')
        if not m: print(f'{lab:<20} N/A'); continue
        tr = 'RISING' if m['late']>m['early']*1.5 else ('FALLING' if m['late']<m['early']*0.5 else 'STABLE')
        print(f'{lab:<20} {fmt(m["mean"]):>10} {fmt(m["mx"]):>10} {fmt(m["early"]):>10} {fmt(m["late"]):>10} {tr:>12}')
    print()
    print(S)
    print('TABLE 4: clip_fraction')
    print(S)
    print(f'{"Run":<20} {"Mean":>10} {"Max":>10} {"Assessment":>20}'); print(D)
    for lab, res in ar.items():
        m = res.get('clip')
        if not m: print(f'{lab:<20} N/A'); continue
        a = 'HIGH (>0.2)' if m['mean']>0.2 else ('MODERATE' if m['mean']>0.1 else 'HEALTHY')
        print(f'{lab:<20} {fmt(m["mean"]):>10} {fmt(m["mx"]):>10} {a:>20}')
    print()
    print(S)
    print('TABLE 5: explained_variance')
    print(S)
    print(f'{"Run":<20} {"Mean":>10} {"Min":>10} {"Early":>10} {"Late":>10} {"Assessment":>20}'); print(D)
    for lab, res in ar.items():
        m = res.get('ev')
        if not m: print(f'{lab:<20} N/A'); continue
        a = 'GOOD (>0.5)' if m['late']>0.5 else ('MODERATE' if m['late']>0.0 else 'POOR (<0)')
        print(f'{lab:<20} {fmt(m["mean"]):>10} {fmt(m["mn"]):>10} {fmt(m["early"]):>10} {fmt(m["late"]):>10} {a:>20}')
    print()
    print(S)
    print('TABLE 6: policy_gradient_loss')
    print(S)
    print(f'{"Run":<20} {"Mean":>10} {"Min":>10} {"Max":>10}'); print(D)
    for lab, res in ar.items():
        m = res.get('pgloss')
        if not m: print(f'{lab:<20} N/A'); continue
        print(f'{lab:<20} {fmt(m["mean"]):>10} {fmt(m["mn"]):>10} {fmt(m["mx"]):>10}')
    print()
    hg = any('grad' in res for res in ar.values())
    if hg:
        print(S)
        print('TABLE 7: grad_norm')
        print(S)
        print(f'{"Run":<20} {"Mean":>10} {"Max":>10}'); print(D)
        for lab, res in ar.items():
            m = res.get('grad')
            if not m: print(f'{lab:<20} N/A'); continue
            print(f'{lab:<20} {fmt(m["mean"]):>10} {fmt(m["mx"]):>10}')
        print()
    else:
        print('TABLE 7: grad_norm -- NOT LOGGED (SB3 default does not log this).'); print()
    print(S)
    print('KL SPIKE DIAGNOSTIC SUMMARY (threshold > 0.05)')
    print(S)
    any_sp = False
    for lab, res in ar.items():
        m = res.get('approx_kl')
        if m and m['spikes']>0:
            any_sp = True
            print(f'  {lab}: {m["spikes"]} spikes ({fmt(m["spike_pct"],1)}% of updates), max KL = {fmt(m["mx"])}')
    if not any_sp:
        print('  No KL spikes > 0.05 detected in any run.')
    print()
    print(S)
    print('COMPACT COMPARISON TABLE (all runs side by side)')
    print(S)
    cols = list(ar.keys())
    short = [c.split(':')[0].strip() for c in cols]
    lw, cw = 16, 14

    def row(name, key, field):
        vals = []
        for c in cols:
            m = ar[c].get(key)
            if m is None: vals.append('N/A'); continue
            v = m.get(field)
            vals.append(fmt(v) if v is not None else 'N/A')
        print(f'{name:<{lw}}' + ''.join(f'{v:>{cw}}' for v in vals))

    def rows(name, extractor):
        vals = []
        for c in cols:
            try: vals.append(str(extractor(ar[c])))
            except: vals.append('N/A')
        print(f'{name:<{lw}}' + ''.join(f'{v:>{cw}}' for v in vals))

    print(f'{"Metric":<{lw}}' + ''.join(f'{s:>{cw}}' for s in short))
    print('-' * (lw + cw * len(cols)))
    rows('Track', lambda r: list(RUNS.keys())[list(ar.values()).index(r)].split(':')[1].strip())
    rows('Updates', lambda r: r.get('approx_kl',{}).get('n','?'))
    print()
    row('KL mean', 'approx_kl', 'mean')
    row('KL max', 'approx_kl', 'mx')
    row('KL early', 'approx_kl', 'early')
    row('KL late', 'approx_kl', 'late')
    rows('KL spikes', lambda r: r.get('approx_kl',{}).get('spikes','N/A'))
    print()
    row('Ent start', 'entropy', 'start')
    row('Ent end', 'entropy', 'end')
    row('Ent delta', 'entropy', 'delta')
    print()
    row('VLoss mean', 'vloss', 'mean')
    row('VLoss max', 'vloss', 'mx')
    row('VLoss early', 'vloss', 'early')
    row('VLoss late', 'vloss', 'late')
    print()
    row('Clip mean', 'clip', 'mean')
    row('Clip max', 'clip', 'mx')
    print()
    row('EV mean', 'ev', 'mean')
    row('EV min', 'ev', 'mn')
    row('EV late', 'ev', 'late')
    print()
    row('PG loss', 'pgloss', 'mean')

    print()
    print(S)
    print('Analysis complete.')


if __name__ == '__main__':
    main()
