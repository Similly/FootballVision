from collections import defaultdict, deque
from .config import (
    VOTE_MIN_TOTAL_CONF, ONE_DIGIT_PENALTY, HYSTERESIS_MARGIN,
    DELTA_MIN, LEAKY_DECAY, USE_LEAKY
)

# Track histories & voting accumulators
track_history = defaultdict(lambda: deque(maxlen=50))

track_assigned_num = {}                         # tid -> int (current decision)
last_ocr_frame  = defaultdict(lambda: -10)
last_box_area   = defaultdict(lambda: 0.0)

vote_sum   = defaultdict(lambda: defaultdict(float))  # tid -> {number: weight}
vote_sum_offline = defaultdict(lambda: defaultdict(float))
total_conf = defaultdict(float)                       # tid -> sum conf over votes

def get_dominant_team(history_deque):
    if not history_deque:
        return None
    return max(set(history_deque), key=history_deque.count)

def add_vote(tid, num, conf):
    w = float(conf) * (1.0 if num >= 10 else ONE_DIGIT_PENALTY)
    vote_sum[tid][int(num)] += w
    vote_sum_offline[tid][int(num)] += w
    total_conf[tid] += float(conf)

def _top2_votes(d):
    if not d:
        return None, (None, 0.0)
    items = sorted(d.items(), key=lambda kv: kv[1], reverse=True)
    top1 = items[0]
    top2 = items[1] if len(items) > 1 else (None, 0.0)
    return top1, top2

def maybe_assign_number(tid):
    if total_conf[tid] < VOTE_MIN_TOTAL_CONF:
        return
    top1, _top2 = _top2_votes(vote_sum[tid])
    if top1 is None:
        return
    top1_num, top1_w = int(top1[0]), float(top1[1])
    prev = track_assigned_num.get(tid, None)
    if prev is None:
        track_assigned_num[tid] = int(top1[0])
        return
    if int(prev) != int(top1[0]):
        prev_w = float(vote_sum[tid].get(int(prev), 0.0))
        if (top1_w >= HYSTERESIS_MARGIN * prev_w) and (top1_w - prev_w >= DELTA_MIN):
            track_assigned_num[tid] = top1_num

def leaky_decay_all():
    if not USE_LEAKY:
        return
    for tid in list(vote_sum.keys()):
        for n in list(vote_sum[tid].keys()):
            vote_sum[tid][n] *= (1.0 - LEAKY_DECAY)
            if vote_sum[tid][n] < 1e-6:
                del vote_sum[tid][n]
        total_conf[tid] *= (1.0 - LEAKY_DECAY)
