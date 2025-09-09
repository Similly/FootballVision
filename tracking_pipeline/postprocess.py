from collections import defaultdict, Counter

def postprocess_and_write(output_rows, vote_sum_offline, mot_file_path):
    # 1) retro-fill jersey numbers per tid with offline winner
    final_numbers = {}
    for tid in set([row[1] for row in output_rows]):
        if vote_sum_offline[tid]:
            best_num, _w = max(vote_sum_offline[tid].items(), key=lambda kv: kv[1])
            final_numbers[tid] = int(best_num)

    filled_rows = []
    for row in output_rows:
        frame_i, tid, x, y, w, h, conf_o, extra, jersey = row
        if extra in ('L', 'R') and tid in final_numbers:
            jersey = int(final_numbers[tid])
        filled_rows.append((frame_i, tid, x, y, w, h, conf_o, extra, jersey))

    # 2) per-tid team majority
    team_votes = defaultdict(list)
    for frame_i, tid, x, y, w, h, conf_o, extra, jersey in filled_rows:
        if extra in ('L', 'R'):
            team_votes[tid].append(extra)

    team_final = {}
    for tid, votes in team_votes.items():
        if votes:
            team_final[tid] = Counter(votes).most_common(1)[0][0]

    # 3) final jersey per tid (stable after fill)
    jersey_final = {}
    for frame_i, tid, x, y, w, h, conf_o, extra, jersey in filled_rows:
        if extra in ('L', 'R') and jersey is not None and jersey != -1:
            jersey_final[tid] = int(jersey)

    # 4) tid -> key (team, jersey)
    tid_key = {}
    for tid in set([r[1] for r in filled_rows]):
        t = team_final.get(tid, None)
        j = jersey_final.get(tid, None)
        if t in ('L', 'R') and j is not None and 1 <= j <= 99:
            tid_key[tid] = (t, j)

    # spans & first/last per tid
    tid_span = {}
    tid_first = {}
    tid_last  = {}

    for frame_i, tid, x, y, w, h, conf_o, extra, jersey in filled_rows:
        if extra in ('L', 'R'):
            if tid not in tid_span:
                tid_span[tid] = [frame_i, frame_i]
                tid_first[tid] = (frame_i, (x, y, w, h))
            else:
                tid_span[tid][1] = frame_i
            tid_last[tid] = (frame_i, (x, y, w, h))

    # key -> tids
    key_to_tids = defaultdict(list)
    for tid in set([r[1] for r in filled_rows]):
        t = team_final.get(tid, None)
        j = jersey_final.get(tid, None)
        if t in ('L', 'R') and j is not None and 1 <= j <= 99:
            key_to_tids[(t, j)].append(tid)

    # stitch non-overlapping tids per key
    tid_new_id = {}
    for key, tids in key_to_tids.items():
        tids = [t for t in tids if t in tid_span and t in tid_first and t in tid_last]
        if not tids:
            continue
        tids_sorted = sorted(tids, key=lambda t: tid_span[t][0])

        chains = []
        for t in tids_sorted:
            placed = False
            t_start, t_end = tid_span[t]
            for chain in chains:
                prev = chain[-1]
                _p_start, p_end = tid_span[prev]
                if p_end < t_start:
                    chain.append(t)
                    placed = True
                    break
            if not placed:
                chains.append([t])

        for chain in chains:
            new_id = min(chain)
            for t in chain:
                tid_new_id[t] = new_id

    # relabel rows
    relabeled_rows = []
    for frame_i, tid, x, y, w, h, conf_o, extra, jersey in filled_rows:
        if extra in ('L', 'R'):
            team_out = team_final.get(tid, extra)
            new_id = tid_new_id.get(tid, tid)
            relabeled_rows.append((frame_i, new_id, x, y, w, h, conf_o, team_out, jersey))
        else:
            relabeled_rows.append((frame_i, tid, x, y, w, h, conf_o, extra, jersey))

    # write file
    with open(mot_file_path, 'w') as mot_writer:
        for r in relabeled_rows:
            frame_i, tid, x, y, w, h, conf_o, extra, jersey = r
            mot_writer.write(f"{frame_i},{tid},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{conf_o:.2f},{extra},{jersey}\n")

    return relabeled_rows
