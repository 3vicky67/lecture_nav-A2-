from typing import List, Dict

def build_overlapping_windows(
    segments: List[dict],
    window_seconds: int = 45,
    overlap_seconds: int = 15,
) -> List[Dict[str, object]]:
    if not segments:
        return []
    total_end = float(max(s.get("end", 0.0) for s in segments))
    win_start = 0.0
    windows: List[Dict[str, object]] = []
    idx = 0
    while win_start < total_end:
        win_end = win_start + float(window_seconds)
        collected: List[str] = []
        for s in segments:
            s_start = float(s.get("start", 0.0))
            s_end = float(s.get("end", 0.0))
            if s_end > win_start and s_start < win_end:
                txt = (s.get("text") or "").strip()
                if txt:
                    collected.append(txt)
        segment_text = " ".join(collected).strip() or "<empty_segment>"
        windows.append({
            "segment_text": segment_text,
            "t_start": round(win_start, 2),
            "t_end": round(min(win_end, total_end), 2),
            "idx": idx,
        })
        idx += 1
        win_start = max(win_end - float(overlap_seconds), win_start + 1.0)
    return windows

def choose_optimal_window_size(video_duration: float) -> int:
    if video_duration < 600:
        return 30
    elif video_duration < 1800:
        return 45
    else:
        return 60


