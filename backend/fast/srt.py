from typing import List

def parse_srt_file(srt_path: str) -> List[dict]:
    segments = []
    with open(srt_path, 'r', encoding='utf-8') as f:
        content = f.read()
    blocks = content.strip().split('\n\n')
    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) >= 3:
            time_line = lines[1].strip()
            if ' --> ' in time_line:
                start_time, end_time = time_line.split(' --> ')
                start_seconds = srt_time_to_seconds(start_time)
                end_seconds = srt_time_to_seconds(end_time)
                text = ' '.join(lines[2:]).strip()
                if text:
                    segments.append({'start': start_seconds, 'end': end_seconds, 'text': text})
    return segments

def srt_time_to_seconds(time_str: str) -> float:
    try:
        time_part = time_str.split(',')[0]
        hours, minutes, seconds = map(int, time_part.split(':'))
        return hours * 3600 + minutes * 60 + seconds
    except Exception:
        return 0.0


