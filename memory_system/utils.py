from typing import Iterable
def moving_average(seq: Iterable[float], window: int) -> float:
    seq = list(seq)[-window:]
    return sum(seq) / max(1, len(seq))