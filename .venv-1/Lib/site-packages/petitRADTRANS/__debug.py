"""Store debug functions.
Not to be used in released or committed codes.
"""
import linecache
import os
import tracemalloc

__megabyte = 1024 ** 2


def malloc_peak_snapshot(label: str = '', reset_peak: bool = True) -> None:
    """Display the traced memory and the peak memory since tracemalloc.start() in MiB.

    Args:
        label: label to be printed next to the display.
        reset_peak: if True, the memory peak will be reset.
    """
    size, peak = tracemalloc.get_traced_memory()
    size /= __megabyte
    peak /= __megabyte

    print(f"{label} {size=:.3f} MiB, {peak=:.3f} MiB")

    if reset_peak:
        tracemalloc.reset_peak()


def malloc_top_lines_snapshot(label: str = '', n_lines: int = 3) -> None:
    """Display the top lines in terms of memory usage, in MiB.

    Args:
        label: label to be printed next to the display.
        n_lines: number of lines to show.
    """
    def __display_top(_snapshot, key_type='lineno', _n_lines=n_lines):
        _snapshot = _snapshot.filter_traces((
            tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
            tracemalloc.Filter(False, "<unknown>"),
        ))
        top_stats = _snapshot.statistics(key_type)

        print("Top %s lines" % _n_lines)
        for index, stat in enumerate(top_stats[:_n_lines], 1):
            frame = stat.traceback[0]
            # replace "/path/to/module/file.py" with "module/file.py"
            filename = os.sep.join(frame.filename.split(os.sep)[-2:])
            print("#%s: %s:%s: %.3f MiB"
                  % (index, filename, frame.lineno, stat.size / __megabyte))
            line = linecache.getline(frame.filename, frame.lineno).strip()
            if line:
                print('    %s' % line)

        other = top_stats[_n_lines:]

        if other:
            size = sum(stat.size for stat in other)
            print("%s other: %.3f MiB" % (len(other), size / __megabyte))

        total = sum(stat.size for stat in top_stats)
        print("Total allocated size: %.3f MiB" % (total / __megabyte))

    snapshot = tracemalloc.take_snapshot()
    print(f'\n{label}')
    __display_top(snapshot)
    print('\n')
