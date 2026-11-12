"""Bounded, deterministic latency statistics for arbitrarily long jobs."""
import math
import os


class LatencyHistogram:
    """Logarithmic bins: quantiles are upper bounds, within 1% (floor 1 us)."""
    def __init__(self):
        self.bins = {}
        self.count = 0
        self.total = 0.
        self.maximum = 0.

    def add(self, seconds):
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError('Latency must be finite and nonnegative')
        bucket = max(0, math.ceil(math.log(max(seconds, 1e-6)/1e-6)/math.log(1.01)))
        self.bins[bucket] = self.bins.get(bucket, 0) + 1
        self.count += 1
        self.total += seconds
        self.maximum = max(self.maximum, seconds)

    def quantile(self, fraction):
        if not 0 <= fraction <= 1:
            raise ValueError('Quantile must be in [0, 1]')
        if not self.count:
            return None
        target = max(1, math.ceil(self.count*fraction))
        seen = 0
        for bucket, count in sorted(self.bins.items()):
            seen += count
            if seen >= target:
                return min(self.maximum, 1e-6*1.01**bucket)

    def summary(self):
        return {'count': self.count, 'mean_seconds': self.total/self.count if self.count else None,
                'p50_seconds': self.quantile(.5), 'p95_seconds': self.quantile(.95),
                'max_seconds': self.maximum if self.count else None,
                'quantile_method': 'log histogram upper bound; 1% bins, 1 us floor'}


def process_rss_bytes():
    """Current process RSS; unavailable platforms report None, never a fake zero."""
    try:
        if os.name == 'nt':
            import ctypes
            from ctypes import wintypes
            class Counters(ctypes.Structure):
                _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD)] + [
                    (name, ctypes.c_size_t) for name in ('PeakWorkingSetSize', 'WorkingSetSize',
                    'QuotaPeakPagedPoolUsage', 'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage',
                    'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage')]
            counters = Counters()
            counters.cb = ctypes.sizeof(counters)
            kernel = ctypes.WinDLL('kernel32', use_last_error=True)
            kernel.GetCurrentProcess.restype = wintypes.HANDLE
            psapi = ctypes.WinDLL('psapi', use_last_error=True)
            psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
            if psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
                return int(counters.WorkingSetSize)
        else:
            from pathlib import Path
            pages = int(Path('/proc/self/statm').read_text().split()[1])
            return pages*os.sysconf('SC_PAGE_SIZE')
    except (OSError, ValueError, AttributeError, IndexError):
        pass
    return None
