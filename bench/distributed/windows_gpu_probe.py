"""Measure available Windows RAM/VRAM and pinned host-to-device copies.

Run from the checkout with 64-bit Python 3.8+ and the installed NVIDIA driver:
    py -3 bench/distributed/windows_gpu_probe.py --output windows-gpu.json
Use --device N for a different CUDA ordinal. No model, toolkit or server required.
"""

import argparse
import ctypes as C
import datetime
import json
import math
import platform
import statistics
import sys
import time
from contextlib import ExitStack
from pathlib import Path


MIB = 1024 * 1024
TRANSFER_BYTES = (2 * MIB, 20 * MIB)
WARMUP = 5
SAMPLES = 30


class MemoryStatus(C.Structure):
    _fields_ = [("length", C.c_uint32), ("load", C.c_uint32),
                ("total_physical", C.c_uint64), ("available_physical", C.c_uint64),
                ("total_page", C.c_uint64), ("available_page", C.c_uint64),
                ("total_virtual", C.c_uint64), ("available_virtual", C.c_uint64),
                ("available_extended", C.c_uint64)]


def windows_ram():
    kernel = C.WinDLL("kernel32.dll", use_last_error=True, winmode=0x800)  # system32
    query = kernel.GlobalMemoryStatusEx
    query.argtypes = [C.POINTER(MemoryStatus)]
    query.restype = C.c_int
    data = MemoryStatus()
    data.length = C.sizeof(data)
    if not query(C.byref(data)):
        raise C.WinError(C.get_last_error())
    return {"total_bytes": data.total_physical, "available_bytes": data.available_physical}


def load_driver():
    driver = C.WinDLL("nvcuda.dll", winmode=0x800)
    pointer = C.POINTER
    deviceptr = C.c_uint64
    signature = {
        "cuInit": [C.c_uint], "cuDriverGetVersion": [pointer(C.c_int)],
        "cuDeviceGet": [pointer(C.c_int), C.c_int],
        "cuDeviceGetName": [pointer(C.c_char), C.c_int, C.c_int],
        "cuCtxCreate_v2": [pointer(C.c_void_p), C.c_uint, C.c_int],
        "cuCtxDestroy_v2": [C.c_void_p], "cuCtxSynchronize": [],
        "cuMemGetInfo_v2": [pointer(C.c_size_t), pointer(C.c_size_t)],
        "cuMemHostAlloc": [pointer(C.c_void_p), C.c_size_t, C.c_uint],
        "cuMemFreeHost": [C.c_void_p],
        "cuMemAlloc_v2": [pointer(deviceptr), C.c_size_t], "cuMemFree_v2": [deviceptr],
        "cuMemcpyHtoD_v2": [deviceptr, C.c_void_p, C.c_size_t],
    }

    def checked(code, function, args):
        if code:
            raise RuntimeError(f"{function.__name__} returned CUDA error {code}")
        return code

    for name, args in signature.items():
        func = getattr(driver, name)
        func.argtypes = args
        func.restype = C.c_int
        func.errcheck = checked
    return driver


def measure(driver, device_ordinal, query_ram=windows_ram):
    driver.cuInit(0)
    device, version = C.c_int(), C.c_int()
    driver.cuDriverGetVersion(C.byref(version))
    driver.cuDeviceGet(C.byref(device), device_ordinal)
    name = C.create_string_buffer(256)
    driver.cuDeviceGetName(name, len(name), device)
    with ExitStack() as cleanup:
        ctx = C.c_void_p()
        driver.cuCtxCreate_v2(C.byref(ctx), 0, device)
        cleanup.callback(driver.cuCtxDestroy_v2, ctx)
        free, total = C.c_size_t(), C.c_size_t()
        driver.cuMemGetInfo_v2(C.byref(free), C.byref(total))
        ram = query_ram()
        report = {
            "schema_version": 1, "method": "pinned_h2d_copy_and_context_sync_wall_clock",
            "captured_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "os": platform.platform(), "python": platform.python_version(),
            "device": name.value.decode("utf-8", "replace"), "device_ordinal": device_ordinal,
            "cuda_driver_api_version": version.value, "ram": ram,
            "vram_after_context": {"free_bytes": free.value, "total_bytes": total.value},
            "warmups_per_size": WARMUP, "samples_per_size": SAMPLES, "results": [],
        }
        capacity = max(TRANSFER_BYTES)
        if min(free.value, ram["available_bytes"]) < 2 * capacity:
            report["skipped"] = "less than twice the copy buffer available on host or device"
            return report
        host, gpu = C.c_void_p(), C.c_uint64()
        driver.cuMemHostAlloc(C.byref(host), capacity, 0)
        cleanup.callback(driver.cuMemFreeHost, host)
        driver.cuMemAlloc_v2(C.byref(gpu), capacity)
        cleanup.callback(driver.cuMemFree_v2, gpu)
        C.memset(host, 0x5A, capacity)  # touch all pinned pages before measurement
        for size in TRANSFER_BYTES:
            elapsed = []
            for i in range(WARMUP + SAMPLES):
                start = time.perf_counter_ns()
                driver.cuMemcpyHtoD_v2(gpu, host, size)
                driver.cuCtxSynchronize()
                ns = time.perf_counter_ns() - start
                if i >= WARMUP:
                    elapsed.append(ns)
            report["results"].append({
                "bytes": size, "elapsed_ns": elapsed, "median_ms": statistics.median(elapsed) / 1e6,
                "p95_ms_nearest_rank": sorted(elapsed)[math.ceil(SAMPLES * 0.95) - 1] / 1e6,
            })
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", type=Path, help="write UTF-8 JSON to a new file")
    args = parser.parse_args()
    if sys.platform != "win32" or C.sizeof(C.c_void_p) != 8:
        parser.exit(1, "This probe requires 64-bit Python on Windows.\n")
    try:
        report = measure(load_driver(), args.device)
        payload = json.dumps(report, indent=2) + "\n"
        if args.output is None:
            sys.stdout.write(payload)
        else:
            with args.output.open("x", encoding="utf-8") as out:
                out.write(payload)
    except (OSError, RuntimeError, AttributeError) as err:
        parser.exit(1, f"GPU probe failed: {err}\n")


if __name__ == "__main__":
    main()
