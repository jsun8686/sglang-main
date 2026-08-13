"""
shm_mem C library binding for NPU multi-tier memory.
"""
import ctypes
import os
from enum import IntEnum
from typing import List, Optional, Tuple

PATH_MAX_LEN = 256

class shm_type_t(IntEnum):
    SHM_NEAR = 0
    SHM_FAR = 1
    SHM_BUFF = 2


class shm_desc_t(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("dev_path", ctypes.c_char * PATH_MAX_LEN),
        ("size", ctypes.c_size_t),
        ("ptr", ctypes.c_void_p),
        ("dev_ptr", ctypes.c_void_p),
    ]


class ShmMemAdapter:
    def __init__(self, lib_path: str = "./libshm_mem.so"):
        self.lib = ctypes.CDLL(os.path.abspath(lib_path))
        self._setup_function_signatures()
        self.mapped_descs: List[shm_desc_t] = []

    def _setup_function_signatures(self):
        self.lib.shm_mem_alloc.restype = ctypes.c_int
        self.lib.shm_mem_alloc.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_size_t)]

        self.lib.shm_mem_get.restype = ctypes.c_int
        self.lib.shm_mem_get.argtypes = [ctypes.c_char_p, ctypes.POINTER(shm_desc_t), ctypes.POINTER(ctypes.c_uint32)]

        self.lib.shm_mem_free.restype = ctypes.c_int
        self.lib.shm_mem_free.argtypes = [ctypes.c_char_p]

        # 新增 C 接口声明
        self.lib.shm_mem_mmap.restype = ctypes.c_int
        self.lib.shm_mem_mmap.argtypes = [ctypes.POINTER(shm_desc_t), ctypes.c_int]

        self.lib.shm_mem_unmap.restype = ctypes.c_int
        self.lib.shm_mem_unmap.argtypes = [ctypes.POINTER(shm_desc_t)]

    def alloc(self, shm_name: str, sizes: List[int]) -> int:
        size_list = (ctypes.c_size_t * len(sizes))(*sizes)
        return self.lib.shm_mem_alloc(shm_name.encode('utf-8'), size_list)

    def get(self, shm_name: str) -> Tuple[List[Optional[shm_desc_t]], int]:
        c_descs = (shm_desc_t * shm_type_t.SHM_BUFF)()
        desc_num = ctypes.c_uint32(shm_type_t.SHM_BUFF)

        ret = self.lib.shm_mem_get(shm_name.encode('utf-8'), c_descs, ctypes.byref(desc_num))
        if ret != 0:
            raise RuntimeError(f"shm_mem_get failed: {ret}")

        result_list: List[Optional[shm_desc_t]] = [None] * shm_type_t.SHM_BUFF
        for i in range(desc_num.value):
            desc_item = c_descs[i]
            if 0 <= desc_item.type < shm_type_t.SHM_BUFF:
                result_list[desc_item.type] = desc_item

        return result_list, desc_num.value

    def mmap_region(self, desc: Optional[shm_desc_t], pin: bool = True) -> Optional[int]:
        if desc is None:
            return None

        is_pin = 1 if pin else 0
        ret = self.lib.shm_mem_mmap(ctypes.byref(desc), is_pin)
        if ret != 0:
            raise RuntimeError(f"C layer shm_mem_mmap failed for {desc.dev_path.decode()}")

        self.mapped_descs.append(desc)
        return desc.ptr

    def free(self, shm_name: str) -> int:
        for desc in self.mapped_descs:
            self.lib.shm_mem_unmap(ctypes.byref(desc))
        self.mapped_descs.clear()

        return self.lib.shm_mem_free(shm_name.encode('utf-8'))


if __name__ == "__main__":
    adapter = ShmMemAdapter("./libshm_mem.so")
    shm_pool_name = "npu_mem_demo"

    print("--- 1. Allocating SHM (single tier) ---")
    # Only allocate SHM_NEAR; far tier size is 0.
    alloc_ret = adapter.alloc(shm_pool_name, [8192, 0])
    print(f"Alloc return code: {alloc_ret}")

    print("\n--- 2. Getting SHM Descriptors ---")
    descs, count = adapter.get(shm_pool_name)
    print(f"Successfully retrieved {count} regions.")

    print("\n--- 3. Mapping via Business Logic ---")
    host_ptr = adapter.mmap_region(descs[shm_type_t.SHM_NEAR], pin=True)

    test_data = b"Hello NPU Shm!"
    if host_ptr is not None:
        print(f"  [SHM_HOST] Address: {host_ptr}, Size: {descs[shm_type_t.SHM_NEAR].size} bytes")
        ctypes.memmove(host_ptr, test_data, len(test_data))
        print("  -> Write data to SHM_HOST success.")
        read_back = ctypes.string_at(host_ptr, len(test_data))
        print(f"  -> Read back from SHM_HOST: {read_back}")
    else:
        print("  [SHM_HOST] Not allocated.")

    if descs[shm_type_t.SHM_FAR] is not None:
        print("  [SHM_FAR] Unexpectedly allocated in single-tier mode.")
    else:
        print("  [SHM_FAR] Skipped as expected in single-tier mode.")

    print("\n--- 5. Freeing SHM ---")
    free_ret = adapter.free(shm_pool_name)
    print(f"Free return code: {free_ret} (System /dev/shm files cleaned)")