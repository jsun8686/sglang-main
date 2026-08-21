#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <errno.h>
#include <acl/acl.h>

// ==========================================
// 结构体与宏定义
// ==========================================

#define SHM_NAME_MAX_LEN 128
#define PATH_MAX_LEN     256

typedef enum {
    shm_near = 0,
    shm_far,
    shm_buff
} shm_type_t;

typedef struct {
    shm_type_t type;
    char dev_path[PATH_MAX_LEN];
    size_t size;
    void *ptr;      // host virtual address (mmap result)
    void *dev_ptr;  // NPU device address (aclrtHostRegister result)
} shm_desc_t;

static const char* shm_type_strs[] = {
    "_near",
    "_far"
};

static void format_shm_name(char *dest, size_t dest_len, const char *shm_name, shm_type_t type) {
    snprintf(dest, dest_len, "/%s%s", shm_name, shm_type_strs[type]);
}

static void format_dev_path(char *dest, size_t dest_len, const char *shm_name, shm_type_t type) {
    snprintf(dest, dest_len, "/dev/shm/%s%s", shm_name, shm_type_strs[type]);
}

static size_t page_align_up(size_t size) {
    long page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0) {
        page_size = 4096; // 兜底默认 4KB
    }
    return (size + page_size - 1) & ~(page_size - 1);
}

// ==========================================
// 核心接口实现
// ==========================================

int shm_mem_alloc(char *shm_name, size_t shm_size_list[shm_buff]) {
    if (!shm_name || !shm_size_list) {
        return -1;
    }

    for (int i = 0; i < shm_buff; i++) {
        size_t size = page_align_up(shm_size_list[i]);
        if (size == 0) continue;

        char posix_shm_name[SHM_NAME_MAX_LEN];
        format_shm_name(posix_shm_name, sizeof(posix_shm_name), shm_name, (shm_type_t)i);

        int fd = shm_open(posix_shm_name, O_CREAT | O_RDWR | O_TRUNC, 0666);
        if (fd == -1) {
            perror("shm_open alloc failed");
            return -1;
        }

        if (ftruncate(fd, size) == -1) {
            perror("ftruncate failed");
            close(fd);
            return -1;
        }

        close(fd);
    }

    return 0;
}

int shm_mem_get(char *shm_name, shm_desc_t *shm_desc_list, uint32_t *shm_desc_num) {
    if (!shm_name || !shm_desc_list || !shm_desc_num || *shm_desc_num == 0) {
        return -1;
    }

    uint32_t max_num = *shm_desc_num;
    uint32_t count = 0;

    for (int i = 0; i < shm_buff && count < max_num; i++) {
        char posix_shm_name[SHM_NAME_MAX_LEN];
        format_shm_name(posix_shm_name, sizeof(posix_shm_name), shm_name, (shm_type_t)i);

        int fd = shm_open(posix_shm_name, O_RDONLY, 0666);
        if (fd == -1) {
            continue; // 不存在则跳过
        }

        struct stat st;
        if (fstat(fd, &st) == -1) {
            perror("fstat failed");
            close(fd);
            return -1;
        }
        close(fd);

        shm_desc_list[count].type = (shm_type_t)i;
        shm_desc_list[count].size = st.st_size;

        char path_buf[PATH_MAX_LEN];
        format_dev_path(path_buf, sizeof(path_buf), shm_name, (shm_type_t)i);
        snprintf(shm_desc_list[count].dev_path, PATH_MAX_LEN, "%s", path_buf);

        count++;
    }

    *shm_desc_num = count;
    return 0;
}

int shm_mem_free(char *shm_name) {
    if (!shm_name) {
        return -1;
    }

    int ret = 0;
    for (int i = 0; i < shm_buff; i++) {
        char posix_shm_name[SHM_NAME_MAX_LEN];
        format_shm_name(posix_shm_name, sizeof(posix_shm_name), shm_name, (shm_type_t)i);

        if (shm_unlink(posix_shm_name) == -1) {
            if (errno != ENOENT) {
                ret = -1;
            }
        }
    }

    return ret;
}

int shm_mem_mmap(shm_desc_t *desc, int pin) {
    if (!desc || desc->size == 0 || strlen(desc->dev_path) == 0) {
        return -1;
    }

    int fd = open(desc->dev_path, O_RDWR);
    if (fd == -1) {
        perror("C mmap open failed");
        return -1;
    }

    size_t size = page_align_up(desc->size);

    void *addr = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);

    if (addr == MAP_FAILED) {
        perror("C mmap failed");
        return -1;
    }

    void *dev_ptr = NULL;
    if (pin) {
        if (mlock(addr, size) != 0) {
            perror("C mlock (Pin) failed");
            munmap(addr, size);
            return -1;
        }
        aclError ret = aclrtHostRegister(addr, size, 0, &dev_ptr);
        if (ret != ACL_SUCCESS) {
            fprintf(stderr, "aclrtHostRegister failed, error code: %d\n", ret);
            munlock(addr, size);
            munmap(addr, size);
            return -1;
        }
    }

    desc->ptr = addr;
    desc->dev_ptr = dev_ptr;
    return 0;
}

int shm_mem_unmap(shm_desc_t *desc) {
    if (!desc || !desc->ptr || desc->size == 0) {
        return -1;
    }

    size_t size = page_align_up(desc->size);

    aclError ret = aclrtHostUnregister(desc->ptr);
    if (ret != ACL_SUCCESS) {
        fprintf(stderr, "aclrtHostUnregister failed, error code: %d\n", ret);
    }

    munlock(desc->ptr, size);

    if (munmap(desc->ptr, size) != 0) {
        perror("C munmap failed");
        return -1;
    }

    desc->ptr = NULL;
    desc->dev_ptr = NULL;
    return 0;
}

// ==========================================
// 测试验证
// ==========================================
int main() {
    // 1. 初始化 Ascend NPU 运行环境
    printf("--- 0. Initializing Ascend ACL ---\n");
    aclError acl_ret = aclInit(NULL);
    if (acl_ret != ACL_SUCCESS) {
        fprintf(stderr, "aclInit failed, error code: %d\n", acl_ret);
        return -1;
    }

    int32_t device_id = 0;
    acl_ret = aclrtSetDevice(device_id);
    if (acl_ret != ACL_SUCCESS) {
        fprintf(stderr, "aclrtSetDevice failed, error code: %d\n", acl_ret);
        aclFinalize();
        return -1;
    }
    printf("Ascend ACL initialized successfully on device %d.\n", device_id);

    // 2. 分配多进程共享内存
    char *my_shm = "shm_test";
    size_t test_size = 1024 * 1024; // 测试 1MB 数据
    size_t sizes[shm_buff] = {
        [shm_near] = test_size,
        [shm_far]  = 0 // 暂时不测试 far 区域
    };

    printf("\n--- 1. Allocating Shared Memory ---\n");
    if (shm_mem_alloc(my_shm, sizes) != 0) {
        fprintf(stderr, "shm_mem_alloc failed\n");
        goto CLEANUP_ACL;
    }
    printf("Allocated shared memory successfully.\n");

    // 3. 获取共享内存描述符并建立 mmap 映射（含硬件注册）
    printf("\n--- 2. Getting & Mapping Shared Memory ---\n");
    shm_desc_t results[shm_buff];
    uint32_t count = shm_buff;

    if (shm_mem_get(my_shm, results, &count) != 0 || count == 0) {
        fprintf(stderr, "shm_mem_get failed or no shm found\n");
        goto CLEANUP_SHM;
    }

    // 对检测到的 near 区域执行 mmap 并进行 Pinned 注册
    shm_desc_t *near_desc = &results[0];
    printf("Mapping Path: %s, Size: %lu\n", near_desc->dev_path, near_desc->size);
    if (shm_mem_mmap(near_desc, 1) != 0) {
        fprintf(stderr, "shm_mem_mmap failed\n");
        goto CLEANUP_SHM;
    }
    printf("Mmap and aclrtHostRegister successful. Host Ptr: %p\n", near_desc->ptr);

    // 先将共享内存里的数据清零，方便后续验证
    memset(near_desc->ptr, 0, test_size);

    // 4. 在 NPU Device 侧准备测试数据
    printf("\n--- 3. Preparing Device Data ---\n");
    void *device_ptr = NULL;
    acl_ret = aclrtMalloc(&device_ptr, test_size, ACL_MEM_MALLOC_NORMAL_ONLY);
    if (acl_ret != ACL_SUCCESS) {
        fprintf(stderr, "aclrtMalloc failed, error code: %d\n", acl_ret);
        goto CLEANUP_MMAP;
    }

    // 在 Host 侧临时弄一块普通动态内存，填满特征数据（例如 0x5A），然后打入 Device
    void *tmp_host_buf = malloc(test_size);
    memset(tmp_host_buf, 0x5A, test_size);
    acl_ret = aclrtMemcpy(device_ptr, test_size, tmp_host_buf, test_size, ACL_MEMCPY_HOST_TO_DEVICE);
    free(tmp_host_buf);

    if (acl_ret != ACL_SUCCESS) {
        fprintf(stderr, "Host to Device memcpy failed, error code: %d\n", acl_ret);
        goto CLEANUP_DEV;
    }
    printf("Device data ready (filled with 0x5A).\n");

    // 5. 执行真正的硬件 DMA 拷贝 (Device -> 我们的共享内存)
    printf("\n--- 4. Executing Real DMA (Device -> Registered SHM Host Ptr) ---\n");
    // 这里的目标地址直接传入 near_desc->ptr，因为已经由 aclrtHostRegister 注册过，它将走高效的硬件 DMA
    acl_ret = aclrtMemcpy(near_desc->ptr, test_size, device_ptr, test_size, ACL_MEMCPY_DEVICE_TO_HOST);
    if (acl_ret != ACL_SUCCESS) {
        fprintf(stderr, "DMA Transfer (ACL_MEMCPY_DEVICE_TO_HOST) failed! Error code: %d\n", acl_ret);
    } else {
        printf("DMA Transfer API returned SUCCESS.\n");

        // 6. 验证数据正确性
        uint8_t *check_ptr = (uint8_t *)near_desc->ptr;
        int success = 1;
        for (size_t i = 0; i < test_size; i++) {
            if (check_ptr[i] != 0x5A) {
                fprintf(stderr, "Data mismatch at index %lu: expected 0x5A, got 0x%02X\n", i, check_ptr[i]);
                success = 0;
                break;
            }
        }
        if (success) {
            printf(">>> SUCCESS: Data verification passed! DMA wrote 0x5A into SHM successfully. <<<\n");
        }
    }

    // 7. 资源释放与清理
CLEANUP_DEV:
    aclrtFree(device_ptr);
CLEANUP_MMAP:
    shm_mem_unmap(near_desc);
CLEANUP_SHM:
    printf("\n--- 5. Freeing Shared Memory ---\n");
    shm_mem_free(my_shm);
CLEANUP_ACL:
    printf("\n--- 6. Finalizing Ascend ACL ---\n");
    aclrtResetDevice(device_id);
    aclFinalize();
    printf("Done.\n");

    return 0;
}