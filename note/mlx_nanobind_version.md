# MLX and nanobind version note

## Context

`gsplat_core` uses nanobind to expose C++ functions that accept and return
`mlx::core::array`. The extension is built against the nanobind package from the
active Python environment.

The Python `mlx` package also depends on a compatible nanobind ABI/caster setup.
If the installed nanobind version does not match the version expected by the
installed `mlx` package, nanobind may fail to convert `mlx.core.array` even when
the error message appears to show the same type on both sides.

## Symptom

Example failure:

```text
TypeError: dummy_array_add(): incompatible function arguments.
The following argument types are supported:
    1. dummy_array_add(a: mlx::core::array, b: mlx::core::array) -> mlx::core::array

Invoked with types: mlx.core.array, mlx.core.array
```

This can look confusing because the supported C++ type and Python argument type
both refer to MLX arrays. In practice, the bridge can still fail when the MLX
package and nanobind package are not version-compatible.

## Current conclusion

The root cause observed during `gsplat_core` bring-up was an MLX package version
that required a specific nanobind version. After aligning nanobind with the MLX
package requirement, `mlx.core.array` argument conversion worked.

## Prompt:
5/26/2026, 12:06:23 PM

python mlx ，那你檢查 mlx github repo，這裡有標註 tag， https://github.com/ml-explore/mlx/blob/main/CMakeLists.txt#L360，我要你檢查最近幾個 mlx tag 對應的 nanobind 版本

## Response:

| MLX tag | nanobind 來源 | 對應 nanobind 版本 |
|---|---|---|
| `main` | `CMakeLists.txt` 的 `FetchContent_Declare(nanobind ...)` | `v2.12.0` |
| `v0.31.2` | `CMakeLists.txt` 的 `FetchContent_Declare(nanobind ...)` | `v2.12.0` |
| `v0.31.1` | `CMakeLists.txt` 的 `FetchContent_Declare(nanobind ...)` | `v2.10.2` |
| `v0.31.0` | `CMakeLists.txt` 的 `FetchContent_Declare(nanobind ...)` | `v2.10.2` |
| `v0.30.6` | `CMakeLists.txt` 的 `FetchContent_Declare(nanobind ...)` | `v2.10.2` |
| `v0.30.5` | `CMakeLists.txt` 的 `FetchContent_Declare(nanobind ...)` | `v2.10.2` |
| `v0.30.4` | `CMakeLists.txt` 的 `FetchContent_Declare(nanobind ...)` | `v2.10.2` |
| `v0.30.3` | `CMakeLists.txt` 的 `FetchContent_Declare(nanobind ...)` | `v2.10.2` |
| `v0.30.1` | `pyproject.toml` pin + CMake `find_package(nanobind)` | `2.10.2` |
| `v0.30.0` | `pyproject.toml` pin + CMake `find_package(nanobind)` | `2.4.0` |
| `v0.29.4` | `pyproject.toml` pin + CMake `find_package(nanobind)` | `2.4.0` |

MLX releases 頁面目前顯示 `v0.31.2` 是 latest，接著有 `v0.31.1`、`v0.31.0`、`v0.30.6`、`v0.30.5`、`v0.30.4`、`v0.30.3`、`v0.30.1`、`v0.30.0`、`v0.29.4` 等 tags；我沒有看到 `v0.30.2` release，直接查 raw tag 也回 404。([GitHub](https://github.com/ml-explore/mlx/releases))

重點變更是：**從 `v0.30.3` 開始，MLX 的 `CMakeLists.txt` 直接用 `FetchContent_Declare` 下載指定 nanobind tag**。`v0.30.3` 到 `v0.31.1` 都是 `GIT_TAG v2.10.2`；`v0.31.2` 和目前 `main` 則升到 `GIT_TAG v2.12.0`。([GitHub](https://raw.githubusercontent.com/ml-explore/mlx/v0.30.3/CMakeLists.txt))

比較早的 `v0.30.1`、`v0.30.0`、`v0.29.4` 不是在 CMake 裡 pin `GIT_TAG`，而是 CMake 透過 `python -m nanobind --cmake_dir` / `find_package(nanobind CONFIG REQUIRED)` 找 build env 裡的 nanobind；實際版本由 `pyproject.toml` 的 build-system pin 決定：`v0.30.1` 是 `nanobind==2.10.2`，`v0.30.0` 和 `v0.29.4` 是 `nanobind==2.4.0`。([GitHub](https://raw.githubusercontent.com/ml-explore/mlx/v0.30.1/CMakeLists.txt))

對你現在的 extension 來說，如果你安裝的是 **MLX `v0.31.2`**，那 MLX 自己 build Python binding 時用的是 **nanobind `v2.12.0`**。你的 `gsplat_core` extension 最好也確認 CMake 找到的 nanobind 是同一個 major/minor，尤其你現在碰到 `mlx::core::array` caster / domain 問題時，版本不一致會讓問題更難判斷。

---

**Sources:**

- [Releases · ml-explore/mlx · GitHub](https://github.com/ml-explore/mlx/releases)
- [raw.githubusercontent.com](https://raw.githubusercontent.com/ml-explore/mlx/v0.30.3/CMakeLists.txt)
