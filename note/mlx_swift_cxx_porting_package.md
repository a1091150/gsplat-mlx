# MLX Swift C++ Porting Package Note

## 目的

如果要把 `gsplat_core` 這類 MLX C++ extension 移植到 Xcode app，建議不要讓
Swift 直接碰 `mlx::core::*` 或 `gsplat_core::*`。比較穩的邊界是：

```text
Swift app
  -> C / Objective-C bridging header
  -> Objective-C++ / C++ wrapper
  -> mlx::core C++ and gsplat_core C++
```

這樣 Swift 只看見 C ABI，例如 `const char *checkMLXEvalStatus(void)` 或
`int gsplat_projection_forward(...)`，真正的 MLX C++ 實作藏在 `.mm` / `.cpp`
裡。

## 已驗證結論

在 `/tmp/FooProduct` 的 macOS app 測試中，以下路徑可行：

1. 修改本機 `submodules/mlx-swift/Package.swift`，公開 `Cmlx` target。
2. Xcode app 使用本機 `mlx-swift` package。
3. App target link `MLX` 與 `MLXCoreCxx`。
4. App 內 Objective-C++ wrapper include `<mlx/mlx.h>`、`<mlx/ops.h>`、
   `<mlx/transforms.h>`。
5. Swift 透過 bridging header 呼叫 C function。
6. 使用 `xcodebuild` 建置 app，Xcode 會把 `default.metallib` 打包到 app
   resource bundle。

實測指令：

```bash
xcodebuild build \
  -project FooProduct.xcodeproj \
  -scheme FooProduct \
  -destination 'platform=macOS' \
  -derivedDataPath /private/tmp/FooProductDerived
```

建置後可看到：

```text
/private/tmp/FooProductDerived/Build/Products/Debug/FooProduct.app/Contents/Resources/mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib
```

實測執行：

```bash
/private/tmp/FooProductDerived/Build/Products/Debug/FooProduct.app/Contents/MacOS/FooProduct --check-mlx
```

輸出：

```text
FooProduct MLX check: mlx::core::eval status=0
```

另外也驗證過「新的 Swift package 引用修改過的 `mlx-swift`」這條路可行。測試
package 放在：

```text
/private/tmp/MlxSwiftCxxPackageProbe
```

測試指令：

```bash
xcodebuild test \
  -scheme MlxSwiftCxxPackageProbe \
  -destination 'platform=macOS' \
  -derivedDataPath /private/tmp/MlxSwiftCxxPackageProbeDerived
```

結果：

```text
** TEST SUCCEEDED **
```

XCTest 內真的透過 C wrapper 呼叫到：

```cpp
auto input = mlx::core::array(3.5f);
auto output = mlx::core::min(input, mlx::core::Device::cpu);
mlx::core::eval(output);
```

並且 Xcode 有把 `default.metallib` 放進 test bundle：

```text
/private/tmp/MlxSwiftCxxPackageProbeDerived/Build/Products/Debug/MlxSwiftCxxBridgeTests.xctest/Contents/Resources/mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib
```

## 為什麼不能只用 swift build

`mlx-swift` README 已說明：SwiftPM command line 不能 build Metal shaders，最終
建置需要走 Xcode / `xcodebuild`。如果只用 `swift build` 或 `swift run`，C++
通常可以編譯，但執行 `mlx::core::eval(...)` 時會找不到 `default.metallib`。

典型錯誤：

```text
Failed to load the default metallib. library not found ...
```

所以用 SwiftPM probe 只能驗證 include/link 邊界，不能代表 app runtime 成功。
要驗證 MLX Metal runtime，請用 Xcode app 或 `xcodebuild`。

## 修改 mlx-swift Package.swift

在 `submodules/mlx-swift/Package.swift` 的 `products` 加上：

```swift
products: [
    .library(name: "Cmlx", targets: ["Cmlx"]),
    .library(name: "MLXCoreCxx", targets: ["Cmlx"]),
    .library(name: "MLX", targets: ["MLX"]),
    .library(name: "MLXRandom", targets: ["MLXRandom"]),
    .library(name: "MLXNN", targets: ["MLXNN"]),
    .library(name: "MLXOptimizers", targets: ["MLXOptimizers"]),
    .library(name: "MLXFFT", targets: ["MLXFFT"]),
    .library(name: "MLXLinalg", targets: ["MLXLinalg"]),
    .library(name: "MLXFast", targets: ["MLXFast"]),
]
```

`MLXCoreCxx` 只是語意更清楚的 alias，實際 target 仍是 `Cmlx`。這個 target
本來就包含：

```text
Source/Cmlx/mlx-c
Source/Cmlx/mlx
Source/Cmlx/include
```

也就是 MLX C API、MLX C++ source、public C headers 都在同一個 SwiftPM target
裡。

## Xcode project 需要怎麼改

### 1. 使用本機 mlx-swift package

在 Xcode 專案中把原本遠端 package：

```text
https://github.com/ml-explore/mlx-swift
```

改成本機 package：

```text
/Users/yangdunfu/Documents/gsplat_core/submodules/mlx-swift
```

如果直接改 `project.pbxproj`，reference 會類似：

```text
XCLocalSwiftPackageReference "mlx-swift" = {
    isa = XCLocalSwiftPackageReference;
    relativePath = ../../Users/yangdunfu/Documents/gsplat_core/submodules/mlx-swift;
};
```

`relativePath` 要依照 `.xcodeproj` 所在位置調整。以 `/tmp/FooProduct` 為例，
從 `/tmp/FooProduct` 到 `/Users/...` 是：

```text
../../Users/yangdunfu/Documents/gsplat_core/submodules/mlx-swift
```

### 2. App target 加 package products

App target 需要 link：

```text
MLX
MLXCoreCxx
```

如果 Swift app 仍然使用 MLX Swift API，就保留 `MLX`。如果只有 ObjC++ wrapper
使用 MLX C++，理論上核心是 `MLXCoreCxx`，但目前保留 `MLX` 也比較貼近
mlx-swift app 的正常建置路徑。

### 3. App target 加 C++ header search path

`MLXCoreCxx` product 可讓外部 target link 到 `Cmlx`，但 SwiftPM 不會自動把
`Cmlx` target 內部的 C++ header root 傳遞給外部 C++ target。

所以 app target 的 `HEADER_SEARCH_PATHS` 需要加：

```text
/Users/yangdunfu/Documents/gsplat_core/submodules/mlx-swift/Source/Cmlx/mlx
```

這樣 Objective-C++ wrapper 才能：

```cpp
#include <mlx/mlx.h>
#include <mlx/ops.h>
#include <mlx/transforms.h>
```

C API header `<mlx/c/array.h>` 通常可以從 package public headers 找到，但 C++
header root 目前需要手動加。

### 4. Swift bridging header

App target 需要設定：

```text
SWIFT_OBJC_BRIDGING_HEADER = FooProduct/CProduct/FooProduct-Bridging-Header.h
```

bridging header 只 include C-safe header：

```c
// FooProduct/CProduct/FooProduct-Bridging-Header.h
#include "CheckMLX.h"
```

不要在 bridging header 放 C++ header。Swift compiler 不應該看見
`mlx::core::*`。

## 新 Swift package 引用 mlx-swift 的寫法

如果不是直接在 app target 放 `.mm`，而是新增一個 Swift package 專門做
`gsplat_core` / MLX C++ wrapper，也可以。架構一樣保持：

```text
App Swift target
  -> MyGsplatWrapper Swift/C module
  -> MyGsplatWrapper C++ target
  -> MLXCoreCxx product from local mlx-swift
```

重點是：外部 C++ target 依賴 `MLXCoreCxx` 後，仍然要自己補 MLX C++ header
root。`Cmlx` target 內部的 `.headerSearchPath("mlx")` 不會自動傳遞給外部
target。

最小 `Package.swift`：

```swift
// swift-tools-version: 6.0

import PackageDescription

let mlxSwiftPath = "/Users/yangdunfu/Documents/gsplat_core/submodules/mlx-swift"

let package = Package(
    name: "MlxSwiftCxxPackageProbe",
    platforms: [
        .macOS(.v14)
    ],
    products: [
        .library(name: "MlxSwiftCxxBridge", targets: ["MlxSwiftCxxBridge"])
    ],
    dependencies: [
        .package(path: mlxSwiftPath)
    ],
    targets: [
        .target(
            name: "MlxSwiftCxxBridge",
            dependencies: [
                .product(name: "MLXCoreCxx", package: "mlx-swift")
            ],
            publicHeadersPath: "include",
            cxxSettings: [
                .unsafeFlags([
                    "-I", "\(mlxSwiftPath)/Source/Cmlx/mlx"
                ])
            ]
        ),
        .testTarget(
            name: "MlxSwiftCxxBridgeTests",
            dependencies: ["MlxSwiftCxxBridge"]
        )
    ],
    cxxLanguageStandard: .cxx17
)
```

public C header：

```c
// Sources/MlxSwiftCxxBridge/include/MlxSwiftCxxBridge.h
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

const char *mlx_swift_cxx_eval_status(void);
int mlx_swift_cxx_eval_smoke(void);

#ifdef __cplusplus
}
#endif
```

C++ implementation：

```cpp
// Sources/MlxSwiftCxxBridge/MlxSwiftCxxBridge.cpp
#include "MlxSwiftCxxBridge.h"

#include <exception>
#include <string>

#include <mlx/mlx.h>
#include <mlx/ops.h>
#include <mlx/transforms.h>

const char *mlx_swift_cxx_eval_status(void) {
  static std::string status;

  try {
    auto input = mlx::core::array(3.5f);
    auto output = mlx::core::min(input, mlx::core::Device::cpu);
    mlx::core::eval(output);
    status = "mlx::core::eval status=0";
  } catch (const std::exception &error) {
    status = std::string("mlx::core::eval failed: ") + error.what();
  } catch (...) {
    status = "mlx::core::eval failed: unknown exception";
  }

  return status.c_str();
}

int mlx_swift_cxx_eval_smoke(void) {
  std::string status = mlx_swift_cxx_eval_status();
  return status.find("status=0") != std::string::npos ? 0 : 1;
}
```

Swift test 只 import wrapper，不碰 C++：

```swift
import MlxSwiftCxxBridge
import XCTest

final class MlxSwiftCxxBridgeTests: XCTestCase {
    func testMLXCoreEvalThroughCWrapper() {
        let status = String(cString: mlx_swift_cxx_eval_status())
        XCTAssertEqual(mlx_swift_cxx_eval_smoke(), 0, status)
        XCTAssertEqual(status, "mlx::core::eval status=0")
    }
}
```

## 最小範例程式碼

### C header

```c
// FooProduct/CProduct/CheckMLX.h
#ifndef CheckMLX_h
#define CheckMLX_h

#ifdef __cplusplus
extern "C" {
#endif

const char *checkMLXEvalStatus(void);

#ifdef __cplusplus
}
#endif

#endif
```

### Objective-C++ wrapper

檔案副檔名要是 `.mm`，不是 `.c` 或 `.m`。

```cpp
// FooProduct/CProduct/CheckMLX.mm
#include "CheckMLX.h"

#include <exception>
#include <string>

#include <mlx/mlx.h>
#include <mlx/ops.h>
#include <mlx/transforms.h>

const char *checkMLXEvalStatus(void) {
  static std::string status;

  try {
    auto input = mlx::core::array(3.5f);
    auto result = mlx::core::min(input, mlx::core::Device::cpu);
    mlx::core::eval(result);
    status = "mlx::core::eval status=0";
  } catch (const std::exception &e) {
    status = std::string("mlx::core::eval failed: ") + e.what();
  } catch (...) {
    status = "mlx::core::eval failed: unknown exception";
  }

  return status.c_str();
}
```

### Swift call site

```swift
import SwiftUI

struct ContentView: View {
    @State private var mlxStatus = "Not run"

    var body: some View {
        VStack {
            Text(mlxStatus)
                .font(.system(.body, design: .monospaced))

            Button("Run MLX eval") {
                mlxStatus = String(cString: checkMLXEvalStatus())
            }
        }
        .padding()
        .task {
            mlxStatus = String(cString: checkMLXEvalStatus())
        }
    }
}
```

### Command-line smoke mode for app bundle

可以在 `App.init()` 加一個測試模式，方便 CI 或 terminal 驗證 app bundle 內的
`default.metallib` 是否可被 MLX 找到：

```swift
import SwiftUI

@main
struct FooProductApp: App {
    init() {
        let status = String(cString: checkMLXEvalStatus())
        print("FooProduct MLX check: \(status)")
        if CommandLine.arguments.contains("--check-mlx") {
            exit(status.contains("status=0") ? 0 : 1)
        }
    }

    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}
```

## 對 gsplat_core 的 wrapper 形狀

`gsplat_core` 不應該直接 expose C++ type 給 Swift。建議新增 C ABI，例如：

```c
// GsplatCoreBridge.h
#ifdef __cplusplus
extern "C" {
#endif

int gsplat_projection_forward(
    mlx_array *out_means2d,
    mlx_array *out_depths,
    mlx_array *out_conics,
    mlx_array *out_compensations,
    mlx_array *out_radii,
    const mlx_array means,
    const mlx_array covars,
    const mlx_array quats,
    const mlx_array scales,
    const mlx_array opacities,
    const mlx_array viewmats,
    const mlx_array Ks,
    int image_width,
    int image_height);

#ifdef __cplusplus
}
#endif
```

Objective-C++ / C++ implementation 內部才轉回 MLX C++：

```cpp
#include <mlx/c/array.h>
#include <mlx/c/private/mlx.h>
#include <mlx/mlx.h>

#include "gsplat_projection.h"

extern "C" int gsplat_projection_forward(
    mlx_array *out_means2d,
    mlx_array *out_depths,
    mlx_array *out_conics,
    mlx_array *out_compensations,
    mlx_array *out_radii,
    const mlx_array means,
    const mlx_array covars,
    const mlx_array quats,
    const mlx_array scales,
    const mlx_array opacities,
    const mlx_array viewmats,
    const mlx_array Ks,
    int image_width,
    int image_height) {
  try {
    gsplat_core::ProjectionEWA3DGSFusedInput input = {
        .means = mlx_array_get_(means),
        .covars = mlx_array_get_(covars),
        .quats = mlx_array_get_(quats),
        .scales = mlx_array_get_(scales),
        .opacities = mlx_array_get_(opacities),
        .viewmats = mlx_array_get_(viewmats),
        .Ks = mlx_array_get_(Ks),
        .viewspace_points = mlx::core::zeros({0}, mlx::core::float32),
        .s = mlx::core::Device::gpu,
        .params = {
            .image_width = image_width,
            .image_height = image_height,
            .eps2d = 0.3f,
            .near_plane = 0.01f,
            .far_plane = 1.0e10f,
            .radius_clip = 0.0f,
            .calc_compensations = false,
            .camera_model = 0,
            .use_covars = covars.ctx != nullptr,
            .use_opacities = opacities.ctx != nullptr,
        },
    };

    auto outputs = gsplat_core::gsplat_projection_ewa_3dgs_fused(input);
    mlx_array_set_(*out_means2d, outputs[gsplat_core::kMeans2D]);
    mlx_array_set_(*out_depths, outputs[gsplat_core::kDepths]);
    mlx_array_set_(*out_conics, outputs[gsplat_core::kConics]);
    mlx_array_set_(*out_compensations, outputs[gsplat_core::kCompensations]);
    mlx_array_set_(*out_radii, outputs[gsplat_core::kRadii]);
    return 0;
  } catch (const std::exception &e) {
    mlx_error(e.what());
    return 1;
  }
}
```

上面只是 wrapper 形狀示意。實際移植時要根據 `gsplat_core` 的可選輸入、
stream/device、backward/vjp 行為、Metal kernel resource path 再補完整。

## Binary / package 選項

### 選項 A：App 直接依賴 local mlx-swift + wrapper source

最適合目前 bring-up：

```text
FooProduct.app
  -> local mlx-swift package: MLX, MLXCoreCxx
  -> app target contains CheckMLX.mm / GsplatCoreBridge.mm
  -> Swift calls C ABI
```

優點：

- Xcode 自動產生並打包 `mlx-swift_Cmlx.bundle/default.metallib`。
- 不會產生第二份 MLX runtime。
- Debug 最直覺。

缺點：

- App project 需要知道 local `mlx-swift` 路徑。
- C++ header search path 目前要手動加。

### 選項 B：外部 Swift package 放 wrapper source

可行，且已用 `xcodebuild test` 驗證。形狀是：

```text
MlxSwiftCxxPackageProbe
  -> MlxSwiftCxxBridge target
      -> public C header
      -> C++ source includes <mlx/mlx.h>
      -> depends on MLXCoreCxx
      -> adds -I .../Source/Cmlx/mlx
  -> Swift XCTest imports only MlxSwiftCxxBridge
```

優點：

- app project 可以比較乾淨，不一定要把所有 `.mm/.cpp` 都放在 app target。
- `gsplat_core` wrapper 可以獨立測試。
- Swift 仍然只看到 C ABI。

缺點：

- package 內碰 MLX C++ 的 target 還是需要 `.unsafeFlags(["-I", ...])`。
- 如果只用 `swift build` / `swift test`，可能又回到 `default.metallib` runtime
  問題；要測 `mlx::core::eval`，仍建議用 `xcodebuild test`。

### 選項 C：gsplat_core 做成 binary target / xcframework

可行，但要避免把 MLX C++ runtime 打包兩份。理想形狀：

```text
GSPlatCore.xcframework
  -> expose C ABI
  -> does not embed another independent MLX runtime

App
  -> MLX / MLXCoreCxx from local mlx-swift
  -> GSPlatCore.xcframework
```

風險：

- 如果 `GSPlatCore.xcframework` 自己帶一份 MLX，而 app 又透過 `mlx-swift`
  帶一份 MLX，就可能出現 duplicate symbols、ODR 問題、MLX global state
  不一致，甚至 `mlx_array.ctx` 指向不同 runtime 的物件。

## 常見問題

### 找不到 mlx/mlx.h

原因：`MLXCoreCxx` product 可傳遞 link dependency，但不會自動傳遞 MLX C++
header root。

處理：App / wrapper target 加：

```text
HEADER_SEARCH_PATHS += /Users/yangdunfu/Documents/gsplat_core/submodules/mlx-swift/Source/Cmlx/mlx
```

### 找不到 default.metallib

原因：使用 `swift build` / `swift run`，沒有走 Xcode Metal shader/resource
建置流程。

處理：使用 Xcode 或 `xcodebuild` 建置 app。確認 app bundle 內有：

```text
Contents/Resources/mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib
```

### Swift 看見 C++ header 導致編譯問題

不要在 bridging header include `<mlx/mlx.h>` 或任何 `gsplat_core` C++ header。
bridging header 只 include C-safe wrapper header。

### 在 sandbox 或 terminal 執行時 Metal device access 失敗

先用 Xcode/Finder 或非 sandbox 執行 app 驗證。先前在 sandbox 裡直接跑 binary
時，曾遇到 Metal device list 為空或無輸出的失敗；非 sandbox 執行同一個
Xcode-built app 則成功。

## 建議下一步

1. 保留 `MLXCoreCxx` product patch，作為外部 C++ wrapper 的 MLX runtime
   dependency。
2. 在 app target 中建立 `GsplatCoreBridge.h/.mm`，先接一個最小 forward op。
3. Swift 端只保存 `MLXArray` 或 wrapper-level object，不直接 expose C++。
4. 每次移植一個 op 後，用 `xcodebuild` 建 app，並用 `--check-mlx` 類似模式
   做 runtime smoke test。
5. 等 source wrapper 路徑穩定後，再評估是否抽成 `GSPlatCore.xcframework`。
