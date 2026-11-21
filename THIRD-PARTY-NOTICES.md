# 第三方组件与许可

本仓库**自己的代码**以 MIT 许可发布（见 [LICENSE](LICENSE)）。`engine/models/` 下的两个模型文件由 Google 提供，以 **Apache License 2.0** 分发，**不属于** MIT 覆盖的部分；它们的许可与归属见下。

## MediaPipe Face Landmarker

| 项 | 内容 |
|---|---|
| 文件 | `engine/models/face_landmarker.task`（3,758,596 字节） |
| SHA-256 | `64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff` |
| 上游 | MediaPipe Face Landmarker（FaceLandmarker-V2，float16） |
| 模型卡 | https://huggingface.co/litert-community/MediaPipe-FaceLandmarker |
| 官方下载地址 | https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task |
| 许可证 | Apache License 2.0 |
| 修改 | 无。仓库里的文件就是从上面的官方地址直接下载的原始字节 |

## MediaPipe Selfie Multiclass 256×256

| 项 | 内容 |
|---|---|
| 文件 | `engine/models/selfie_multiclass_256x256.tflite`（16,371,837 字节） |
| SHA-256 | `c6748b1253a99067ef71f7e26ca71096cd449baefa8f101900ea23016507e0e0` |
| 上游 | MediaPipe Image Segmenter — Selfie Multiclass 256×256（float32） |
| 模型卡 | https://huggingface.co/litert-community/MediaPipe-ImageSegmenter-Selfie-Multiclass |
| 官方下载地址 | https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite |
| 许可证 | Apache License 2.0 |
| 修改 | 无。同上，直接下载的原始字节 |

`scripts/setup-engine.sh` 会自动从这两个官方地址下载；仓库里附带一份，是为了让 `npm pack` 的产物直接可用，不需要用户自己去拉。

## 许可证全文

- Apache License 2.0：见 [LICENSES/Apache-2.0.txt](LICENSES/Apache-2.0.txt)（取自 https://www.apache.org/licenses/LICENSE-2.0.txt）
- 本项目自身：见 [LICENSE](LICENSE)（MIT）

## 商标

MediaPipe 和 Google 是 Google LLC 的商标。本项目与 Google 没有隶属关系，也没有获得 Google 的赞助或背书；项目名里出现「MediaPipe」只是为了说明模型文件的来源。

## 如果你要再分发

Apache License 2.0 允许商用、修改和再分发，条件是：

1. 保留这份说明和 [LICENSES/Apache-2.0.txt](LICENSES/Apache-2.0.txt)；
2. 保留模型文件里的版权与归属信息；
3. 如果你改动了模型文件或做了格式转换，要注明「已修改」；
4. 不要用 Google 的商标暗示你的版本得到了官方背书。

本项目把模型文件原样打包（未修改），所以 1、2 两条已经满足，3 不适用。
