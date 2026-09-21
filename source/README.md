# Repo đã clone — bản đồ repo ↔ paper

Clone shallow (`--depth 1`, `GIT_LFS_SKIP_SMUDGE=1`) ngày 2026-09-21. PDF tương ứng ở `../research/paper/`.

| Thư mục | Paper | PDF | Vai trò trong đề tài |
|---|---|---|---|
| `HippoRAG/` | HippoRAG 2, ICML 2025 ([2502.14802](https://arxiv.org/abs/2502.14802)) | `HippoRAG2_FromRAGtoMemory_ICML2025.pdf` | **Paper chính.** Repo cũng chứa HippoRAG v1 để so sánh |
| `MiniRAG/` | MiniRAG ([2501.06713](https://arxiv.org/abs/2501.06713)) | `MiniRAG_SLM_2025.pdf` | **Phương án B** |
| `FlashRAG/` | FlashRAG, WWW 2025 ([2405.13576](https://arxiv.org/abs/2405.13576)) | `FlashRAG_WWW2025.pdf` | Khung chạy baseline + ablation |
| `LightRAG/` | LightRAG, EMNLP 2025 ([2410.05779](https://arxiv.org/abs/2410.05779)) | `LightRAG_EMNLP2025.pdf` | Baseline đối chứng chi phí |
| `Adaptive-RAG/` | Adaptive-RAG, NAACL 2024 ([2403.14403](https://arxiv.org/abs/2403.14403)) | `AdaptiveRAG_NAACL2024.pdf` | Baseline (chạy qua FlashRAG, repo gốc pin torch<2.0) |
| `CRAG/` | Corrective RAG ([2401.15884](https://arxiv.org/abs/2401.15884)) | `CRAG_CorrectiveRAG_2024.pdf` | Tham khảo — pin không hợp Blackwell |
| `Search-R1/` | Search-R1 ([2503.09516](https://arxiv.org/abs/2503.09516)) | `SearchR1_2025.pdf` | Tham khảo — RL, quá nặng cho 1 GPU |
| `sufficientcontext/` | Sufficient Context, ICLR 2025 ([2411.06037](https://arxiv.org/abs/2411.06037)) | `SufficientContext_ICLR2025.pdf` | **Chỉ có README + ảnh, không có code.** Dùng làm lăng kính phân tích |
| `GraphRAG-Benchmark/` | GraphRAG-Bench ([2506.02404](https://arxiv.org/abs/2506.02404)) | — | Đánh giá graph-RAG |

PDF thêm, không có repo: `HippoRAG1_NeurIPS2024.pdf`, `UnbiasedEval_GraphRAG_2025.pdf`, `RAGvsGraphRAG_2025.pdf`.

## Ghi chú môi trường (RTX 5090 = sm_120)

sm_120 cần **PyTorch ≥ 2.7 + cu128**. Env sẵn có:

- `/data/anhvv/envs/mri5090` — torch 2.11.0+cu128, arch list có `sm_120` ✅
- `/data/anhvv/envs/hivqa` — torch 2.7.0+cu128, có `sm_120` và `compute_120` ✅

Chưa cài: vLLM, FAISS, sentence-transformers. HF cache đang trống (cần tải model, còn 878 GB).

Pin phụ thuộc cần gỡ:

- `HippoRAG/`: `torch==2.5.1`, `transformers==4.45.2` → gỡ pin torch (chỉ dùng cho embedder); extra `vllm==0.6.6.post1`
  → **không cài**, thay bằng vLLM bản mới chạy ở env riêng, gọi qua endpoint OpenAI-compatible.
- `CRAG/`: `torch==2.1.2`, `vllm==0.2.5`, `flash-attn==2.2.2` → không cứu được nếu không viết lại nhiều.
- `Search-R1/`: `vllm<=0.6.3`, `transformers<4.48` → tương tự.
- `Adaptive-RAG/`: `torch>=1.7,<2.0` → dùng bản cài đặt trong FlashRAG thay thế.

## Dữ liệu đã có sẵn trong repo (không cần tải thêm)

`HippoRAG/reproduce/dataset/` — đúng các tập paper dùng:

| Dataset | Passage | Query |
|---|---|---|
| MuSiQue | 11,656 | 1,000 |
| HotpotQA | 9,811 | 1,000 |
| 2WikiMultihopQA | 6,119 | 1,000 |

`MiniRAG/dataset/LiHua-World/` — dataset LiHua-World đóng gói sẵn (`LiHuaWorld.zip` + `query_set.csv`).
