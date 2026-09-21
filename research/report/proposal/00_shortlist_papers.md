# Shortlist paper RAG 2024–2025 — đánh giá theo tiêu chí "ý tưởng thú vị + chạy được trên 1×RTX 5090"

Cập nhật: 2026-09-21. Tất cả repo đã clone về `rag_prj/source/`, PDF ở `rag_prj/research/paper/`.

## Ràng buộc phần cứng thực tế của mình

| Hạng mục | Giá trị |
|---|---|
| GPU | 1× RTX 5090, 32 GB VRAM, **compute capability sm_120 (Blackwell)** |
| Driver / CUDA | 580.126.20 / CUDA 13.0 |
| CPU / RAM | 32 core / 123 GB |
| Disk trống | 878 GB trên `/data` |
| Env sẵn có | `/data/anhvv/envs/mri5090` (torch 2.11.0+cu128, có sm_120), `/data/anhvv/envs/hivqa` (torch 2.7.0+cu128, có sm_120) |
| Chưa có | vLLM, FAISS, HF model cache (trống) |

**Điểm chặn quan trọng nhất:** sm_120 chỉ được hỗ trợ từ **PyTorch ≥ 2.7 (cu128)** trở lên. Mọi repo pin `torch < 2.7`,
`vllm ≤ 0.6.x`, hay `flash-attn` cũ đều **không build/chạy được** trên card này nếu không gỡ pin. Đây là tiêu chí lọc
cứng, quan trọng hơn cả VRAM.

## Bảng so sánh 9 ứng viên

| # | Paper | Venue / arXiv | Ý tưởng cốt lõi | Pin phụ thuộc trong repo | Chạy được trên 5090? | Verdict |
|---|---|---|---|---|---|---|
| 1 | **HippoRAG 2** — *From RAG to Memory* | ICML 2025 / [2502.14802](https://arxiv.org/abs/2502.14802) | KG mở + Personalized PageRank mô phỏng hippocampus; query→triple linking, triple filtering ("recognition memory") | `torch==2.5.1`, `vllm==0.6.6.post1` (optional extra) | **Có** — torch chỉ dùng cho embedder, gỡ pin dễ; LLM gọi qua HTTP OpenAI-compatible nên chạy vLLM ở env riêng | Phương án B — ý tưởng mạnh hơn nhưng nặng và khó hiểu hơn nhiều |
| 2 | **MiniRAG** | arXiv 01/2025 / [2501.06713](https://arxiv.org/abs/2501.06713) | Đồ thị dị thể (chunk + entity) + retrieval theo topology, thiết kế riêng cho SLM | **Không pin torch** | **Có, dễ nhất** — model đích là Phi-3.5-mini / Qwen2.5-3B / GLM-Edge-1.5B | ⭐ **Đã chốt — paper chính** |
| 3 | **Adaptive-RAG** | NAACL 2024 / [2403.14403](https://arxiv.org/abs/2403.14403) | Classifier phân loại độ khó câu hỏi → route sang no-retrieval / 1-step / multi-step | `torch>=1.7,<2.0`, transformers pin theo git SHA | Repo gốc **không**; nhưng FlashRAG đã có sẵn method `adaptive` | Dùng làm baseline, không làm paper chính |
| 4 | **Sufficient Context** | ICLR 2025 / [2411.06037](https://arxiv.org/abs/2411.06037) | Định nghĩa "đủ ngữ cảnh" thay cho "liên quan"; autorater + selective generation để model biết từ chối trả lời | — | Repo **chỉ có README + ảnh PNG, không có code** | ❌ Không đạt yêu cầu "chạy full thực nghiệm"; nhưng là **lăng kính phân tích rất tốt** để ghép vào paper chính |
| 5 | **CRAG** — Corrective RAG | 2024 / [2401.15884](https://arxiv.org/abs/2401.15884) | Evaluator nhẹ chấm chất lượng retrieval → correct / incorrect / ambiguous, bù bằng web search | `torch==2.1.2`, `vllm==0.2.5`, `flash-attn==2.2.2` | **Không** (Blackwell). Còn phụ thuộc Google Search API tính phí | ❌ Loại |
| 6 | **Search-R1** | 2025 / [2503.09516](https://arxiv.org/abs/2503.09516) | RL (PPO/GRPO) dạy LLM tự đan xen reasoning và gọi search engine | `vllm<=0.6.3`, `transformers<4.48`, `flash-attn` | **Không** — pin không hợp Blackwell, và RL cần actor + rollout + ref model đồng thời, 32 GB quá chật | ❌ Rủi ro rất cao, dễ cháy cả kỳ vào infra |
| 7 | **LightRAG** | EMNLP 2025 / [2410.05779](https://arxiv.org/abs/2410.05779) | Graph RAG hai cấp (low/high-level keyword), rẻ hơn GraphRAG | Không pin torch | Có | Dùng làm **baseline đối chứng**, không làm paper chính (xem #9) |
| 8 | **FlashRAG** | WWW 2025 Resource / [2405.13576](https://arxiv.org/abs/2405.13576) | Toolkit: 36 dataset tiền xử lý + 23 thuật toán RAG dưới một khung thống nhất | `torch` không pin | Có | 🔧 **Không phải paper chính** — là *bộ khung chạy baseline + ablation*. Nên cài dù chọn paper nào |
| 9 | **Unbiased Evaluation for GraphRAG** | 2025 / [2506.06331](https://arxiv.org/abs/2506.06331) | Chỉ ra protocol đánh giá của GraphRAG/LightRAG bị lệch: cùng một LightRAG so với chính nó vẫn ra win-rate 90/10 | — | — | 🔍 **Đạn dược cho phần "hạn chế"** — dùng để phản biện bất kỳ paper graph-RAG nào |

Bonus đã tải: [HippoRAG v1 (NeurIPS 2024)](https://arxiv.org/abs/2405.14831), [RAG vs GraphRAG (2025)](https://arxiv.org/abs/2502.11371).

## Quyết định cuối: chọn MiniRAG

Ban đầu xếp HippoRAG 2 đứng đầu vì ý tưởng mạnh hơn. Đã đổi sang **MiniRAG** vì hai lý do:

1. **Độ khó khái niệm.** HippoRAG 2 đòi hiểu Personalized PageRank (damping factor, reset probability vector,
   synonym threshold) cộng với ánh xạ thần kinh học. MiniRAG chỉ có 2 loại node, 2 loại cạnh, cosine similarity
   và đi đường trên đồ thị — phần method gần như toàn diễn giải bằng lời, không có đại số ma trận.
   Với đồ án môn học, hiểu chắc toàn bộ cơ chế quan trọng hơn là chọn paper oách.
2. **Ngân sách GPU.** ~10–15 giờ (MiniRAG) so với ~25–30 giờ (HippoRAG 2), và mỗi lần chạy dưới 1 giờ.

Đổi lại, MiniRAG là paper yếu hơn — nhưng điều đó **có lợi** cho yêu cầu "nhìn từ tốt đến hạn chế" của thầy:
benchmark do chính nhóm tác giả tạo, baseline LightRAG cũng của nhóm đó, metric chấm bằng LLM, và
79% câu hỏi trong LiHua-World là single-hop dù bài toán đặt ra là suy luận đa bước. Có nhiều thứ để nói.

Chi tiết ở [`01_proposal_stage1.md`](./01_proposal_stage1.md).

## Phụ lục: vì sao ban đầu xếp HippoRAG 2 đứng đầu (đã đổi sang MiniRAG)

1. **Ý tưởng có chiều sâu, không phải kỹ thuật vụn.** Ánh xạ KG ↔ hippocampal index, PPR ↔ pattern completion,
   triple filtering ↔ recognition memory. Có câu chuyện để trình bày, không chỉ là "thêm một module".
2. **Dữ liệu tái lập nằm sẵn trong repo.** `source/HippoRAG/reproduce/dataset/` có **đúng** các tập paper dùng:
   MuSiQue (11,656 passage / 1,000 query), 2Wiki (6,119 / 1,000), HotpotQA (9,811 / 1,000). Không phải tải index
   Wikipedia 70 GB như các paper open-domain khác — đây là khác biệt lớn về chi phí.
3. **Chi phí index đã được paper công bố, và nó nằm trong tầm.** Bảng 12: index MuSiQue tốn 9.2M input + 3.0M output
   token. Đối chiếu: LightRAG tốn 68.5M input (7.4×), GraphRAG tốn 115.5M (12.6×). Tức HippoRAG 2 là paper graph-RAG
   *rẻ nhất* trong nhóm — hợp đúng ràng buộc của mình. VRAM lúc QA chỉ 9.9 GB.
4. **Kiến trúc tách rời LLM.** Repo gọi model qua endpoint OpenAI-compatible → chạy vLLM (env `mri5090`) ở process
   riêng, phần còn lại của HippoRAG chỉ cần numpy/igraph. Né hoàn toàn địa ngục pin phụ thuộc.
5. **Có sẵn khoảng trống phản biện, do chính tác giả để lộ.** Error analysis của paper: 18% mẫu **còn 0 triple sau khi
   filter**, và triple filtering là một trong hai nguồn lỗi chính. Cấu hình headline lại là Llama-3.3-70B trên 4×H100 +
   NV-Embed-v2 (7B). Câu hỏi "lợi thế này còn lại bao nhiêu khi hạ compute?" là câu hỏi thật, chưa ai trả lời có hệ thống,
   và nó *sinh ra từ chính ràng buộc phần cứng của mình* thay vì là hạn chế phải xin lỗi.

## Rủi ro đã biết của lựa chọn này

- Không thể tái lập con số headline (cần 70B + NV-Embed-v2 7B). **Phải nói rõ trong proposal** rằng mục tiêu là tái lập
  *xu hướng và thứ hạng*, không phải con số tuyệt đối.
- NV-Embed-v2 (7B, ~16 GB fp16) không chạy song song với vLLM trên cùng 32 GB → phải tách pha index/embed, hoặc
  thay bằng GTE-Qwen2-1.5B-Instruct / BGE-M3. Việc thay embedder chính là một biến ablation, nên không phải mất mát.
- Ước tính thông lượng ở tài liệu proposal là **ước tính chưa đo**, cần benchmark thật ở tuần 1.
