# Proposal — Stage 1

**Paper chọn:** *From RAG to Memory: Non-Parametric Continual Learning for Large Language Models* (HippoRAG 2)
Gutiérrez et al., **ICML 2025** — [arXiv:2502.14802](https://arxiv.org/abs/2502.14802) — code: [OSU-NLP-Group/HippoRAG](https://github.com/OSU-NLP-Group/HippoRAG)

**Tiêu đề đề tài đề xuất:**
> *Bộ nhớ đồ thị có sống sót khi thu nhỏ compute không? Tái lập và mổ xẻ HippoRAG 2 dưới ngân sách một GPU*

---

## 1. Paper nói gì

RAG tiêu chuẩn lấy từng đoạn văn bản độc lập, nên hỏng ở ba việc mà trí nhớ dài hạn của người làm tốt:
**liên kết** (nối các sự kiện rời rạc qua nhiều bước), **hiểu tổng thể** (sense-making trên văn bản dài),
và **học liên tục** (tích hợp kiến thức mới mà không phải huấn luyện lại).

HippoRAG 2 xử lý bằng cách coi corpus như một mạng nhớ thay vì một đống vector:

1. **Offline** — chạy OpenIE bằng LLM để rút triple `(chủ thể, quan hệ, đối tượng)` từ mỗi passage, dựng
   một KG mở trong đó **cả node cụm từ lẫn node passage** cùng tồn tại (khác HippoRAG 1 chỉ có node cụm từ).
2. **Online, query→triple linking** — thay vì NER câu hỏi rồi khớp vào node (cách của v1), nó embed *cả câu hỏi*
   và khớp vào *triple*. Ablation của paper: **+12.5% Recall@5** so với NER-to-node.
3. **Triple filtering ("recognition memory")** — LLM lọc lại tập triple ứng viên trước khi lan truyền.
4. **Personalized PageRank** trên đồ thị, với trọng số reset của node passage nhân hệ số 0.05 để cân bằng hai loại node.
   PPR đóng vai trò *pattern completion*: từ vài điểm neo, lan ra toàn bộ vùng ký ức liên quan.

**Kết quả chính:** vượt dense retriever mạnh nhất (NV-Embed-v2 7B) **+5.0% Recall@5 trên MuSiQue** và **+13.9% trên
2Wiki**; là phương pháp graph-RAG duy nhất vượt được baseline dense một cách đáng kể mà vẫn rẻ hơn GraphRAG/LightRAG
nhiều lần. Và nó giữ được lợi thế khi corpus phình dần (thí nghiệm continual learning, chia dataset thành 4 phần).

**Ánh xạ thần kinh học** là thứ làm paper thú vị chứ không chỉ là một pipeline: KG ↔ hippocampal index,
embedding model ↔ neocortex, PPR ↔ pattern completion, bước lọc triple ↔ recognition memory.

## 2. Vì sao chọn paper này (và vì sao nó chạy được trên máy của em)

Chi tiết so sánh 9 ứng viên ở [`00_shortlist_papers.md`](./00_shortlist_papers.md). Ba lý do quyết định:

- **Dữ liệu tái lập đã nằm sẵn trong repo.** `reproduce/dataset/` chứa đúng các tập paper dùng: MuSiQue
  (11,656 passage / 1,000 query), 2WikiMultihopQA (6,119 / 1,000), HotpotQA (9,811 / 1,000). Không cần tải và
  index Wikipedia dump 70 GB như các paper open-domain khác — đây là khác biệt sống còn về chi phí.
- **Paper tự công bố chi phí, và chi phí đó vừa túi.** Bảng 12: index MuSiQue = 9.2M input + 3.0M output token,
  VRAM lúc QA 9.9 GB. Cùng bảng đó: LightRAG 68.5M input (7.4×), GraphRAG 115.5M (12.6×). HippoRAG 2 là phương pháp
  graph-RAG **rẻ nhất** trong nhóm.
- **Kiến trúc tách rời LLM.** Repo gọi model qua endpoint OpenAI-compatible, nên có thể chạy vLLM trong env riêng
  (đã có `torch 2.11+cu128`, hỗ trợ sm_120 của RTX 5090) và HippoRAG chỉ cần numpy/igraph. Né được vấn đề pin phụ
  thuộc cũ vốn làm hỏng phần lớn repo RAG 2024 trên card Blackwell.

**Giới hạn phải nói thẳng ngay từ đầu:** cấu hình headline của paper là **Llama-3.3-70B trên 4×H100 + NV-Embed-v2 (7B)**.
Em **không** đặt mục tiêu tái lập con số tuyệt đối đó. Và chính khoảng cách ấy là đề tài.

## 3. Câu hỏi nghiên cứu

Trục xuyên suốt: **HippoRAG 2 hoạt động nhờ dùng LLM rất mạnh làm bộ máy trích xuất và lọc tri thức. Chuyện gì xảy ra
khi LLM đó không mạnh nữa?** Đây không phải câu hỏi bịa ra cho vừa phần cứng — nó nằm đúng chỗ paper mỏng nhất.

**RQ1 — Tái lập.** Với backbone hạ từ 70B xuống 8B, có tái lập được *thứ hạng* giữa HippoRAG 2, HippoRAG 1, và dense
retriever trên MuSiQue / 2Wiki / HotpotQA không? Chênh lệch tuyệt đối là bao nhiêu?

**RQ2 — Thu nhỏ compute (đóng góp chính).** Quét hai chiều độc lập:
- **LLM indexer**: Llama-3.1-8B → Qwen2.5-3B → Qwen2.5-1.5B
- **Embedder**: GTE-Qwen2-1.5B → BGE-M3 → Contriever

Lợi thế của graph so với dense biến mất ở đâu? **Giả thuyết:** lợi thế co lại nhanh hơn theo chiều LLM hơn là chiều
embedder, vì chất lượng KG phụ thuộc trực tiếp vào năng lực OpenIE. Nếu đúng, kết luận thực dụng là: *với ngân sách
nhỏ, tiền nên đổ vào embedder chứ không phải vào đồ thị* — một kết luận trái với thông điệp của paper, và kiểm chứng được.

**RQ3 — Ablation thành phần dưới SLM.** Paper đã ablate linking / graph construction / filtering **với 70B**. Thứ hạng
đó có đổi khi dùng SLM không? Nhắm riêng vào **triple filtering**: error analysis của chính paper cho biết **18% mẫu
còn 0 triple sau khi lọc**, và filtering là một trong hai nguồn lỗi chính. Giả thuyết: với model nhỏ, bước lọc **gây hại
nhiều hơn lợi**, vì model nhỏ lọc quá tay.

**RQ4 — Chi phí công bằng.** Đo lại thật: token, wall-clock, VRAM. Rồi đặt câu hỏi kiểu
[Unbiased Evaluation for GraphRAG (2506.06331)](https://arxiv.org/abs/2506.06331): *nếu cho baseline dense đúng
lượng compute mà HippoRAG 2 tiêu để dựng đồ thị* (dùng để chạy reranker mạnh hơn, hoặc top-k lớn hơn), baseline có đuổi
kịp không? Báo cáo **F1 trên mỗi GPU-hour**, không chỉ F1.

## 4. Cải tiến đề xuất

Xuất phát từ RQ3, không phải ý tưởng thả từ trên trời:

> **Thay bước lọc triple bằng LLM bằng một cross-encoder reranker nhỏ** (`bge-reranker-v2-m3`, 568M) — và thêm
> **cơ chế bỏ qua thích ứng**: khi số triple ứng viên đã dưới ngưỡng, không lọc nữa.

Lý do tin là nó ăn thua: (a) xử lý trực tiếp lỗi "0 triple" mà tác giả tự chỉ ra; (b) cross-encoder được huấn luyện
đúng cho việc chấm liên quan, trong khi LLM nhỏ phải làm việc đó zero-shot; (c) rẻ hơn hẳn — bỏ được một lượt gọi LLM
trên mỗi truy vấn, nên cải thiện cả hai trục của RQ4. Không cần huấn luyện gì thêm.

Nếu cải tiến này không ăn thua, kết quả âm tính vẫn là kết quả — nó chỉ ra lọc triple cần năng lực *sinh*, không chỉ
năng lực *chấm điểm*, và đó cũng là một phát hiện.

## 5. Thiết kế thực nghiệm

**Dataset:** MuSiQue, 2WikiMultihopQA, HotpotQA (mỗi bộ 1,000 query + corpus riêng, đã có sẵn trong repo).
Nếu còn thời gian, thêm NQ / PopQA để kiểm tra phía single-hop — paper cho thấy lợi thế của HippoRAG 2 ở đó nhỏ hơn nhiều.

**Baseline:** (1) không retrieval; (2) BM25; (3) dense retriever thuần; (4) HippoRAG 1; (5) HippoRAG 2 (tái lập);
(6) HippoRAG 2 + cải tiến ở §4. LightRAG chạy **một** cấu hình duy nhất để kiểm chứng con số chi phí của Bảng 12 —
không chạy full vì đắt gấp 7.4×.

**Metric:** Recall@5 (retrieval), EM / F1 (QA) — giống paper. Cộng thêm: token index, phút index, giây/truy vấn,
VRAM đỉnh, và **F1 / GPU-hour**.

**Khung chạy:** FlashRAG (WWW 2025) cho baseline và metric — đã có sẵn BM25, dense, và cả `adaptive`, `ircot`, `selfrag`
được cài đặt thống nhất, nên không phải tự viết baseline và cũng không phải tự bào chữa cho chúng.

## 6. Ngân sách compute

Phần cứng: **1× RTX 5090 (32 GB, sm_120)**, 32 core, 123 GB RAM, 878 GB trống.

Suy từ Bảng 12 của paper (9.2M input + 3.0M output token / dataset), với Llama-3.1-8B chạy vLLM có batching:

| Hạng mục | Ước tính |
|---|---|
| Index 1 dataset, 1 cấu hình LLM | ~30–60 phút |
| QA 1,000 query | ~20 phút |
| RQ1 (3 dataset × 8B) | ~3–4 h |
| RQ2 (quét 3 LLM × 3 embedder, MuSiQue + 2Wiki) | ~10–14 h |
| RQ3 (ablation thành phần, MuSiQue) | ~4–6 h |
| RQ4 + cải tiến §4 | ~4–6 h |
| **Tổng** | **~25–30 GPU-hour**, chia được thành các lần chạy rời 1–2 h |

Không có lần chạy nào cần quá 2 giờ liên tục — hợp với ràng buộc "không chạy được lâu" vì máy là tài nguyên chung.
Mỗi bước index đều cache lại, nên lần chạy bị cắt giữa chừng không mất hết.

> ⚠️ **Các con số trên là ước tính từ token count của paper, chưa đo trên máy.** Việc đầu tiên của tuần 1 là
> benchmark thông lượng thật của vLLM + Llama-3.1-8B trên 5090 và cập nhật lại bảng này.

VRAM: LLM 8B (fp16, vLLM, `gpu-memory-utilization` hạ bớt) ~18–20 GB; embedder 1.5B ~3 GB; PPR chạy trên CPU bằng
igraph. Tách pha index/embed nếu chật. NV-Embed-v2 (7B, ~16 GB) chỉ chạy được khi tách pha hẳn — vì vậy nó nằm ngoài
cấu hình mặc định và chỉ dùng cho một lần kiểm chứng.

## 7. Rủi ro và phương án dự phòng

| Rủi ro | Xử lý |
|---|---|
| Repo pin `torch==2.5.1`, không có sm_120 | Gỡ pin; torch chỉ dùng cho embedder. LLM chạy qua HTTP ở env `mri5090` (torch 2.11+cu128) |
| vLLM chưa cài, bản mới có thể lỗi trên Blackwell | Fallback sang HF `transformers` + batching (chậm hơn ~3× nhưng vẫn trong ngân sách) |
| Không tái lập được con số paper | Đã dự liệu: RQ1 nhắm vào *thứ hạng*, không phải giá trị tuyệt đối. Khoảng cách chính là dữ liệu cho RQ2 |
| Hết thời gian | Cắt theo thứ tự ưu tiên RQ1 → RQ3 → RQ2 → RQ4. RQ1 + RQ3 riêng đã đủ thành một bài nộp được |
| Toàn bộ hướng HippoRAG bị chặn | **Phương án B: MiniRAG** ([2501.06713](https://arxiv.org/abs/2501.06713), repo đã clone) — không pin torch, model đích 1.5–4B, dataset đóng gói sẵn. Gần như không có rủi ro hạ tầng |

## 8. Timeline (8 tuần — cần chỉnh theo hạn thật của thầy)

| Tuần | Việc |
|---|---|
| 1 | Dựng env, cài vLLM + FlashRAG, **benchmark thông lượng thật**, chạy được `sample` dataset end-to-end |
| 2 | RQ1: tái lập 3 dataset với Llama-3.1-8B; đối chiếu với bảng của paper |
| 3–4 | RQ2: quét LLM × embedder |
| 5 | RQ3: ablation thành phần, tập trung vào triple filtering |
| 6 | Cải tiến §4: cross-encoder filter + bỏ qua thích ứng |
| 7 | RQ4: đo chi phí, so sánh cùng ngân sách compute |
| 8 | Viết báo cáo, dựng biểu đồ, slide |

## 9. Sản phẩm bàn giao

1. Báo cáo trình bày lại paper: ý tưởng, ánh xạ thần kinh học, kết quả, và **điểm yếu** (phụ thuộc compute, độ mong
   manh của bước lọc triple, lợi thế thu hẹp ở single-hop).
2. Bảng tái lập đặt cạnh bảng gốc của paper, kèm phân tích chênh lệch.
3. Đường cong thu-nhỏ-compute — sản phẩm chính: lợi thế graph-RAG như một hàm của năng lực LLM và embedder.
4. Ablation thành phần dưới SLM, đối chiếu với ablation 70B của paper.
5. Cải tiến cross-encoder filter, kèm cả kết quả âm tính nếu có.
6. Bảng chi phí đo thật + F1/GPU-hour.
7. Code và script tái lập trong `rag_prj/source/`.

## 10. Lăng kính bổ sung cho phần thảo luận

Ba paper đã tải về, dùng để soi phần hạn chế thay vì chỉ nêu cảm tính:

- **[Sufficient Context (ICLR 2025)](https://arxiv.org/abs/2411.06037)** — phân biệt "liên quan" với "đủ để trả lời".
  Dùng để phân tích: khi HippoRAG 2 lấy đúng passage mà vẫn sai, là hỏng ở retrieval hay ở reader? (Repo chỉ có
  README + prompt, không có code — nên dùng làm khung phân tích, không phải để tái lập.)
- **[Unbiased Evaluation for GraphRAG (2506.06331)](https://arxiv.org/abs/2506.06331)** — chứng minh protocol đánh giá
  graph-RAG bị lệch (so LightRAG với chính nó vẫn ra win-rate 90/10). Cơ sở để yêu cầu so sánh cùng ngân sách token ở RQ4.
- **[RAG vs GraphRAG (2502.11371)](https://arxiv.org/abs/2502.11371)** — khảo sát hệ thống về việc khi nào đồ thị thực
  sự đáng công.
