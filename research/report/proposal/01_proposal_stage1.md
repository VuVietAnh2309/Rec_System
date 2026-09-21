# Proposal — Stage 1

**Paper chọn:** *MiniRAG: Towards Extremely Simple Retrieval-Augmented Generation*
Tianyu Fan, Jingyuan Wang, Xubin Ren, Chao Huang (HKU) — [arXiv:2501.06713](https://arxiv.org/abs/2501.06713), v1 **12/01/2025**
Code: [HKUDS/MiniRAG](https://github.com/HKUDS/MiniRAG)

**Tiêu đề đề tài đề xuất:**
> *Lợi thế của MiniRAG đến từ truy xuất tốt hơn, hay từ việc nó ít chịu nói "không biết"?
> Tái lập và mổ xẻ một hệ RAG đồ thị dành cho mô hình nhỏ*

---

## 1. Paper nói gì

Hầu hết hệ RAG hiện đại giả định phía sau là một LLM mạnh — dùng để rút trích thực thể, viết mô tả quan hệ,
tóm tắt, và đọc hiểu ngữ cảnh dài. Khi thay bằng **Small Language Model (SLM, 1.5–4B)** để chạy trên thiết bị
biên hoặc môi trường riêng tư, các hệ này sụp. Paper dẫn con số: LightRAG rơi từ 56.90% xuống 35.42% khi đổi
từ LLM sang SLM, còn GraphRAG **hỏng hoàn toàn**, không sinh nổi nội dung dùng được.

MiniRAG đề xuất một pipeline được thiết kế để **không đòi hỏi năng lực sinh văn bản cao**:

1. **Đồ thị dị thể (heterogeneous graph)** với hai loại node: **node chunk** (đoạn văn gốc) và **node thực thể**.
   SLM chỉ phải làm việc dễ là *rút tên thực thể*, không phải viết mô tả tóm tắt như GraphRAG/LightRAG.
   Giữ nguyên chunk gốc trong đồ thị nghĩa là không phụ thuộc vào khả năng tóm tắt của SLM.
2. **Hai loại cạnh**: thực thể–thực thể (quan hệ ngữ nghĩa) và thực thể–chunk (thực thể này trích từ chunk nào).
3. **Truy xuất theo topology**: embed câu hỏi bằng một embedder rất nhẹ để tìm *thực thể hạt giống*, rồi đi theo
   đường đi trên đồ thị để lần ra các chunk liên quan, rồi rank. Cấu trúc đồ thị gánh phần suy luận thay cho model.

**Kết quả công bố:** hiệu quả cao hơn 1.3–2.5× trong khi chỉ tốn **25% dung lượng lưu trữ** so với LightRAG.
Paper cũng đóng góp **LiHua-World**, benchmark mô phỏng kịch bản on-device (một năm tin nhắn của người dùng ảo LiHua).

## 2. Vì sao chọn paper này

Chi tiết so sánh 9 ứng viên ở [`00_shortlist_papers.md`](./00_shortlist_papers.md). Ba lý do:

**a) Vừa đúng phần cứng.** Em có 1× RTX 5090 (32 GB) và không thể chiếm máy lâu vì là tài nguyên chung.
MiniRAG nhắm thẳng vào Phi-3.5-mini (3.8B), Qwen2.5-3B, GLM-Edge-1.5B, MiniCPM3-4B — đều nằm thoải mái trong 32 GB.
Embedder là `all-MiniLM-L6-v2`, **22 triệu tham số**. Corpus LiHua-World chỉ ~1 MB (496 file), 637 câu hỏi.
Repo **không pin phiên bản torch**, nên tránh được vấn đề sm_120 (Blackwell) vốn làm hỏng phần lớn repo RAG 2024.

**b) Đọc là hiểu.** Phần method của paper gần như toàn diễn giải bằng lời cộng một hình minh hoạ: hai loại node,
hai loại cạnh, tìm thực thể hạt giống bằng cosine similarity, đi đường trên đồ thị, chấm điểm chunk.
Không có Personalized PageRank, không có power iteration, không có đại số ma trận. Đây là điểm em cân nhắc kỹ —
một paper mà em nắm chắc toàn bộ cơ chế thì mới phản biện được, còn paper hiểu lơ mơ thì chỉ đọc lại abstract.

**c) Có lỗ hổng thật để mổ xẻ.** Xem §3 — và đây mới là phần chính của đồ án.

## 3. Quan sát khởi nguồn: lợi thế của MiniRAG có thể bị nhiễu

Đọc bảng kết quả MultiHop-RAG trong chính paper (`acc↑` / `err↓`, đơn vị %):

| Model | NaiveRAG | LightRAG | MiniRAG |
|---|---|---|---|
| Phi-3.5-mini | 42.72 / 31.34 | 27.03 / **11.78** | **49.96** / 28.44 |
| Qwen2.5-3B | 39.48 / 31.69 | 21.91 / **13.73** | **48.55** / 33.10 |
| MiniCPM3-4B | 39.24 / 31.42 | 19.48 / **10.41** | **47.77** / 26.88 |

MiniRAG thắng rõ về accuracy. Nhưng **error rate của nó cao gấp 2.0–2.4 lần LightRAG**. Và vì `acc + err ≠ 100`,
tồn tại một nhóm thứ ba — các câu model không trả lời được. Với Qwen2.5-3B: LightRAG bỏ trống **64.4%**,
MiniRAG chỉ **18.4%**.

Nghĩa là hai hệ này đang ở **hai điểm vận hành khác hẳn nhau** trên trục đánh đổi *trả lời nhiều ↔ trả lời đúng*.
Một phần lợi thế accuracy của MiniRAG có thể chỉ đơn giản là **nó chịu đoán nhiều hơn**, chứ không phải nó
truy xuất tốt hơn. Paper không bàn tới điều này.

Đây đúng là hiện tượng mà [Sufficient Context (ICLR 2025)](https://arxiv.org/abs/2411.06037) mô tả: thêm ngữ cảnh
làm model tự tin hơn, nên nó *bịa* thay vì *im lặng*.

**Và LiHua-World có đủ dữ liệu để kiểm chứng chuyện này mà không tốn thêm một giờ GPU nào.** Em đã kiểm tra
`query_set.csv`:

| Thuộc tính | Giá trị |
|---|---|
| Tổng query | 637 |
| Phân loại | **506 Single-hop, 66 Multi-hop, 65 Null** |
| Cột `Evidence` | **có đủ cho cả 637 query** — tài liệu ground-truth |
| Trung bình tài liệu/query | 1.15 |
| Query có đáp án đúng là "Insufficient information" | **65** |

Ba hệ quả:

- Có `Evidence` nghĩa là **đo được Recall của khâu truy xuất một cách trực tiếp**, không cần LLM làm giám khảo.
  Tách bạch được "truy xuất hỏng" với "sinh câu trả lời hỏng" — điều paper không làm.
- 65 câu **Null** là phép thử khả năng từ chối đã được cài sẵn trong dataset. Paper gộp tất cả vào một con số `acc`.
- **506/637 = 79% câu hỏi là single-hop**, trung bình chỉ cần 1.15 tài liệu. Một hệ suy luận đa bước trên đồ thị
  lại được đánh giá chủ yếu trên câu hỏi một bước — cần xem lại lợi thế thực sự nằm ở đâu.

## 4. Câu hỏi nghiên cứu

**RQ1 — Tái lập.** Tái lập bảng chính trên LiHua-World và MultiHop-RAG, với NaiveRAG / LightRAG / MiniRAG
× các SLM. Con số có khớp paper không?

**RQ2 — Lợi thế đến từ đâu? (đóng góp chính).** Tách chỉ số `acc` gộp thành ba phần *đúng / sai / từ chối*,
và tách riêng hai tầng:
- **Tầng truy xuất:** Recall@k tính bằng cột `Evidence`, hoàn toàn độc lập với LLM giám khảo.
- **Tầng sinh:** với cùng một tập tài liệu truy xuất được, các hệ trả lời tốt đến đâu?

*Giả thuyết:* lợi thế của MiniRAG ở tầng truy xuất nhỏ hơn nhiều so với chênh lệch `acc` được báo cáo, và phần
lớn khoảng cách đến từ khác biệt hành vi từ chối trả lời.

**RQ3 — Bóc tách theo loại câu hỏi.** Báo cáo riêng cho Single / Multi / Null. *Giả thuyết:* lợi thế tập trung ở
66 câu Multi-hop, còn trên 506 câu Single-hop thì MiniRAG ngang hoặc thua NaiveRAG — nếu đúng thì con số tổng
đang bị 79% câu single-hop pha loãng, và động lực "đồ thị để suy luận đa bước" chưa được chứng minh.

**RQ4 — GraphRAG có "hỏng hoàn toàn" thật không?** Paper để trống ô GraphRAG với SLM và giải thích là hệ thống
sụp. Cần kiểm tra đây là giới hạn thật của phương pháp hay chỉ là lỗi parse output của SLM không đúng định dạng.
Khác biệt này quan trọng: một bên là kết luận khoa học, một bên là lỗi kỹ thuật.

**RQ5 — Tái lập ablation.** Paper có sẵn ba biến thể: `-I` (thay indexing đồ thị dị thể bằng indexing dựa mô tả),
`-R_edge` (bỏ thông tin cạnh), `-R_chunk` (bỏ node chunk). Chạy lại và đối chiếu.

## 5. Cải tiến đề xuất

Xuất phát trực tiếp từ RQ2, không phải ý tưởng thả từ trên trời:

> **Gắn ngưỡng từ chối cho MiniRAG, rồi so sánh các hệ trên đường cong accuracy–coverage thay vì một con số đơn lẻ.**

Cụ thể: dùng điểm số truy xuất của MiniRAG (độ tương đồng thực thể hạt giống, độ dài đường đi trên đồ thị) làm tín
hiệu tin cậy. Dưới ngưỡng thì trả lời "không đủ thông tin". Quét ngưỡng để vẽ đường cong, rồi đặt LightRAG và
NaiveRAG lên cùng đồ thị. Khi đó mới trả lời được câu hỏi công bằng: **ở cùng một mức coverage, hệ nào chính xác hơn?**

Việc này rẻ (chỉ là hậu xử lý điểm số đã có, không cần chạy lại index), đo được trực tiếp trên 65 câu Null, và
áp dụng đúng phương pháp luận của Sufficient Context vào một paper chưa từng được soi bằng lăng kính đó.

Nếu kết quả cho thấy MiniRAG vẫn thắng ở mọi mức coverage thì đó là **bằng chứng mạnh hơn** cho paper so với con
số gốc. Dù kết quả ngả về bên nào, đồ án vẫn có kết luận.

## 6. Thiết kế thực nghiệm

**Dataset:** LiHua-World (đóng gói sẵn trong repo, 496 file / 637 query) là dataset chính vì có cột `Evidence`
và nhãn `Type`. MultiHop-RAG là dataset phụ, cần tải thêm.

**Hệ so sánh:** NaiveRAG, LightRAG, MiniRAG, MiniRAG + ngưỡng từ chối (§5), và ba biến thể ablation của RQ5.
GraphRAG chạy riêng cho RQ4.

**Mô hình:** Phi-3.5-mini-instruct (3.8B) và Qwen2.5-3B-Instruct làm cấu hình chính. Thêm GLM-Edge-1.5B
và MiniCPM3-4B nếu còn thời gian. gpt-4o-mini làm mốc trần trong paper — **thay bằng một model local lớn hơn**
(Qwen2.5-14B-Instruct lượng tử hoá) để không phát sinh chi phí API.

**Metric:**
- Truy xuất: Recall@k dựa trên cột `Evidence` — *không có trong paper, là đóng góp của đồ án*
- Sinh: accuracy / error / abstention, bóc tách theo Single / Multi / Null
- Đường cong accuracy–coverage
- Chi phí: thời gian index, dung lượng lưu trữ (kiểm chứng lại con số "25%"), VRAM đỉnh, giây/truy vấn

**Về giám khảo:** paper dùng gpt-4o-mini chấm điểm. Đây đúng loại protocol mà
[Unbiased Evaluation for GraphRAG](https://arxiv.org/abs/2506.06331) chứng minh là bị lệch (so LightRAG với chính
nó vẫn ra win-rate 90/10). Vì vậy đồ án **ưu tiên các chỉ số không cần giám khảo** (Recall@k theo `Evidence`, và
khớp chuỗi với `Gold Answer` — phần lớn đáp án là Yes/No/tên riêng nên khớp được).

## 7. Ngân sách compute

Phần cứng: 1× RTX 5090 (32 GB, sm_120), 32 core, 123 GB RAM, 878 GB trống.
Env sẵn có: `/data/anhvv/envs/mri5090` (torch 2.11.0+cu128, hỗ trợ sm_120).

| Hạng mục | Ước tính |
|---|---|
| Index LiHua-World (~1 MB), 1 model | ~15–40 phút |
| Trả lời 637 query | ~10–20 phút |
| RQ1 toàn bộ (2 dataset × 2 model × 3 hệ) | ~4–6 h |
| RQ2/RQ3 (phân tích lại log, **không tốn GPU**) | ~0 h |
| RQ4 (GraphRAG) | ~2–4 h |
| RQ5 (3 biến thể ablation) | ~2–3 h |
| Cải tiến §5 (hậu xử lý điểm số) | ~1 h |
| **Tổng** | **~10–15 GPU-hour**, mỗi lần chạy dưới 1 giờ |

> ⚠️ Đây là ước tính từ kích thước dữ liệu, **chưa đo trên máy**. Việc đầu tiên của tuần 1 là đo thật và cập nhật bảng này.

Điểm đáng chú ý: **RQ2 và RQ3 — phần đóng góp chính — gần như không tốn GPU**, vì chúng phân tích lại log sinh ra
từ RQ1. Nghĩa là rủi ro "hết thời gian máy" không đe doạ phần giá trị nhất của đồ án.

## 8. Rủi ro và phương án dự phòng

| Rủi ro | Xử lý |
|---|---|
| Không tái lập được con số paper | RQ2/RQ3 không phụ thuộc vào việc khớp tuyệt đối — chúng phân tích *chính các lần chạy của mình*. Chênh lệch so với paper tự nó là một phát hiện cần báo cáo |
| Cột `Evidence` khó khớp với ID chunk sau khi index | Kiểm tra ngay tuần 1. `Evidence` là timestamp khớp trực tiếp với tên file (`20260118_1200.txt`) nên khả năng cao là khớp được |
| GraphRAG không chạy nổi (RQ4) | Đó chính là kết quả của RQ4 — chỉ cần ghi lại *hỏng ở đâu* (lỗi parse hay chất lượng nội dung) |
| MultiHop-RAG tải/chuẩn bị mất thời gian | Cắt bỏ. LiHua-World một mình đã đủ cho RQ1–RQ5 |
| Thiếu thời gian | Ưu tiên RQ1 → RQ2 → RQ3 → §5 → RQ5 → RQ4. Bốn mục đầu đã đủ thành bài nộp hoàn chỉnh |

## 9. Timeline

> Đang để 8 tuần làm giả định — **cần chỉnh lại theo hạn thật của Stage 1**.

| Tuần | Việc |
|---|---|
| 1 | Dựng env, cài MiniRAG, chạy được LiHua-World end-to-end với Phi-3.5-mini; kiểm tra khớp `Evidence` ↔ chunk; đo thông lượng thật |
| 2 | RQ1: tái lập NaiveRAG / LightRAG / MiniRAG × 2 model, lưu đầy đủ log truy xuất |
| 3 | RQ2: tách truy xuất khỏi sinh, tính Recall@k, bóc tách đúng/sai/từ chối |
| 4 | RQ3: bóc tách theo Single / Multi / Null |
| 5 | §5: ngưỡng từ chối, vẽ đường cong accuracy–coverage |
| 6 | RQ5 ablation + RQ4 GraphRAG |
| 7 | Đo chi phí, kiểm chứng con số "25% dung lượng" |
| 8 | Viết báo cáo, dựng biểu đồ, slide |

## 10. Sản phẩm bàn giao

1. Báo cáo trình bày lại paper: động cơ (RAG cho SLM), cơ chế đồ thị dị thể, truy xuất theo topology, kết quả.
2. Bảng tái lập đặt cạnh bảng gốc của paper.
3. **Phân tích tách tầng truy xuất/sinh và bóc tách theo loại câu hỏi** — phần này paper không có.
4. Đường cong accuracy–coverage so sánh công bằng giữa các hệ.
5. Tái lập ablation + kết luận về GraphRAG "complete failure".
6. Bảng chi phí đo thật.
7. Code và script tái lập trong `rag_prj/source/`.

## 11. Ba góc nhìn cho phần thảo luận

- **Điểm mạnh:** vấn đề có thật và chưa được giải quyết (RAG cho SLM); thiết kế hợp với động cơ (đẩy gánh nặng từ
  năng lực sinh sang cấu trúc đồ thị); code sạch, dữ liệu mở, rẻ.
- **Điểm yếu:** benchmark chính do chính nhóm tác giả tạo ra, và MiniRAG lại được so với LightRAG — cũng của nhóm đó;
  metric dựa trên LLM giám khảo; 79% câu hỏi là single-hop trong khi bài toán đặt ra là suy luận đa bước;
  hành vi từ chối trả lời không được kiểm soát khi so sánh.
- **Lăng kính bổ sung:** [Sufficient Context (ICLR 2025)](https://arxiv.org/abs/2411.06037) cho khung phân tích
  đúng/sai/từ chối; [Unbiased Evaluation for GraphRAG (2506.06331)](https://arxiv.org/abs/2506.06331) cho lập luận
  vì sao cần tránh giám khảo LLM và so sánh cùng ngân sách.
