# Nền Tảng Học Tăng Cường (Reinforcement Learning - RL)

Tài liệu tổng hợp và phân tích chuyên sâu về các nguyên lý cơ bản của Học tăng cường, hệ thống phương trình Bellman, so sánh hai phương pháp nền tảng **Quy hoạch động (Dynamic Programming - DP)** và **Monte Carlo (MC)**, cùng với phân tích chi tiết các thuật toán đại diện.

Nội dung được đối chiếu và hệ thống hóa dựa trên bộ bài giảng kinh điển về Học tăng cường của GS. David Silver (UCL & Google DeepMind).

---

## Mục Lục
- [1. Định Nghĩa Học Tăng Cường và Các Thành Phần Cốt Lõi](#1-định-nghĩa-học-tăng-cường-và-các-thành-phần-cốt-lõi)
  - [1.1. Định nghĩa Học tăng cường (Reinforcement Learning)](#11-định-nghĩa-học-tăng-cường-reinforcement-learning)
  - [1.2. Các đặc trưng khác biệt của Học tăng cường](#12-các-đặc-trưng-khác-biệt-của-học-tăng-cường)
  - [1.3. Các thành phần chính và ý nghĩa](#13-các-thành-phần-chính-và-ý-nghĩa)
- [2. Các Phương Trình Bellman (Bellman Equations)](#2-các-phương-trình-bellman-bellman-equations)
  - [2.1. Phương trình Bellman Kỳ vọng (Bellman Expectation Equation)](#21-phương-trình-bellman-kỳ-vọng-bellman-expectation-equation)
  - [2.2. Phương trình Bellman Tối ưu (Bellman Optimality Equation)](#22-phương-trình-bellman-tối-ưu-bellman-optimality-equation)
- [3. So Sánh Quy Hoạch Động (DP) và Monte Carlo (MC)](#3-so-sánh-quy-hoạch-động-dp-và-monte-carlo-mc)
  - [3.1. Bảng đối chiếu tổng quan](#31-bảng-đối-chiếu-tổng-quan)
  - [3.2. Ý tưởng và phạm vi áp dụng của Quy hoạch động (DP)](#32-ý-tưởng-và-phạm-vi-áp-dụng-của-quy-hoạch-động-dp)
  - [3.3. Ý tưởng và phạm vi áp dụng của Monte Carlo (MC)](#33-ý-tưởng-và-phạm-vi-áp-dụng-của-monte-carlo-mc)
- [4. Phân Tích Chi Tiết Hai Thuật Toán Đại Diện](#4-phân-tích-chi-tiết-hai-thuật-toán-đại-diện)
  - [4.1. Thuật toán Quy hoạch động: Policy Iteration (Lặp chính sách)](#41-thuật-toán-quy-hoạch-động-policy-iteration-lặp-chính-sách)
  - [4.2. Thuật toán Monte Carlo: GLIE Monte Carlo Control](#42-thuật-toán-monte-carlo-glie-monte-carlo-control)
- [5. Tài Liệu Bài Giảng Đính Kèm](#5-tài-liệu-bài-giảng-đính-kèm)

---

## 1. Định Nghĩa Học Tăng Cường và Các Thành Phần Cốt Lõi

### 1.1. Định nghĩa Học tăng cường (Reinforcement Learning)
**Học tăng cường (Reinforcement Learning - RL)** là một nhánh nền tảng của Trí tuệ nhân tạo và Học máy, tập trung vào bài toán **ra quyết định theo chuỗi thời gian (sequential decision making)**: một **Tác tử (Agent)** học cách ứng xử trong một **Môi trường (Environment)** chưa biết trước thông qua cơ chế **thử - và - sai (trial-and-error)**, nhằm **tối đa hóa phần thưởng tích lũy kỳ vọng** theo thời gian.

Nói một cách trực quan: Học tăng cường giống như việc huấn luyện một sinh vật thích nghi với thế giới — không có người chỉ dẫn từng bước phải làm gì, tác tử tự mình thực hiện hành động, nhận về kết quả (thưởng nếu làm đúng, phạt nếu làm sai), và qua hàng ngàn lần tương tác sẽ tự tìm ra chiến lược tối ưu nhất.

```
                     Hành động A_t (Action)
             ┌─────────────────────────────────────┐
             │                                     ▼
      ┌──────────────┐                      ┌──────────────┐
      │              │                      │              │
      │ TÁC TỬ       │                      │ MÔI TRƯỜNG   │
      │ (Agent)      │                      │ (Environment)│
      │              │                      │              │
      └──────────────┘                      └──────────────┘
             ▲                                     │
             │       Quan sát S_{t+1} (State)      │
             └─────────────────────────────────────┘
                     Phần thưởng R_{t+1} (Reward)
```

### 1.2. Các đặc trưng khác biệt của Học tăng cường
Khác với Học có giám sát (Supervised Learning) và Học không giám sát (Unsupervised Learning):
1. **Không có giám sát viên (No supervisor):** Tác tử không được cung cấp nhãn đúng/sai, chỉ nhận tín hiệu phần thưởng đánh giá độ hiệu quả.
2. **Phản hồi bị trễ (Delayed feedback):** Một hành động thực hiện ở thời điểm hiện tại có thể nhiều bước sau mới đem lại kết quả (ví dụ: nước đi thí quân trong ván cờ).
3. **Thời gian mang tính quyết định (Time matters):** Dữ liệu thu thập tuần tự, mang tính phụ thuộc thời gian (không thỏa mãn giả định độc lập và cùng phân phối - *non-i.i.d*).
4. **Hành động của tác tử quyết định dữ liệu tiếp theo:** Mỗi lựa chọn của tác tử tác động trực tiếp đến trạng thái tương lai của môi trường và dữ liệu mà nó sẽ quan sát được.

---

### 1.3. Các thành phần chính và ý nghĩa

1. **Tác tử (Agent):**
   * *Ý nghĩa:* Thực thể đưa ra quyết định hành động và thực hiện việc học hỏi.
2. **Môi trường (Environment):**
   * *Ý nghĩa:* Không gian bên ngoài mà tác tử tương tác, tiếp nhận hành động và phản hồi trạng thái mới cùng phần thưởng.
3. **Lịch sử (History) và Trạng thái (State - $S$):**
   * *Lịch sử ($H_t$):* Toàn bộ chuỗi trải nghiệm quan sát, hành động, phần thưởng từ đầu đến thời điểm $t$:

$$
H_t = O_1, R_1, A_1, \dots, A_{t-1}, O_t, R_t
$$

   * *Trạng thái ($S_t$):* Thông tin tóm tắt dùng để xác định diễn biến tiếp theo, $S_t = f(H_t)$.
   * *Tính chất Markov (Markov Property):* Một trạng thái là Markov khi:

$$
\mathbb{P}[S_{t+1} \mid S_t] = \mathbb{P}[S_{t+1} \mid S_1, S_2, \dots, S_t]
$$

   *(Nghĩa là: Tương lai độc lập với quá khứ một khi đã biết trạng thái hiện tại).*

4. **Hành động (Action - $A$):**
   * *Ý nghĩa:* Quyết định mà tác tử có thể đưa ra tại mỗi bước. Tập hợp tất cả các hành động khả dĩ ký hiệu là $\mathcal{A}$.
5. **Phần thưởng (Reward - $R_t$):**
   * *Ý nghĩa:* Tín hiệu số thực vô hướng phản ánh mức độ tốt/xấu tức thời của hành vi tại bước $t$.
   * *Giả thuyết phần thưởng (Reward Hypothesis):* Mọi mục tiêu đều có thể quy về việc tối đa hóa phần thưởng tích lũy kỳ vọng.
6. **Lợi tức tích lũy (Return - $G_t$):**
   * *Ý nghĩa:* Tổng phần thưởng chiết khấu mà tác tử nhận được từ thời điểm $t$ cho đến khi kết thúc:

$$
G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \dots = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}
$$

   Với $\gamma \in [0, 1]$ là **hệ số chiết khấu (discount factor)**:
   * $\gamma \to 0$: Tác tử thiển cận (*myopic*), chỉ quan tâm phần thưởng trước mắt.
   * $\gamma \to 1$: Tác tử nhìn xa trông rộng (*far-sighted*), ưu tiên phần thưởng bền vững dài hạn.
7. **Chính sách (Policy - $\pi$):**
   * *Ý nghĩa:* Hàm hành vi quy định cách tác tử lựa chọn hành động dựa trên trạng thái hiện tại.
     * Chính sách tất định (*Deterministic policy*): $a = \pi(s)$.
     * Chính sách ngẫu nhiên (*Stochastic policy*): $\pi(a \mid s) = \mathbb{P}[A_t = a \mid S_t = s]$.
8. **Hàm giá trị (Value Function):**
   * *Ý nghĩa:* Đo lường mức độ "tốt" của một trạng thái hoặc một cặp trạng thái - hành động về mặt lợi tức kỳ vọng lâu dài:
     * **Hàm giá trị trạng thái ($v_\pi(s)$ - State-Value Function):**

$$
v_\pi(s) = \mathbb{E}_\pi [G_t \mid S_t = s]
$$

     * **Hàm giá trị hành động ($q_\pi(s, a)$ - Action-Value Function):**

$$
q_\pi(s, a) = \mathbb{E}_\pi [G_t \mid S_t = s, A_t = a]
$$

9. **Mô hình môi trường (Model):**
   * *Ý nghĩa:* Đại diện giả lập bên trong não bộ của tác tử về cơ chế vận hành của môi trường:
     * Ma trận chuyển trạng thái: $\mathcal{P}_{ss'}^a = \mathbb{P}[S_{t+1} = s' \mid S_t = s, A_t = a]$
     * Hàm phần thưởng: $\mathcal{R}_s^a = \mathbb{E}[R_{t+1} \mid S_t = s, A_t = a]$

---

## 2. Các Phương Trình Bellman (Bellman Equations)

Các phương trình Bellman thiết lập mối quan hệ đệ quy nền tảng: **Giá trị hiện tại bằng phần thưởng nhận ngay cộng với giá trị chiết khấu của trạng thái tiếp theo**.

```
          s                    s
         / \                  / \
        /   \                /   \
       a     a              a     a
      / \   / \            / \   / \
     s'  s' s' s'         s'  s' s' s'
  (Expectation Backup)  (Optimality Backup - lấy max)
```

### 2.1. Phương trình Bellman Kỳ vọng (Bellman Expectation Equation)
Dùng để đánh giá giá trị của một chính sách $\pi$ đã biết (*Policy Evaluation*):

#### a) Cho State-Value Function $v_\pi(s)$:
Định nghĩa đệ quy kỳ vọng:

$$
v_\pi(s) = \mathbb{E}_\pi [R_{t+1} + \gamma v_\pi(S_{t+1}) \mid S_t = s]
$$

Triển khai chi tiết qua không gian hành động $\mathcal{A}$ và không gian trạng thái kế tiếp $\mathcal{S}$:

$$
v_\pi(s) = \sum_{a \in \mathcal{A}} \pi(a \mid s) \left( \mathcal{R}_s^a + \gamma \sum_{s' \in \mathcal{S}} \mathcal{P}_{ss'}^a v_\pi(s') \right)
$$

---

#### b) Cho Action-Value Function $q_\pi(s, a)$:
Định nghĩa đệ quy kỳ vọng:

$$
q_\pi(s, a) = \mathbb{E}_\pi [R_{t+1} + \gamma q_\pi(S_{t+1}, A_{t+1}) \mid S_t = s, A_t = a]
$$

Triển khai chi tiết:

$$
q_\pi(s, a) = \mathcal{R}_s^a + \gamma \sum_{s' \in \mathcal{S}} \mathcal{P}_{ss'}^a \sum_{a' \in \mathcal{A}} \pi(a' \mid s') q_\pi(s', a')
$$

> **Giải thích trực quan các thành phần trong công thức:**
> * $\mathcal{R}_s^a$: Phần thưởng nhận được ngay lập tức khi ở trạng thái $s$ và làm hành động $a$.
> * $\mathcal{P}_{ss'}^a$: Xác suất môi trường chuyển từ trạng thái $s$ sang $s'$ sau hành động $a$.
> * $\pi(a' \mid s')$: Xác suất tác tử sẽ chọn tiếp hành động $a'$ khi đến trạng thái mới $s'$.
> * $q_\pi(s', a')$: Giá trị hành động của bước tiếp theo $(s', a')$.
> * $\gamma$: Hệ số chiết khấu phần thưởng tương lai.

---

#### c) Dạng ma trận:

$$
v_\pi = \mathcal{R}^\pi + \gamma \mathcal{P}^\pi v_\pi
$$

Vì là hệ phương trình đại số tuyến tính, nghiệm giải tích đóng có thể tính trực tiếp:

$$
v_\pi = (I - \gamma \mathcal{P}^\pi)^{-1} \mathcal{R}^\pi
$$

---

### 2.2. Phương trình Bellman Tối ưu (Bellman Optimality Equation)
Đặc tả hàm giá trị dưới chính sách tối ưu $\pi^*$, khi tác tử luôn chọn hành động mang lại lợi ích cao nhất:

#### a) Cho Optimal State-Value Function $v_*(s)$:

$$
v_*(s) = \max_{a \in \mathcal{A}} q_*(s, a)
$$

Triển khai chi tiết:

$$
v_*(s) = \max_{a \in \mathcal{A}} \left( \mathcal{R}_s^a + \gamma \sum_{s' \in \mathcal{S}} \mathcal{P}_{ss'}^a v_*(s') \right)
$$

#### b) Cho Optimal Action-Value Function $q_*(s, a)$:

$$
q_*(s, a) = \mathcal{R}_s^a + \gamma \sum_{s' \in \mathcal{S}} \mathcal{P}_{ss'}^a \max_{a' \in \mathcal{A}} q_*(s', a')
$$

#### c) Đặc điểm cốt lõi:
* Phương trình Bellman tối ưu có chứa toán tử $\max$, do đó đây là **hệ phương trình phi tuyến (non-linear)**.
* Không có nghiệm dạng đại số đóng thông qua nghịch đảo ma trận $(I - \gamma \mathcal{P})^{-1}$.
* Cần các thuật toán lặp như **Value Iteration**, **Policy Iteration**, **Q-Learning**, **Sarsa** để giải tìm điểm hội tụ.

---

## 3. So Sánh Quy Hoạch Động (DP) và Monte Carlo (MC)

### 3.1. Bảng đối chiếu tổng quan

| Tiêu chí | Quy hoạch động (Dynamic Programming - DP) | Monte Carlo (MC) |
| :--- | :--- | :--- |
| **Bản chất bài toán** | **Lập kế hoạch (Planning):** Môi trường đã được mô hình hóa hoàn chỉnh. | **Học không cần mô hình (Model-Free Learning):** Học trực tiếp từ tương tác thực tế. |
| **Yêu cầu mô hình** | **Bắt buộc biết toàn bộ MDP** ($\mathcal{P}_{ss'}^a, \mathcal{R}_s^a$). | **Không cần mô hình** (*Model-free*); chỉ cần các tập mẫu dữ liệu trải nghiệm. |
| **Kiểu cập nhật** | **Full-width Backups:** Xét tất cả hành động và toàn bộ trạng thái tiếp theo. | **Sample Backups:** Cập nhật dọc theo một nhánh trải nghiệm mẫu thực tế. |
| **Bootstrapping** | **Có Bootstrapping:** Cập nhật giá trị hiện tại dựa vào ước lượng giá trị trạng thái kế tiếp ($v(s')$). | **Không Bootstrapping:** Cập nhật theo lợi tức thực tế $G_t$ thu được đến hết tập. |
| **Độ chệch & Phương sai** | Có **Bias** ban đầu do ước lượng, nhưng **Zero Variance** (tính toán xác định). | **Zero Bias** (không chệch), nhưng **High Variance** (phương sai cao do ngẫu nhiên tích lũy). |
| **Loại bài toán** | Áp dụng cho cả **Episodic** và **Continuing tasks**. | **Chỉ áp dụng cho Episodic tasks** (bắt buộc tập phải kết thúc để tính được $G_t$). |
| **Hạn chế chính** | Lời nguyền số chiều (*Curse of Dimensionality*) khi không gian trạng thái lớn. | Phải chờ tập kết thúc mới học được (*offline*), tốc độ hội tụ chậm do phương sai lớn. |

### 3.2. Ý tưởng và phạm vi áp dụng của Quy hoạch động (DP)
* **Ý tưởng:** Dựa vào hai nguyên lý tối ưu toán học:
  1. *Cấu trúc con tối ưu (Optimal substructure):* Nghiệm tối ưu có thể phân rã thành nghiệm của các bài toán con nhỏ hơn.
  2. *Bài toán con chồng lấn (Overlapping subproblems):* Các trạng thái lặp lại nhiều lần, có thể lưu vào bộ nhớ để tái sử dụng.
  DP sử dụng các phương trình Bellman làm toán tử co (*Contraction mapping*) để lặp tính giá trị trạng thái.
* **Phạm vi áp dụng:**
  * Dùng cho các bài toán đã biết đầy đủ mô hình toán học của môi trường (xác suất chuyển trạng thái và hàm thưởng).
  * Không gian trạng thái và hành động hữu hạn, kích thước vừa phải (dưới hàng triệu trạng thái).

### 3.3. Ý tưởng và phạm vi áp dụng của Monte Carlo (MC)
* **Ý tưởng:** Dựa trên nguyên lý thống kê thực nghiệm cơ bản:

$$
\text{Giá trị trạng thái} = \text{Trung bình mẫu của Lợi tức tích lũy} \quad (\text{Value} = \text{Mean Return})
$$

  Theo Luật số lớn (*Law of Large Numbers*), khi số lần ghé thăm một trạng thái $N(s) \to \infty$, giá trị trung bình mẫu hội tụ chính xác về giá trị kỳ vọng $v_\pi(s)$.
* **Phạm vi áp dụng:**
  * Áp dụng khi môi trường không có mô hình giải tích hoặc mô hình quá phức tạp nhưng có thể mô phỏng (lấy mẫu trải nghiệm).
  * Bắt buộc bài toán phải kết thúc (*Episodic MDPs*).

---

## 4. Phân Tích Chi Tiết Hai Thuật Toán Đại Diện

### 4.1. Thuật toán Quy hoạch động: Policy Iteration (Lặp chính sách)

#### a) Cơ chế hoạt động (Generalised Policy Iteration - GPI)
Thuật toán lặp chính sách giải bài toán tìm kiếm chính sách tối ưu bằng cách xen kẽ hai pha liên tiếp:
1. **Policy Evaluation (Đánh giá chính sách):** Tính toán chính xác hàm giá trị $v_\pi$ của chính sách hiện tại $\pi$ bằng cách lặp phương trình Bellman kỳ vọng.
2. **Policy Improvement (Cải tiến chính sách):** Tạo ra chính sách mới tốt hơn bằng cách hành động tham lam (*greedy*) theo hàm giá trị vừa tìm được: $\pi' = \text{greedy}(v_\pi)$.

$$
\pi_0 \xrightarrow{\text{Eval}} v_{\pi_0} \xrightarrow{\text{Improve}} \pi_1 \xrightarrow{\text{Eval}} v_{\pi_1} \dots \longrightarrow \pi^* \xrightarrow{\text{Eval}} v_*
$$

#### b) Chi tiết toán học từng bước:
* **Bước 1: Policy Evaluation**
  Cập nhật đồng bộ cho mọi trạng thái $s \in \mathcal{S}$ tại vòng lặp $k$:

$$
v_{k+1}(s) = \sum_{a \in \mathcal{A}} \pi(a \mid s) \left( \mathcal{R}_s^a + \gamma \sum_{s' \in \mathcal{S}} \mathcal{P}_{ss'}^a v_k(s') \right)
$$

  Lặp cho tới khi $\max_{s \in \mathcal{S}} |v_{k+1}(s) - v_k(s)| < \theta$.
* **Bước 2: Policy Improvement**
  Với mỗi trạng thái $s$, chọn hành động cực đại hóa giá trị hành động kỳ vọng:

$$
\pi'(s) = \arg\max_{a \in \mathcal{A}} q_\pi(s, a) = \arg\max_{a \in \mathcal{A}} \left( \mathcal{R}_s^a + \gamma \sum_{s' \in \mathcal{S}} \mathcal{P}_{ss'}^a v_\pi(s') \right)
$$

#### c) Chứng minh tính hội tụ (Policy Improvement Theorem):

$$
q_\pi(s, \pi'(s)) = \max_{a \in \mathcal{A}} q_\pi(s, a) \ge q_\pi(s, \pi(s)) = v_\pi(s)
$$

Khai triển theo thời gian:

$$
v_\pi(s) \le q_\pi(s, \pi'(s)) = \mathbb{E}_{\pi'}[R_{t+1} + \gamma v_\pi(S_{t+1}) \mid S_t = s] \le \dots \le v_{\pi'}(s)
$$

Do đó chính sách mới $\pi'$ luôn tốt hơn hoặc bằng chính sách cũ $\pi$. Khi không thể cải tiến được nữa ($\pi' = \pi$), ta có:

$$
v_\pi(s) = \max_{a \in \mathcal{A}} q_\pi(s, a)
$$

Đây chính là phương trình Bellman tối ưu, chứng minh chính sách đã hội tụ về $\pi^*$.

#### d) Mã giả Python:
```python
def policy_iteration(env, gamma=0.99, theta=1e-6):
    """Thuật toán Policy Iteration giải bài toán lập kế hoạch MDP."""
    # Khởi tạo giá trị V và chính sách pi
    V = {s: 0.0 for s in env.states}
    pi = {s: env.random_action(s) for s in env.states}
    
    while True:
        # --- 1. Policy Evaluation ---
        while True:
            delta = 0.0
            for s in env.states:
                v_old = V[s]
                a = pi[s]
                # Cập nhật Bellman Expectation Backup
                V[s] = env.R(s, a) + gamma * sum(
                    env.P(s, a, s_next) * V[s_next] for s_next in env.states
                )
                delta = max(delta, abs(v_old - V[s]))
            if delta < theta:
                break

        # --- 2. Policy Improvement ---
        policy_stable = True
        for s in env.states:
            old_action = pi[s]
            # Chọn hành động tốt nhất theo Q(s, a)
            best_action = max(
                env.actions,
                key=lambda a: env.R(s, a) + gamma * sum(
                    env.P(s, a, s_next) * V[s_next] for s_next in env.states
                )
            )
            pi[s] = best_action
            if best_action != old_action:
                policy_stable = False

        if policy_stable:
            break

    return pi, V
```

---

### 4.2. Thuật toán Monte Carlo: GLIE Monte Carlo Control

#### a) Cơ chế hoạt động
Trong bài toán Model-free Control, tác tử không biết xác suất chuyển trạng thái $\mathcal{P}_{ss'}^a$. Do đó, tác tử phải học trực tiếp hàm **$Q(s, a)$** thay vì $V(s)$ để có thể cải tiến chính sách mà không cần mô hình:

$$
\pi'(s) = \arg\max_{a \in \mathcal{A}} Q(s, a)
$$

Để tránh việc tác tử bị mắc kẹt vào các lựa chọn địa phương do thiếu khám phá (*Exploration vs. Exploitation*), thuật toán áp dụng nguyên lý **GLIE (Greedy in the Limit with Infinite Exploration)** kết hợp chiến lược **$\epsilon$-greedy**.

#### b) Nguyên lý GLIE & Chiến lược $\epsilon$-greedy:
* **Chiến lược $\epsilon$-greedy:**

$$
\pi(a \mid s) = \begin{cases} 1 - \epsilon + \dfrac{\epsilon}{m}, & \text{nếu } a = \arg\max_{a'} Q(s, a') \\[8pt] \dfrac{\epsilon}{m}, & \text{nếu } a \ne \arg\max_{a'} Q(s, a') \end{cases}
$$

* **Điều kiện GLIE:**
  1. Mọi cặp $(s, a)$ đều được khám phá vô hạn lần: $\lim_{k \to \infty} N_k(s, a) = \infty$.
  2. Chính sách dần hội tụ về tham lam tuyệt đối: $\lim_{k \to \infty} \pi_k(a \mid s) = \mathbf{1}(a = \arg\max_{a'} Q_k(s, a'))$.
  * Để đạt GLIE, ta giảm dần tỷ lệ khám phá theo số tập: $\epsilon_k = \dfrac{1}{k}$.

#### c) Công thức cập nhật tăng dần (Incremental Mean Update):
Sau mỗi tập trải nghiệm $S_1, A_1, R_2, \dots, S_T$, ta duyệt và cập nhật hàm $Q$ theo công thức trung bình động:

$$
N(S_t, A_t) \leftarrow N(S_t, A_t) + 1
$$

$$
Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \frac{1}{N(S_t, A_t)} \Big( G_t - Q(S_t, A_t) \Big)
$$

Trong đó sai số $(G_t - Q(S_t, A_t))$ đóng vai trò điều chỉnh giá trị ước lượng dần tiến về giá trị kỳ vọng thực tế.

#### d) Mã giả Python:
```python
import random

def glie_monte_carlo_control(env, num_episodes, gamma=0.99):
    """Thuật toán GLIE Monte Carlo Control giải bài toán điều khiển Model-Free."""
    # Khởi tạo bảng Q và bộ đếm N
    Q = {(s, a): 0.0 for s in env.states for a in env.actions}
    N = {(s, a): 0 for s in env.states for a in env.actions}
    
    for k in range(1, num_episodes + 1):
        # 1. Giảm epsilon theo quy tắc GLIE
        epsilon = 1.0 / k
        
        # 2. Sinh một tập trải nghiệm hoàn chỉnh (Episode)
        episode = []
        s = env.reset()
        done = False
        while not done:
            # Chọn hành động theo epsilon-greedy
            if random.random() < epsilon:
                a = random.choice(env.actions)
            else:
                a = max(env.actions, key=lambda act: Q[(s, act)])
                
            s_next, reward, done = env.step(a)
            episode.append((s, a, reward))
            s = s_next
            
        # 3. Tính Lợi tức G_t và cập nhật giá trị Q (First-visit MC)
        G = 0.0
        visited_pairs = set()
        
        # Duyệt ngược từ bước cuối cùng T-1 về bước đầu tiên
        for s_t, a_t, r_next in reversed(episode):
            G = r_next + gamma * G
            
            # Kiểm tra First-visit: chỉ cập nhật lần đầu tiên gặp cặp (s, a)
            if (s_t, a_t) not in visited_pairs:
                visited_pairs.add((s_t, a_t))
                N[(s_t, a_t)] += 1
                # Cập nhật tăng dần
                Q[(s_t, a_t)] += (1.0 / N[(s_t, a_t)]) * (G - Q[(s_t, a_t)])
                
    return Q
```

---

## 5. Tài Liệu Bài Giảng Đính Kèm

Toàn bộ slide gốc trong khóa học của GS. David Silver được lưu trữ cùng thư mục:
* `intro_rl.pdf` — *Lecture 1: Introduction to Reinforcement Learning*
* `lecture-2-mdp.pdf` — *Lecture 2: Markov Decision Processes*
* `lecture-3-planning-by-dynamic-programming-.pdf` — *Lecture 3: Planning by Dynamic Programming*
* `lecture-4-model-free-prediction-.pdf` — *Lecture 4: Model-Free Prediction*
* `lecture-5-model-free-control-.pdf` — *Lecture 5: Model-Free Control*
