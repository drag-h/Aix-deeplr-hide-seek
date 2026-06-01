# Multi-Agent Pacman — MAPPO 기반 Hide & Seek

## 프로젝트 소개

1 Pacman vs 2 Ghosts 구도의 11×11 격자 환경에서 **MAPPO(Multi-Agent Proximal Policy Optimization)** 알고리즘을 이용해 부분 관측 기반 추격·도주 행동을 학습하는 강화학습(Reinforcement Learning) 프로젝트이다.

Ghost는 Pacman을 시야(ray-based radar) 안에 포착하면 보상을 받고, Pacman은 Ghost에게 들키지 않도록 도주 전략을 학습한다. Actor는 에이전트별 로컬 관측만 사용하고 Critic은 전역 상태를 사용하는 **Centralized Training, Decentralized Execution(CTDE)** 구조를 따른다.

총 4단계(빈 맵 → 벽 추가 → 보상 구조 개선 → 시야 밸런싱)에 걸쳐 진행되었으며, 각 단계의 체크포인트와 학습 노트북이 디렉터리별로 보존되어 있음.

---

## 핵심 기능

| 기능 | 설명 |
|---|---|
| 커스텀 MARL 환경 | 11×11 격자, 3종 맵(empty / medium / hard), 충돌 허용 |
| Ray 기반 부분 관측 | 8방향 radar, 에이전트마다 독립 시야 범위 설정 |
| MAPPO 학습 | Recurrent Actor-Critic (LSTM) + GAE + PPO clip |
| 병렬 환경 롤아웃 | 다수의 환경을 동시에 수집하여 학습 효율 향상 |
| 체크포인트 저장·재개 | 업데이트 번호 단위로 저장, 이어서 학습 가능 |
| 텍스트 렌더링 추론 | 저장된 체크포인트를 로드하여 에피소드를 시각화 |

---

## 전체 서비스 흐름

```
1. 환경 초기화
   └── 맵 선택(empty / medium / hard), 시야 범위 및 시드 설정

2. 학습 (pacmannew.ipynb)
   ├── 병렬 환경에서 Rollout 수집 (horizon=128 steps × N envs)
   ├── GAE로 Advantage 계산
   ├── PPO 업데이트 (Pacman 네트워크 / Ghost 네트워크 각각)
   └── 지정 주기마다 Google Drive에 체크포인트 저장

3. 추론 (pacmannewplay.ipynb)
   ├── 체크포인트 로드
   ├── 에피소드 스텝 실행 (stochastic 또는 greedy)
   └── 매 스텝 텍스트 렌더링으로 게임 진행 시각화
```

---

## 기술 스택

| 분류 | 라이브러리 / 플랫폼 |
|---|---|
| 딥러닝 프레임워크 | PyTorch |
| 수치 연산 | NumPy |
| 실행 환경 | Google Colab |
| 체크포인트 저장 | Google Drive |
| 인터페이스 | Jupyter Notebook |

---

## 프로젝트 구조

```
Aix-deep/
├── 0.open wall 1speed/          # 0단계: 빈 맵, Pacman 속도 1 로 설정
│   ├── pacmannew.ipynb          # 학습 노트북
│   ├── pacmannewplay.ipynb      # 추론/시각화 노트북
│   └── checkpoint_1speed_just_chase.pt
│
├── 1.wall 2speed/               # 1단계: 벽 추가 맟 Pacman 속도 증가 (1 -> 2)
│   ├── pacmannew.ipynb
│   ├── pacmannewplay.ipynb
│   ├── checkpoint_2speed_just_chase.pt
│   └── checkpoint_2speed_final.pt
│
├── 2./                          # 2단계: 시야 기반 보상, MAPPO 적용
│   ├── pacmannew.ipynb
│   ├── pacmannewplay.ipynb
│   ├── checkpoint_1speed_just_chase.pt
│   ├── checkpoint_update_900.pt
│   └── checkpoint_update_1250_seed451.pt
│
├── 3/                           # 3단계: 시야 밸런싱 (Pacman 10 / Ghost 2→3)
│   ├── pacmannew.ipynb          # 최신 학습 코드
│   ├── pacmannewplay-2.ipynb    # 최신 추론 코드
│   └── checkpoint_update_1000_empty_final.pt
│
└── README.md
```

> 각 디렉터리는 독립적인 실험 단계이고, 전 단계의 체크포인트를 `resume_path`에 지정하여 이어서 학습 가능함.

---

## 설치 방법

로컬에서 실행하려면 아래 명령어로 설치하기

```bash
pip install torch numpy
```

---

## 환경 변수 설정

별도의 `.env` 파일은 사용하지 않음. 체크포인트 저장 경로는 노트북 내 상수로 직접 지정한다.

```python
# pacmannew.ipynb 내 학습 설정
save_dir = "/content/drive/MyDrive/tag_mappo_checkpoints"
resume_path = "/content/drive/MyDrive/tag_mappo_checkpoints/checkpoint_update_1000.pt"  # 없으면 None

# pacmannewplay.ipynb 내 추론 설정
CHECKPOINT_PATH = "/content/drive/MyDrive/tag_mappo_checkpoints/checkpoint_update_1000.pt"
```

---

## 실행 방법

### 학습

`pacmannew.ipynb`를 Colab에서 열고 아래 셀을 실행한다.

```python
env0, pacman_net, ghost_net, stats = train_marl(
    num_envs=64,          # 병렬 환경 수
    total_updates=1500,   # 총 PPO 업데이트 횟수
    horizon=128,          # 롤아웃 길이
    lr=3e-4,
    hidden_dim=128,
    seed=42,
    max_steps=400,
    pacman_view_range=4,  # Pacman 시야 거리
    ghost_view_range=3,   # Ghost 시야 거리
    map_name="medium",    # "empty" | "medium" | "hard"
    save_every=50,
    save_dir="/content/drive/MyDrive/tag_mappo_checkpoints",
    resume_path=None,     # 이어서 학습할 경우 체크포인트 경로 지정
)
```

### 추론 및 시각화

`pacmannewplay.ipynb`(또는 `pacmannewplay-2.ipynb`)를 열고 실행한다.

```python
run_inference(
    checkpoint_path="/content/drive/MyDrive/tag_mappo_checkpoints/checkpoint_update_1000.pt",
    map_name="medium",
    seed=451,
    max_steps=400,
    pacman_view_range=4,
    ghost_view_range=3,
    stochastic=True,   # False 이면 greedy(argmax) 정책 사용
    render=True,
    delay=0.1,
)
```

---

## 주요 코드 흐름

### 환경 (`TagMARLEnv`)

- `reset()` — 맵을 초기화하고 Pacman/Ghost 위치를 랜덤 배치 (최소 거리 `min_start_dist` 보장)
- `step(pacman_action, ghost_actions)` — 에이전트 이동 → 시야 기반 보상 계산 → 종료 조건 확인
- `get_obs()` — Actor 로컬 관측 + Critic 전역 상태 반환

### 관측 구조

| 에이전트 | Actor 관측 차원 | Critic 관측 차원 |
|---|---|---|
| Pacman | 8 rays × 5 = **40** | **190** (전역 상태) |
| Ghost (각각) | 8 rays × 9 = **72** | **190** (전역 상태) |

Ghost의 ray 특징: `[wall_dist, pacman_dist, pacman_dx, pacman_dy, pacman_mask, ally_dist, ally_dx, ally_dy, ally_mask]`

### Manhattan 거리 기반 보상의 오류

초기에는 Manhattan distance 기반 보상을 사용하였으나, 벽 뒤에서도 reward가 발생하는 문제가 있었고 이로 인해 학습 noise가 발생하였다.
이를 해결하기 위해 거리 기반 보상을 제거하고 시야 기반 보상으로 변경하였다.

### 시야 기반 보상 구조

```
Ghost i가 Pacman을 시야 내에서 발견한 경우:
  Ghost i reward = +1.0
  Pacman reward  = -1.0  (어느 Ghost라도 발견 시)

발견하지 못한 경우:
  모든 에이전트 reward = 0.0
```

### 모델 (`RecurrentActorCritic`)

```
Actor:  Linear(obs_dim → 128) → ReLU → Linear → ReLU → LSTM(128) → policy_head(5)
Critic: Linear(obs_dim → 128) → ReLU → Linear → ReLU → LSTM(128) → value_head(1)
```

Pacman과 Ghost 각각 독립적인 네트워크를 가진다.

### 학습 루프

```
collect_rollout_parallel()
  ├── horizon 스텝 동안 병렬 환경에서 경험 수집
  └── GAE(γ=0.99, λ=0.95)로 Advantage 계산

ppo_update()
  ├── epochs=4 반복
  ├── PPO clip loss (ε=0.2)
  ├── Value function loss (계수 0.5)
  └── Entropy bonus (계수 0.01)
```

---

## 데이터 구조

### 체크포인트 파일 (`.pt`)

```python
{
    "update": int,                          # 저장 시점의 업데이트 번호
    "pacman_model_state_dict": dict,        # Pacman 네트워크 가중치
    "ghost_model_state_dict": dict,         # Ghost 네트워크 가중치
    "pacman_optimizer_state_dict": dict,    # Pacman Adam optimizer 상태
    "ghost_optimizer_state_dict": dict,     # Ghost Adam optimizer 상태
    "stats": {
        "pacman_return": [float, ...],      # 업데이트별 Pacman 누적 보상
        "ghost_return": [float, ...],       # 업데이트별 Ghost 누적 보상
    }
}
```

### 맵 형식

`#` = 벽, `0` = 빈 공간 (11×11 문자열 리스트)

---

## 테스트 및 검증

별도의 테스트 프레임워크는 구성되어 있지 않다. 학습 후 다음 방법으로 결과를 검증할 수 있다.

```python
# 학습 노트북에서 evaluate() 호출
evaluate(env0, pacman_net, ghost_net, episodes=3, render=True, device=device)
```

---


## 단계 별 요약

| 단계 | 맵 | Pacman 속도 | Ghost 속도 | 보상 방식 | 주요 관찰 |
|---|---|---|---|---|---|
| 0 | empty | 1 | 1 | Manhattan 거리 | step 340 부근에서 추격 행동 시작 |
| 1 | medium (벽) | 2 | 1 | Manhattan 거리 | 벽 근처 비정상 행동, 학습 noise 발생 |
| 2 | medium | 1 | 1 | **시야 기반** | step 800~1100 안정적 협력·포위 전략 |
| 3 | medium | 1 | 1 | 시야 기반 | 시야 범위 조정(Pacman 4 / Ghost 3)으로 밸런스 개선 |

---

## 한계 

- 시각화가 텍스트 렌더링 수준 — Matplotlib/pygame 기반 GUI 시각화 미구현
- Ghost 수를 2개로 고정 — 가변 에이전트 수 지원 미구현
- 학습 곡선(reward 그래프) 자동 저장 미구현
