# Multi-Agent Pacman — MAPPO 기반 Hide & Seek 강화학습 프로젝트

> **AIX Deep Learning 수업 프로젝트**  
> 멀티 에이전트 강화학습(MARL)으로 숨바꼭질(Hide & Seek) 행동을 학습시키는 실험

---

## 목차

1. [프로젝트 개요](#1-프로젝트-개요)
2. [강화학습 배경 지식](#2-강화학습-배경-지식)
3. [MAPPO 알고리즘 설명](#3-mappo-알고리즘-설명)
4. [게임 환경 설계](#4-게임-환경-설계)
5. [관측 구조 및 보상 설계](#5-관측-구조-및-보상-설계)
6. [모델 아키텍처](#6-모델-아키텍처)
7. [학습 파이프라인](#7-학습-파이프라인)
8. [실험 단계별 진행 과정](#8-실험-단계별-진행-과정)
9. [실행 방법](#9-실행-방법)
10. [프로젝트 구조](#10-프로젝트-구조)
11. [한계점 및 향후 과제](#11-한계점-및-향후-과제)
12. [기술 스택](#12-기술-스택)

---

## 1. 프로젝트 개요

### 프로젝트 동기

본 프로젝트는 openai의 Emergent Tool Use From Multi-Agent Autocurricula 논문에서 영감을 받아 3차원이 아닌 단순한 2차원의 pacman 게임 기반에서도 단순한 보상조건에서 협력 전략이 생성되는지 확인하는 것을 목표로 한다.

논문 https://arxiv.org/pdf/1909.07528

사이트 https://openai.com/index/emergent-tool-use/

논문의 경우 시야기반 보상 하나만으로 추격, 회피, 집단전략 등 다양한 모습을 확인할 수 있었으며 본 프로젝트에서도 같은 결과를 기대하며 진행했다.


### 프로젝트 설명

11×11 격자 맵에서 **1마리의 Pacman(도망자)** 과 **2마리의 Ghost(추격자)** 가 서로 상반된 목표를 가지고 행동 전략을 스스로 학습하는 강화학습 시뮬레이션이다.

openai의 논문과 같은 방식으로 보상은 한가지 시야에 들었는지의 유무로 진행했다. 따라서 두 에이전트의 목표는 다음과 같다고 할 수 있다.
- **Ghost의 목표**: Pacman을 자신의 시야 안에 포착하기
- **Pacman의 목표**: Ghost의 시야에 들키지 않고 최대한 오래 생존하기

사람이 규칙을 직접 넣는 대신, 에이전트들이 수천 번의 게임을 반복하면서 **스스로** 데이터를 형성해 추격·포위·도주 전략을 발견하도록 학습시켰다.

이 문제는 단순한 1vs1 게임이 아니다. Ghost 2마리가 **서로 협력** 해야 Pacman을 효과적으로 포위할 수 있고, Pacman은 **두 적을 동시에** 의식하며 도주 경로를 계획해야 한다. 이처럼 여러 에이전트가 상호작용하는 환경을 **Multi-Agent Reinforcement Learning(MARL)** 이라 부른다.

Ghost:      G         👻, 👾          

Pacman:     P          🟡

벽:         #          🟦

빈공간:      .           .

```
┌──────────────────────┐
│  G . . . . . . . . . │  G = Ghost (추격자, 2마리)
│  . # # . . . # # . . │  P = Pacman (도망자, 1마리)
│  . # . . . . . # . . │  # = 벽
│  . . . . # . . . . . │  . = 빈 공간
│  . . . # # # . . . . │
│  . . . # . # . . . . │
│  . . . . # . . . . . │
│  . . . . . . . . . . │
│  . # . . . . . . # . │
│  . # # . . . # # . . │
│  . . . . P . . . . . │
└──────────────────────┘
     11×11 격자 환경 예시
```

---

## 2. 강화학습 배경 지식

강화학습(Reinforcement Learning, RL)은 에이전트가 환경과 상호작용하면서 **보상(reward)** 을 최대화하는 방향으로 행동 정책(policy)을 학습하는 머신러닝 패러다임이다.

### 핵심 개념

| 용어 | 설명 |
|---|---|
| **에이전트(Agent)** | 행동을 결정하는 주체 (Ghost, Pacman) |
| **환경(Environment)** | 에이전트가 상호작용하는 세계 (11×11 격자 맵) |
| **상태(State)** | 현재 환경의 정보 |
| **관측(Observation)** | 에이전트가 실제로 볼 수 있는 정보 (부분 관측) |
| **행동(Action)** | 에이전트가 취하는 선택 (상·하·좌·우·정지) |
| **보상(Reward)** | 행동의 결과로 받는 피드백 신호 |
| **정책(Policy)** | 관측 → 행동을 결정하는 함수 (신경망으로 표현) |

### 부분 관측 문제 (Partial Observability)

현실적인 설정을 위해 각 에이전트는 **전체 맵을 볼 수 없다**. 자신 주변에 Ray(레이더)를 쏘아 벽과 다른 에이전트까지의 거리 정보만 알 수 있다. 이를 **부분 관측 마르코프 결정 과정(POMDP)** 이라 한다.

```
             ↑
           ray 0
       ray 7 ↑ ray 1
  ray 6 ← [에이전트] → ray 2
       ray 5 ↓ ray 3
           ray 4
             ↓
  
  8방향 레이더로 주변 정보 탐지
```

---

## 3. MAPPO 알고리즘 설명

### PPO란?

**Proximal Policy Optimization(PPO)** 은 현재 정책과 업데이트된 정책의 차이를 clip하여 너무 급격한 정책 변화를 막는 안정적인 강화학습 알고리즘이다. OpenAI에서 개발하여 다양한 분야에서 표준처럼 사용된다.

```
PPO 목적함수:
L_CLIP = E[ min( r_t(θ) × A_t,  clip(r_t(θ), 1-ε, 1+ε) × A_t ) ]

r_t(θ) = π_θ(a|s) / π_θ_old(a|s)  (확률 비율)
A_t = GAE로 계산된 Advantage
ε = 0.2  (clip 범위)
```

### MAPPO란?

**Multi-Agent PPO(MAPPO)** 는 PPO를 멀티 에이전트 환경에 확장한 알고리즘이다. 핵심 아이디어는 **Centralized Training, Decentralized Execution(CTDE)** 이다.

```

                        CTDE 구조                            
                                                             
  학습 시 (Centralized Training)                             
  ┌──────────────┐      전역 상태 (Global State)             
  │    Critic    │ ◄── ─────────────────────────────────    
  │  (가치 추정)  │      Ghost1, Ghost2, Pacman 위치 전부     
  └──────────────┘                                           

  실행 시 (Decentralized Execution)                          
  ┌──────────────┐      로컬 관측 (Local Obs)만              
  │    Actor     │ ◄── ─────────────────────────────────    
  │  (행동 결정)  │      자기 레이더 정보만                   
  └──────────────┘                                           
```

이 구조 덕분에 학습할 때는 전체 정보를 활용하여 더 정확한 가치 추정이 가능하고, 실제 실행 시에는 각 에이전트가 자신의 관측만으로 독립적으로 행동할 수 있다.

### GAE (Generalized Advantage Estimation)

보상 신호의 분산을 줄이면서도 편향을 최소화하기 위해 **GAE** 를 사용하여 Advantage를 계산한다.

```
A_t^GAE = Σ (γλ)^k × δ_{t+k}
δ_t = r_t + γ V(s_{t+1}) - V(s_t)  (TD 오류)

γ = 0.99  (할인율)
λ = 0.95  (GAE 파라미터)
```
---


## 4. 데이터셋 형성

강화학습을 채택하여 수많은 게임을 스스로 진행해가며 스스로 데이터를 얻게했다.
update_env=64을 채택해 update 1회에 같은 전략을 가진 환경 64개를 병렬로 돌려 데이터를 얻게했다. 



---


## 5. 게임 환경 설계

### 기본 설정

| 항목 | 값 |
|---|---|
| 맵 크기 | 11 × 11 격자 |
| 에이전트 | Pacman 1마리 + Ghost 2마리 |
| 행동 공간 | 5가지 (상, 하, 좌, 우, 정지) |
| 최대 스텝 수 | 에피소드당 400 스텝 |
| 충돌 | 벽 충돌 시 제자리 유지, 에이전트 간 충돌 허용 |

### 맵 종류

실험 난이도에 따라 3종류의 맵을 사용했다.

**Empty 맵** — 장애물 없는 완전히 빈 공간. Ghost가 직선으로 추격 가능.
```
┌───────────────────┐
│ . . . . . . . . . │
│ . . . . . . . . . │
│ . . . . . . . . . │
│ . . . . . . . . . │
│ . . . . . . . . . │
│ . . . . . . . . . │
│ . . . . . . . . . │
│ . . . . . . . . . │
│ . . . . . . . . . │
└───────────────────┘
```

**Medium 맵** — 중간 밀도의 벽. 포위 전략이 필요해짐.
```
┌───────────────────┐
│ . . . . . . . . . │
│ . . . # . . . . . │
│ . . . # . . # . . │
│ . # . # . . # . . │
│ . # . # . . # . . │
│ . # . # . . # . . │
│ . # . # . . # . . │
│ . # . # . . . . . │
│ . . . . . . . . . │
└───────────────────┘
```

**Hard 맵** — 복잡한 미로 구조. 경로 탐색 능력이 중요.
```
┌───────────────────┐
│ . . . . # . . . . │
│ . # # . # . # # . │
│ . . . . # . . . . │
│ # # # . . . # . # │
│ . . . # # . . . . │
│ . # . . . # # . . │
│ . . . # . . . # . │
│ . # # . # . . . . │
│ . . . . . . . . . │
└───────────────────┘
```

### 에피소드 종료 조건

- 최대 스텝 수(400) 도달 시 종료 (reward 계산은 단순히 시야에 얼만큰 들어있냐 단 한가지)

---

## 6. 관측 구조 및 보상 설계

### Ray 기반 부분 관측

각 에이전트는 **8방향 레이더**를 발사하여 주변 정보를 관측한다. Ray는 벽에 막히면 그 이상을 보지 못하므로 자연스러운 시야 차단 효과가 생긴다.

#### Pacman 관측 (Actor 입력: 40차원)

| Ray당 특징 (5가지) | 설명 |
|---|---|
| `wall_dist` | 해당 방향 벽까지의 거리 |
| `ghost_dist` | 가장 가까운 Ghost까지의 거리 |
| `ghost_dx` | Ghost 방향 x 성분 |
| `ghost_dy` | Ghost 방향 y 성분 |
| `ghost_mask` | Ghost가 시야 내에 있는지 여부 (0/1) |

총 8방향 × 5특징 = **40차원**

#### Ghost 관측 (Actor 입력: 72차원)

| Ray당 특징 (9가지) | 설명 |
|---|---|
| `wall_dist` | 해당 방향 벽까지의 거리 |
| `pacman_dist` | Pacman까지의 거리 |
| `pacman_dx` | Pacman 방향 x 성분 |
| `pacman_dy` | Pacman 방향 y 성분 |
| `pacman_mask` | Pacman이 시야 내에 있는지 여부 (0/1) |
| `ally_dist` | 아군 Ghost까지의 거리 |
| `ally_dx` | 아군 방향 x 성분 |
| `ally_dy` | 아군 방향 y 성분 |
| `ally_mask` | 아군이 시야 내에 있는지 여부 (0/1) |

총 8방향 × 9특징 = **72차원**

Ghost는 아군 위치도 관측하므로 협력 전략 학습이 가능하다.

#### Critic 입력 (전역 상태: 190차원)

Critic은 학습 시에만 사용되며 **모든 에이전트의 위치와 관측을 합친** 전역 상태를 입력받는다.

### 보상 설계의 변천

#### 초기 방식: Manhattan 거리 기반 보상 (실패)

```python
# Ghost 보상: Pacman과 가까울수록 양수 보상
ghost_reward = -manhattan_distance(ghost_pos, pacman_pos) * 0.01

# Pacman 보상: Ghost와 멀수록 양수 보상
pacman_reward = +manhattan_distance(ghost_pos, pacman_pos) * 0.01
```

**문제점**: 벽 너머에서도 거리가 계산되어 **벽을 무시하고 가까이 가는** 비정상 행동이 학습됨. 에이전트가 벽을 사이에 두고 서로 붙어있는 학습 노이즈 발생.

#### 개선 방식: 시야 기반 보상 (채택)

```python
# Ghost i가 Pacman을 자신의 시야(Ray) 안에서 발견한 경우
if pacman_in_ghost_view[i]:
    ghost_reward[i] = +1.0   # Ghost 보상
    pacman_reward   = -1.0   # Pacman 패널티

# 어떤 Ghost도 Pacman을 발견하지 못한 경우
else:
    모든 에이전트 reward = 0.0
```

**개선 효과**: 보상이 실제 시야에만 발생하므로 에이전트가 벽을 의식하는 현실적인 행동을 학습함.

---

## 7. 모델 아키텍처

### RecurrentActorCritic

```
입력 관측 (obs_dim)
    │
    ▼
Linear(obs_dim → 128) → ReLU
    │
    ▼
Linear(128 → 128) → ReLU
    │
    ▼
LSTM(input=128, hidden=128)     ← 이전 hidden state 유지 (부분 관측 보완)
    │
    ├──▶ policy_head → Linear(128 → 5) → Softmax  [Actor: 5개 행동 확률]
    │
    └──▶ value_head  → Linear(128 → 1)            [Critic: 상태 가치 추정]
```

#### LSTM을 사용하는 이유

부분 관측 환경에서는 현재 관측만으로 최적 행동을 결정하기 어렵다. 예를 들어 Ghost가 벽 뒤에 있어 보이지 않더라도, 직전 스텝에서 본 Ghost의 위치를 **기억**하고 있어야 더 나은 전략을 세울 수 있다. LSTM의 hidden state가 이 **암묵적 기억** 역할을 한다.

#### 에이전트별 독립 네트워크

```
Pacman Network: RecurrentActorCritic(obs_dim=40, critic_obs_dim=190, action_dim=5)
Ghost Network:  RecurrentActorCritic(obs_dim=72, critic_obs_dim=190, action_dim=5)
               (Ghost 2마리가 파라미터 공유)
```

Ghost 2마리는 하나의 네트워크를 **공유**한다. 이를 통해 학습 데이터를 2배로 활용하고, 두 Ghost가 동일한 전략을 구사하므로 협력이 자연스럽게 발생한다.

---

## 8. 학습 파이프라인

### 전체 흐름

```
### 학습 루프 (1500 업데이트 반복)

1. **`collect_rollout_parallel()`**
   - 64개의 환경을 병렬로 실행
   - 128 스텝 동안 `obs, action, reward, done` 수집
   - GAE로 Advantage & Return 계산

2. **`ppo_update()`**
   - `epoch=4` 반복
   - mini-batch로 나누어 그래디언트 계산
   - PPO Clip Loss + Value Loss + Entropy Bonus
   - Adam Optimizer로 파라미터 업데이트

3. **체크포인트 저장** (50 업데이트마다)
   - Google Drive에 `.pt` 파일 저장
```

### 하이퍼파라미터

| 파라미터 | 값 | 설명 |
|---|---|---|
| `num_envs` | 64 | 병렬 환경 수 |
| `total_updates` | 1500 | 총 PPO 업데이트 횟수 |
| `horizon` | 128 | 한 번에 수집하는 스텝 수 |
| `lr` | 3e-4 | Adam 학습률 |
| `hidden_dim` | 128 | LSTM 히든 차원 |
| `gamma` | 0.99 | 보상 할인율 |
| `gae_lambda` | 0.95 | GAE λ 파라미터 |
| `ppo_epochs` | 4 | PPO 업데이트 반복 횟수 |
| `clip_eps` | 0.2 | PPO Clip 범위 |
| `value_coef` | 0.5 | Value Loss 계수 |
| `entropy_coef` | 0.01 | Entropy Bonus 계수 |

### 병렬 환경의 효과

64개의 환경을 동시에 실행하면 1번의 업데이트에 `64 × 128 = 8,192` 스텝의 경험을 수집할 수 있다. 이는 단일 환경 대비 학습 효율을 크게 높이고 데이터의 다양성도 확보한다.

---

## 9. 실험 단계별 진행 과정

프로젝트는 총 4단계에 걸쳐 점진적으로 환경과 보상 구조를 발전시켰다.

### 단계 0: 빈 맵, 거리 기반 보상 (`0.open wall 1speed`)

| 항목 | 설정 |
|---|---|
| 맵 | Empty (벽 없음) |
| Pacman 속도 | 1 (Ghost와 동일) |
| 보상 방식 | Manhattan 거리 기반 |
| 학습 체크포인트 | `checkpoint_1speed_just_chase.pt` |

**목표**: 가장 단순한 환경에서 Ghost가 Pacman을 추격하는 기본 행동을 학습시킨다.

**관찰 결과**:
- 약 340 스텝 이후부터 Ghost가 Pacman 방향으로 이동하기 시작
- 빈 맵이라 직선 추격이 빠르게 학습됨
- 거리 기반 보상이라 실제 시야 개념이 없어 전략이 단순함

---

### 단계 1: 벽 추가 + Pacman 속도 증가 (`1.wall 2speed/`)

| 항목 | 설정 |
|---|---|
| 맵 | Medium (벽 있음) |
| Pacman 속도 | **2** (Ghost보다 빠름) |
| 보상 방식 | Manhattan 거리 기반 (유지) |
| 학습 체크포인트 | `checkpoint_2speed_just_chase.pt`, `checkpoint_2speed_final.pt` |

**목표**: Pacman에게 속도 우위를 주어 도주 전략을 유도하고, 벽 환경에서의 포위 전략을 학습시킨다.

**관찰 결과**:
- Ghost가 벽 근처에서 비정상적인 행동을 보임 (벽을 통과하려는 듯한 움직임)
- 거리 기반 보상이 벽을 고려하지 못해 **벽 너머 보상** 문제 확인
- Pacman이 속도 우위를 활용하여 Ghost를 따돌리는 행동이 일부 관찰됨
- 학습 노이즈 증가 → **보상 구조 변경 필요성 확인**

---

### 단계 2: 시야 기반 보상 도입 (`2.vision_based_reward`)

| 항목 | 설정 |
|---|---|
| 맵 | Medium |
| Pacman 속도 | 1 (동일하게 조정) |
| 보상 방식 | **시야 기반 보상** (핵심 변경) |
| Ghost 시야 범위 | 2 |
| 학습 체크포인트 | `checkpoint_update_900.pt`, `checkpoint_update_1250_seed451.pt` |

**목표**: 거리 기반 보상의 문제를 해결하고 실제 시야 개념을 도입한다.

**관찰 결과**:
- 800~1100 업데이트 구간에서 Ghost의 **협력 포위 전략** 안정적으로 발현
- 두 Ghost가 서로 다른 방향에서 접근하는 행동 관찰
- Pacman도 Ghost 시야에서 벗어나려는 도주 전략 학습
- 벽 근처 비정상 행동이 크게 감소

---

### 단계 3: 시야 범위 밸런싱 (`3.range_rebalancing`)

| 항목 | 설정 |
|---|---|
| 맵 | Medium + Empty (비교) |
| Pacman 속도 | 1 |
| 보상 방식 | 시야 기반 보상 (유지) |
| Pacman 시야 범위 | **4** (확대) |
| Ghost 시야 범위 | **3** (확대) |
| 학습 체크포인트 | `checkpoint_update_1000_empty_final.pt` |

**목표**: Pacman의 시야를 Ghost보다 넓게 설정하여 도주자가 더 멀리 볼 수 있는 현실적인 밸런스를 만든다.

**관찰 결과**:
- Pacman이 Ghost를 더 일찍 발견하고 회피 경로를 선택하는 전략 발전
- Ghost의 시야도 넓어져 협력 탐색 범위 증가
- Empty 맵에서 Ghost가 삼각 포위 전략을 구사하는 행동 관찰

---

### 단계 4: Action 확장 및 환경 상호작용 강화 ('4.place_wall')

| 항목 | 설정 |
|---|---|
| 추가 Action | PLACE_WALL |
| 벽 생성 위치 | 이동 전 위치 (이전 셀) |
| 벽 속성 | 지속 시간 + 쿨다운 보유 |
| 벽 효과 | 이동 및 시야 차단 |
| Observation | 설치 벽 위치 정보 포함하도록 확장 |

**목표**: Pacman이 단순 도주를 넘어 환경을 능동적으로 변화시켜 Ghost를 차단하는 전략을 학습시킨다.

**PLACE_WALL 동작**:

PLACE_WALL 선택 → 이동 방향 재샘플링 → 이전 위치에 임시 벽 생성 → 지속 시간 경과 후 소멸 (쿨다운 적용)

**관찰 결과**:
- 단순 도주를 넘어 환경을 활용하는 전략 학습 시작
- 벽을 이용한 경로 차단 전략 출현
- 벽을 이용한 도주 전략 출현

---

### 단계별 비교 요약

| 단계 | 맵 | Pacman 속도 | Ghost 시야 | 보상 방식 | 주요 성과 |
|---|---|---|---|---|---|
| 0 | Empty | 1 | - | Manhattan 거리 | 기본 추격 행동 학습 |
| 1 | Medium | **2** | - | Manhattan 거리 | 벽 환경 문제점 발견 |
| 2 | Medium | 1 | 2 | **시야 기반** | 협력 포위 전략 발현 |
| 3 | Medium | 1 | **3** | 시야 기반 | 시야 밸런스 + 도주 전략 고도화 |
| 4 | Medium | 1 | **3** | 시야 기반 | pacman에게 벽 생성 선택지 부여 |

---

## 10. 실행 방법

### 환경 설치

로컬 실행 시:
```bash
pip install torch numpy jupyter
```

Google Colab에서 실행 시 별도 설치 불필요.

### 학습 실행 (`pacmannew.ipynb`)

`pacmannew.ipynb`를 Colab에서 열고 아래 함수를 실행한다.

```python
env0, pacman_net, ghost_net, stats = train_marl(
    num_envs=64,           # 병렬 환경 수
    total_updates=1500,    # 총 PPO 업데이트 횟수
    horizon=128,           # 롤아웃 길이 (스텝)
    lr=3e-4,               # 학습률
    hidden_dim=128,        # LSTM 히든 차원
    seed=42,               # 재현성을 위한 랜덤 시드
    max_steps=400,         # 에피소드 최대 스텝
    pacman_view_range=4,   # Pacman 시야 거리
    ghost_view_range=3,    # Ghost 시야 거리
    map_name="medium",     # "empty" | "medium" | "hard"
    save_every=50,         # 체크포인트 저장 주기
    save_dir="/content/drive/MyDrive/tag_mappo_checkpoints",
    resume_path=None,      # 이어서 학습할 체크포인트 경로 (없으면 None)
)
```

이전 체크포인트에서 이어서 학습하려면:
```python
resume_path="/content/drive/MyDrive/tag_mappo_checkpoints/checkpoint_update_1000.pt"
```

### 추론 및 시각화 (`pacmannewplay.ipynb`)

```python
run_inference(
    checkpoint_path="/content/drive/MyDrive/tag_mappo_checkpoints/checkpoint_update_1000.pt",
    map_name="medium",
    seed=451,
    max_steps=400,
    pacman_view_range=4,
    ghost_view_range=3,
    stochastic=True,   # False이면 greedy(argmax) 정책 사용
    render=True,       # 텍스트 렌더링 출력 여부
    delay=0.1,         # 스텝 간 딜레이 (초)
)
```

### 학습 결과 평가

```python
evaluate(env0, pacman_net, ghost_net, episodes=3, render=True, device=device)
```

---

## 11. 프로젝트 구조

```
Aix-deeplr-hide-seek/
│
├── 0.open wall 1speed/                  # 0단계: 빈 맵, 속도 1
│   ├── pacmannew.ipynb                  # 학습 노트북
│   ├── pacmannewplay.ipynb              # 추론·시각화 노트북
│   └── checkpoint_1speed_just_chase.pt  # 추격 행동 학습 체크포인트
│
├── 1.wall 2speed/                       # 1단계: 벽 추가, Pacman 속도 2
│   ├── pacmannew.ipynb
│   ├── pacmannewplay.ipynb
│   ├── checkpoint_2speed_just_chase.pt  # 추격 단계 체크포인트
│   └── checkpoint_2speed_final.pt       # 최종 체크포인트
│
├── 2.vision_based_reward/               # 2단계: 시야 기반 보상 도입
│   ├── pacmannew.ipynb
│   ├── pacmannewplay.ipynb
│   ├── checkpoint_1speed_just_chase.pt
│   ├── checkpoint_update_900.pt         # 900 업데이트 체크포인트
│   └── checkpoint_update_1250_seed451.pt
│
├── 3.range_rebalancing/                 # 3단계: 시야 범위 밸런싱
│   ├── pacmannew.ipynb                  # 최신 학습 코드
│   ├── pacmannewplay-2.ipynb            # 최신 추론 코드
│   └── checkpoint_update_1000_empty_final.pt
│
│
├── 4.place_wall                         # 4단계: 벽 생성 및 최종 학습 환경 구현
│   ├── pacmannew.ipynb                  # 최신 학습 코드
│   ├── pacmannewplay.ipynb              # 최신 추론 코드
│   └── checkpoint_update_1000_empty_final.pt
│
│
└── README.md
```

> 각 디렉터리는 독립적인 실험 단계이다. 이전 단계의 체크포인트를 `resume_path`에 지정하면 이어서 학습할 수 있다.

### 체크포인트 파일 구조 (`.pt`)

```python
{
    "update": int,                           # 저장 시점 업데이트 번호
    "pacman_model_state_dict": dict,         # Pacman 네트워크 가중치
    "ghost_model_state_dict": dict,          # Ghost 네트워크 가중치
    "pacman_optimizer_state_dict": dict,     # Pacman Adam 옵티마이저 상태
    "ghost_optimizer_state_dict": dict,      # Ghost Adam 옵티마이저 상태
    "stats": {
        "pacman_return": [float, ...],       # 업데이트별 Pacman 누적 보상 기록
        "ghost_return": [float, ...],        # 업데이트별 Ghost 누적 보상 기록
    }
}
```

---

## 12. 향후 과제

현재 최종 환경으로 설정한 4.place_wall 환경에서 지속적으로 강화학습 기반 모델을 돌려보며 전략의 변화를 관측해보기

---

## 13. 기술 스택

| 분류 | 라이브러리 / 플랫폼 | 용도 |
|---|---|---|
| 딥러닝 | **PyTorch** | LSTM 기반 Actor-Critic 모델, PPO 업데이트 |
| 수치 연산 | **NumPy** | 환경 상태 계산, Ray 충돌 처리 |
| 실행 환경 | **Google Colab** | GPU 가속 학습 |
| 저장소 | **Google Drive** | 체크포인트 영구 저장 |
| 인터페이스 | **Jupyter Notebook** | 학습 및 추론 실험 |

---

## 참고 문헌

- Yu, C., et al. (2022). *The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games.* NeurIPS 2022.
- Schulman, J., et al. (2017). *Proximal Policy Optimization Algorithms.* arXiv:1707.06347.
- Lowe, R., et al. (2017). *Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments.* NeurIPS 2017.
- Baker, B., et al. (2019). *Emergent Tool Use From Multi-Agent Autocurricula.* OpenAI Blog.

---

*본 프로젝트는 한양대학교 AIX Deep Learning 수업의 팀 프로젝트로 진행되었습니다.*
