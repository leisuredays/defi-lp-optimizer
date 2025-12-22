Uniswap v3 백서에서 수수료 관련 공식들을 정리해 드릴게요.

## 기본 변수

- **γ (gamma)**: 스왑 수수료율 (0.05%, 0.30%, 1% 중 선택)
- **φ (phi)**: 프로토콜 수수료 비율 (0, 1/4, 1/5, ... 1/10)
- **L**: 유동성
- **f_g**: feeGrowthGlobal (단위 유동성당 누적 수수료)
- **f_o**: feeGrowthOutside (틱 외부 누적 수수료)

## 스왑 시 수수료 계산 (Section 6.2.3)

토큰1을 y_in 만큼 입금할 때:

**LP에게 귀속되는 수수료:**
$$\Delta f_{g,1} = y_{in} \cdot \gamma \cdot (1 - \phi)$$

**프로토콜에 귀속되는 수수료:**
$$\Delta f_{p,1} = y_{in} \cdot \gamma \cdot \phi$$

**실제 스왑에 사용되는 금액:**
$$\Delta y = y_{in} \cdot (1 - \gamma)$$

## 범위별 수수료 계산 (Section 6.3)

**틱 i 위에서 발생한 수수료:**
$$f_a(i) = \begin{cases} f_g - f_o(i) & \text{if } i_c \geq i \\ f_o(i) & \text{if } i_c < i \end{cases}$$

**틱 i 아래에서 발생한 수수료:**
$$f_b(i) = \begin{cases} f_o(i) & \text{if } i_c \geq i \\ f_g - f_o(i) & \text{if } i_c < i \end{cases}$$

**특정 범위 [i_l, i_u] 내 누적 수수료:**
$$f_r = f_g - f_b(i_l) - f_a(i_u)$$

## 포지션별 미수령 수수료 (Section 6.4.1)

유동성 l을 가진 포지션의 미수령 수수료:
$$f_u = l \cdot (f_r(t_1) - f_r(t_0))$$

여기서 f_r(t_0)은 포지션에 저장된 feeGrowthInsideLast 값이고, f_r(t_1)은 현재 계산된 범위 내 수수료.

## 틱 크로싱 시 업데이트

틱을 지날 때 f_o 값 반전:
$$f_o := f_g - f_o$$

백서 Section 6을 기반으로 정리할게요.

## 1. Global State (Section 6.2, Table 1)

| 변수 | 표기 | 용도 |
|------|------|------|
| liquidity | L | 현재 가격에서 활성화된 총 유동성 |
| sqrtPriceX96 | √P | 현재 √가격 |
| tick | i_c | 현재 틱 인덱스 |
| feeGrowthGlobal0X128 | f_g,0 | token0 단위유동성당 누적수수료 |
| feeGrowthGlobal1X128 | f_g,1 | token1 단위유동성당 누적수수료 |

## 2. Tick-Indexed State (Section 6.3, Table 2)

내 범위의 하한 틱(i_l)과 상한 틱(i_u) 각각에 대해:

| 변수 | 표기 | 용도 |
|------|------|------|
| liquidityNet | ΔL | 틱 크로싱 시 유동성 변화량 |
| liquidityGross | L_g | 해당 틱을 경계로 하는 총 유동성 |
| feeGrowthOutside0X128 | f_o,0 | 틱 외부 누적수수료 (token0) |
| feeGrowthOutside1X128 | f_o,1 | 틱 외부 누적수수료 (token1) |

## 3. Position-Indexed State (Section 6.4, Table 3)

| 변수 | 표기 | 용도 |
|------|------|------|
| liquidity | l | 내 포지션의 유동성 |
| feeGrowthInside0LastX128 | f_r,0(t_0) | 마지막 업데이트 시점의 범위 내 수수료 |
| feeGrowthInside1LastX128 | f_r,1(t_0) | 마지막 업데이트 시점의 범위 내 수수료 |

## 4. 계산 공식 (Section 6.3, 6.4)

**Step 1: 틱 위/아래 수수료 계산**

$$f_a(i) = \begin{cases} f_g - f_o(i) & i_c \geq i \\ f_o(i) & i_c < i \end{cases}$$

$$f_b(i) = \begin{cases} f_o(i) & i_c \geq i \\ f_g - f_o(i) & i_c < i \end{cases}$$

**Step 2: 범위 내 수수료 성장률**

$$f_r = f_g - f_b(i_l) - f_a(i_u)$$

**Step 3: 미수령 수수료 (Section 6.4.1)**

$$f_u = l \times (f_r(t_1) - f_r(t_0))$$

## 5. 시점별 필요 데이터

| 시점 | 필요 데이터 |
|------|-------------|
| 포지션 생성 시 (t_0) | f_g, f_o(i_l), f_o(i_u), i_c → f_r(t_0) 저장 |
| 수수료 계산 시 (t_1) | f_g, f_o(i_l), f_o(i_u), i_c → f_r(t_1) 계산 |

## 6. 과거 특정 시점 분석 시

백서에는 명시되지 않았지만, 과거 시점 분석을 위해서는 해당 블록에서의:

```
Global:     f_g,0, f_g,1, i_c
Tick i_l:   f_o,0(i_l), f_o,1(i_l)
Tick i_u:   f_o,0(i_u), f_o,1(i_u)
Position:   l, f_r,0(t_0), f_r,1(t_0)
```

이 모든 값을 archive node에서 조회해야 해요.

## 요약

```
최소 필요 데이터: 9개 변수
- Global: 3개 (f_g,0, f_g,1, i_c)
- Lower tick: 2개 (f_o,0, f_o,1)
- Upper tick: 2개 (f_o,0, f_o,1)  
- Position: 2개 (l, f_r(t_0))
```

### V2 기본 IL 공식

$$IL(k) = \frac{2\sqrt{k}}{1+k} - 1$$

여기서 k = P₁/P₀ (가격 변화 비율)

### V3 Concentrated Liquidity IL 공식

$$IL_{v3}(k) = IL(k) \cdot \frac{1}{1 - \frac{1}{\sqrt{r}}}$$

여기서 r = P_b/P_a (상한가/하한가 비율)

## 유도 과정 (이미지 기반)

**1. Virtual reserves 관계:**
$$\sqrt{x_{virtual} \cdot y_{virtual}} = L^2$$

**2. 포지션 가치 계산:**
$$V_0 = 2L\sqrt{P} - L\left(\sqrt{P_a} + \frac{P}{\sqrt{P_b}}\right)$$

$$V_1 = 2L\sqrt{Pk} - L\left(\sqrt{P_a} + \frac{Pk}{\sqrt{P_b}}\right)$$

**3. Hold 가치:**
$$V_{hold} = \frac{2L\sqrt{P} - L\sqrt{P_a}(1+k)}{...}$$

## 의미

| 범위 (r) | IL 배수 |
|----------|---------|
| r = 4 (±100%) | 2x |
| r = 1.21 (±10%) | ~10x |
| r = 1.1 (±5%) | ~20x |

좁은 범위일수록 IL이 기하급수적으로 증폭돼요. 이게 백서에 없는 이유는, 백서는 구현에 집중하고 경제적 분석은 다루지 않기 때문이에요.