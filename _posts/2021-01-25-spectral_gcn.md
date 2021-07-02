---
title: "Spectral GCN 은... 사드세요"
categories: 
    - Research
toc: true
---

**Graph Convolution Network**은 크게 spectral과 spatial 두 가지 관점으로 나뉜다. 여기서는 **spectral**에 대해서 정리한다. 

*긴 글에 앞서 spectral GCN을 공부하는데 많은 조언을 해주신 같은 DSBA 연구실 [윤훈상 선배님](https://hoonst.github.io)께 무한한 감사의 말을 전합니다.*

# Spectrum의 의미

**Spectrum**의 사전적 의미로 라틴어에서는 **`image`** 또는 **`apparition`** 이라는 의미가 있다. **그림 (1)**과 같이 이 개념은 Isaac Newton에 의해 처음 소개되었다. Spectrum 이라는 개념을 가장 쉽게 볼 수 있는 예시는 바로 전자기파의 주파수(frequencies of electromagnetic radiation)가 있다. 또한, 다른 방면으로 spectrum은 양자역학에서 관찰가능량(observables)에 대응하는 에르미트 연산자(Hermitian operator)의 eigenvalues로 측정한 양자들의 집합을 말하기도 한다. 이와 같이 다양한 분야에서 spectrum이라는 말이 사용되고 있다. 현재는 이미지나 음성 데이터에서 spectrum representation을 얻기위해 푸리에 해석(Fourier analysis)를 활용하는 등 spectrum은 머신러닝 분야에서 다양한 접근으로 문제를 해결하기 위해 사용되는 개념이다[^3].

<p align='center'>
    <img src='https://user-images.githubusercontent.com/37654013/105569657-c0686500-5d86-11eb-9ba8-d04e5589d29a.png'><br>그림 1. Spectrum을 실험하는 Isaac Newton을 그린 그림
</p>

# Spectral GCN에 필요한 두 가지 개념

Spectral 관점에서 보기 위해서는 크게 **두 가지 개념**이 필요하다.

1. **Laplacian operator**
2. **Fourier Transform**

**의식의 흐름대로 이해하는 Spectral GCN**

두 가지 개념에 대해 알아보기 전에 아래와 같은 흐름을 먼저 이해하고 개념을 보는게 좋을 거 같다. 

1. Spectral GCN은 방금 말한 것처럼 signal processing (SP)에서 사용하는 Fourier transform을 활용하여 convolution theory와 연결하여 사용한다. 
2. Fourier transform을 위해 사용하는 것이 Laplacian operator이다.
3. Graph Laplacian operator는 두 노드간의 거리를 나타내는 방법이다.  
4. Laplacian matrix에 eigen decomposition을 사용하여 eigenvalue와 eigenvector를 활용한다.
5. Fourier transform에 사용하는 eigenvector를 Laplacian matrix에서 구한 eigenvector로 사용한다.
6. Convolution은 Fourier transform의 곱과 같으므로 convolution 연산을 Fourier transform을 활용하여 계산할 수 있다.

만약 이해가 안된다면 아래 개념을 보고 다시 보는 것을 추천한다.

## Laplacian Operator

**Laplacian operator**은 *divergence of gradient*라는 관점에서 벡터장(vector field)이 균일하지 않은 정도를 파악할 수 있는 방법을 나타낸다. 예를 들어 이미지에서도 Laplacian filter를 사용하게 되면 이미지 내에서 어떤 물체의 경계선 같은 곳이 있을 때 필터의 결과값이 균일하지 않은 곳은 Laplacian 값이 크게 나오게 된다. 그 결과 Laplacian filter를 씌운 이미지는 object의 edge가 나오게 된다.

'[Laplacian Operator](https://micropilot.tistory.com/2970)'는 Laplacian operator를 시각화를 통해 직관적으로 이해하기 쉽게 작성한 글이다. 보는 것을 추천한다.

위 내용에 따라 graph에서 Laplacian operator를 적용하면 각 노드와 이웃 노드 간의 차이를 나타내게 된다. Graph에서 **Laplacian matrix ($$\textbf{L}$$)**는 **degree matrix ($$\textbf{D}$$) - adjacency matrix ($$\textbf{A}$$)** 이다. 예를 들어 **그림 (2)**와 같은 그래프가 있다고 가정할 때 각 노드에 대한 signal에 Laplacian operator를 적용해보면 노드 간의 차이를 나타내게 된다. 

<p align='center'>
    <img width='300' src='https://user-images.githubusercontent.com/37654013/105624172-ed8a4580-5e62-11eb-9165-ebcc94eceefc.png'><br>그림 2. 3개의 노드가 서로 모두 연결된 그래프 예시
</p>


$$\begin{align*}
\textbf{L}\textbf{x} &=  \begin{bmatrix}
                         2 & -1 & -1 \\ -1 & 2 & -1 \\ -1 & -1 & 2
                         \end{bmatrix}
                         \begin{bmatrix}
                         x_1 \\ x_2 \\ x_3
                         \end{bmatrix} \\ 
                     &=  \begin{bmatrix}
                         2x_1 - x_2 - x_3 \\
                         -x_1 + 2x_2 - x_3 \\
                         -x_1 - x_2 + 2x_3
                         \end{bmatrix}  \\ 
                     &=  \begin{bmatrix}
                         (x_1 - x_2) + (x_1 - x_3) \\ 
                         (x_2 - x_1) + (x_2 - x_3) \\
                         (x_3 - x_1) + (x_3 - x_2) 
                         \end{bmatrix} \\
                     &=  \begin{bmatrix}
                         \sum (x_1 - x_i) \\ 
                         \sum (x_2 - x_i) \\
                         \sum (x_3 - x_i)
                         \end{bmatrix}
\end{align*}$$


Spectral GCN에서는 $$\textbf{L}$$를 계산하고 이를 eigen decomposition을 수행하여 eigenvalue와 eigenvector를 계산하는데 이는 Fourier analysis의 basis와 공유하기 위함이다. 

Laplacian을 eigen decomposition하면 **식 (1)**과 같이 나타낼 수 있다.

$$\textbf{L} = \textbf{U}\Lambda \textbf{U}^T \tag{1}$$

여기서 $$\textbf{U} \in \textbf{R}^{n \times n}$$는 eigenvector matrix 를 말하고 $$\Lambda$$는 대각 원소가 오름차순으로 정렬된 eigenvalue를 가진 $$n \times n$$ 대각행렬이다. Laplacian의 eigenvalue는 Fourier transform의 spectrum 또는 frequency를 나타낸다. Graph에서 spectrum은 노드 간의 차이를 나타내기 때문에 $$\lambda_{1} < \lambda_{2} < \dots < \lambda_{n}$$ 과 같이 spectrum의 값이 작도록 오름차순으로 정렬하여 사용한다. 여기서 $$n$$은 노드 수를 말한다. 

## Fourier Transform

만약 Fourier transform 개념이 처음이라면 갓크 프로그래머님의 '[Fourier Transform(푸리에 변환)의 이해와 활용](https://darkpgmr.tistory.com/171)'을 읽기를 추천한다. 해당 글에서 얻을 수 있는 것은 다음과 같다.

1. Fourier transform의 직관적 이해와 수식적 이해
2. Fourier spectrum과 phase 
3. 이미지를 통한 활용 예시

**Fourier transform**은 어떤 신호를 분해하기 위해 사용된다. 음성이나 전파나 어떤 신호든 Fourier transform을 거치면 $$\sin$$과 $$\cos$$ 주기함수들의 합으로 분해가 가능하다. 

<p align='center'>
    <img src='https://user-images.githubusercontent.com/37654013/105569673-f3aaf400-5d86-11eb-9129-e94c22637c7b.png'><br>그림 3. 푸리에 변환
</p>

직관적으로 보자면 **그림 (3)**과 같이 어떤 신호(빨간색)이 들어 왔을 때 Fourier transform을 통해 여러 개의 신호(파란색)로 분리해서 보자는 말이다. Fourier transform은 이때 왼쪽 축인 time domain(시간축에 대해 정의된 신호인 경우)에서 오른쪽 축인 frequency domain으로 변환하는 것을 말한다. 

$$\mathscr{F}^{-1}\{f\}(x) = \int_\mathbb{R} f(v)e^{2j\pi vx}dv \tag{2}$$

$$\mathscr{F}\{f\}(v) = \int_\mathbb{R} f(x)e^{-2j\pi vx}dx \tag{3}$$

**식 (3)**을 Fourier transform이라 부르고 **식 (2)**를 inverse Fourier transform 이라고 부른다.  여기서 $$j$$는 허수단위 $$\sqrt{-1}$$ , $$f(x)$$는 원본 입력 신호, $$e^{j2 \pi vx}$$은 frequency가 $$v$$인 주기함수 성분, 그리고 $$f(v)$$는 해당 주기함수 성분의 계수(coefficient) 또는 강도(amplitude)를 말한다. 여기서 $$e^{j2 \pi vx}$$는 [오일러 공식](https://ko.wikipedia.org/wiki/오일러_공식)을 통해 **식 (4)**와 같이 $$\cos$$과 $$\sin$$으로 정의할 수 있다.

$$e^{2 j \pi x v} = \cos{2\pi vx} + j\sin{2 \pi vx} \tag{4}$$

## Convolution Theorem

> *Convolution in spatial domain is equivalent to multiplication in Fourier domain*

이런 말이 있다. Convolution 연산은 Fourier transform으로 대체 가능하다는 말이다. 이제 **식 (6)**과 같이 convolution의 정의에서부터 Fourier transform이 어떻게 적용되는지 확인해보자

$$H(z) = \int_\mathbb{R}f(x)g(z - x)dx \tag{5}$$ 

$$\begin{align*}
\mathscr{F}\{f*g\}(v) &= \mathscr{F}\{h\}(v) \\
&= \int_\mathbb{R} h(z)e^{-2j \pi zv}dz \\
&= \int_\mathbb{R} \int_\mathbb{R} f(x)g(z-x)e^{-2j \pi vz}dxdz \\ 
&= \int_\mathbb{R} f(x)(\int_\mathbb{R} g(z-x)e^{-2j \pi vz}dz)dx, \ let \ u=z-x \\
&= \int_\mathbb{R} f(x)(\int_\mathbb{R} g(u)e^{-2j \pi v(u+x)}du)dx \\
&= \int_\mathbb{R} f(x)e^{-2j \pi vx}dx \int_\mathbb{R} g(u)e^{-2k \pi vu}du
\tag{6}
\end{align*}$$

위 과정을 거쳐 결과적으로 convolution의 두 함수간 곱은 두 함수의 Fourier transform 곱과 같다.

$$f*g = \mathscr{F}^{-1}\{\mathscr{F}\{f\}\cdot\mathscr{F}\{g\}\} \tag{7}$$

Graph에서 적용한 Fourier transform은 $$\mathscr{F}(\textbf{x})=\textbf{U}^T\textbf{x} = \hat{\textbf{x}}$$ 로 정의된다. 반대로 inverse Fourier transform은 $$\mathscr{F}^{-1}(\hat{\textbf{x}}) = \textbf{U}\hat{\textbf{x}}$$ 이다. 여기서 $$\textbf{x}$$는 각 node의 signal이다. 

$$\textbf{x}*_{G}\textbf{g} =\mathscr{F}^{-1}(\mathscr{F}(\textbf{x})\odot \mathscr{F}(\textbf{g})) = \textbf{U}(\textbf{U}^T \textbf{x} \odot \textbf{U}^T \textbf{g}) \tag{8}$$

**식 (8)**은 filter $$\textbf{g}_\theta$$ 를 $$diag(U^T\textbf{g})$$으로 정의하였기 때문에 **식 (9)**와 같이 바꿀 수 있다. $$\textbf{g}_\theta$$는 학습 파라미터이다.

$$\textbf{g}_\theta = 
\begin{bmatrix} 
\hat{\textbf{g}}(\lambda_1) & 0 & 0 \\
0 & \ddots & 0 \\
0 & 0 & \hat{\textbf{g}}(\lambda_N)
\end{bmatrix}$$

<br>

$$\textbf{x}*_{G}\textbf{g}_\theta = \textbf{U}\textbf{g}_\theta \textbf{U}^T\textbf{x}  \tag{9}$$

아래는 이를 간단한 예제를 만들어 확인해보는 포스팅이다. 보고 오는 것을 추천!

[The Convolution Theorem and Application Examples - DSPIllustrations.com](https://dspillustrations.com/pages/posts/misc/the-convolution-theorem-and-application-examples.html)

# Graph Convolution에서 GCN으로의 과정

위에 내용에 이어서 GCN에 대해 얘기하자면 위 과정은 말그대로 graph convolution을 spectral 관점에서 해석한 내용이다. GCN으로 가기 위해서는 filter paramter에 대한 변화가 있다.

기존 filter인 $$\textbf{g}_\theta$$는 단지 1 hop만 고려할 수 있다. 그래서 localized feature를 뽑기위해 $$\textbf{g}_\theta$$를 **식 (10)**과 같이 polynomial parameterization 해주었다. 여기서 localized라는 의미는 이웃 노드(1 hop), 이웃 노들의 이웃 노드(2 hops) 등등 k hops 만큼 이웃 노드의 정보를 고려하여 feature를 뽑겠다는 말이다.

$$\textbf{g}_\theta(\Lambda) = \sum_{k=0}^{K} \theta_k \Lambda^k = \theta_0 \Lambda^0 + \theta_1 
\Lambda^1 + \cdots + \theta_K \Lambda^K \tag{10}$$

Polynomial parameterization을 적용한 filter를 **식 (9)**에 대입하면 **식 (11)**과 같이 정의할 수 있다.

$$\begin{align*}
\textbf{x}*_{G}\textbf{g}_\theta &= \textbf{U} \sum_{k=0}^{K}\theta_k\Lambda^k \textbf{U}^T\textbf{x} \\
&= \textbf{U}(\theta_0 \Lambda^0 + \theta_1 \Lambda^1 + \cdots + \theta_K \Lambda^K)\textbf{U}^T\textbf{x} \\
&= \sum_{k=0}^{K} \theta_k \textbf{L}^k \textbf{x}
\tag{11}
\end{align*}$$

**Example**

하지만 위의 식을 보아도 k에 따라 k-hop 만큼 이웃 노드를 반영 하겠다는 말이 직관적으로 다가오지 않을 수 있다. 때문에 간단한 예시를 통해 알아보자.

<p align='center'>
    <img width='300' src='https://user-images.githubusercontent.com/37654013/105801375-5a751b00-5fdc-11eb-8755-cb6487a42490.png'><br> 그림 4. 노드가 5개인 그래프 예시
</p>

**그림 (4)**와 같은 undirected graph $$(G)$$가 있다고 가정하자. 이때 $$\textbf{A}$$와 $$\textbf{D}$$는 다음과 같다. 

$$\textbf{A} = 
\begin{bmatrix}
0 & 1 & 0 & 0 & 1 \\
1 & 0 & 1 & 0 & 1 \\
0 & 1 & 0 & 1 & 0 \\
0 & 0 & 1 & 0 & 1 \\
1 & 1 & 0 & 1 & 0 
\end{bmatrix}, 
\textbf{D} = 
\begin{bmatrix}
2 & 0 & 0 & 0 & 0 \\
0 & 3 & 0 & 0 & 0 \\
0 & 0 & 2 & 0 & 0 \\
0 & 0 & 0 & 2 & 0 \\ 
0 & 0 & 0 & 0 & 3
\end{bmatrix}$$

이제 $$\textbf{L}$$의 $$k$$ 제곱이 실제로 이웃 노드의 정보를 반영하는지 알아보자. 여기서 $$\textbf{L}$$은 앞서 구한 $$\textbf{D} - \textbf{A}$$로 계산할 수 있고 $$k=2$$로 하여 예시를 만들어 보았다. 우선 $$\textbf{L}$$은 다음과 같다.

$$\begin{align*}
\textbf{L} &= \textbf{D} - \textbf{A} \\
           &= \begin{bmatrix}
               2 & -1 & 0 & 0 & -1 \\ 
               -1 & 3 & -1 & 0 & -1 \\ 
               0 & -1 & 2 & -1 & 0 \\ 
               0 & 0 & -1 & 2 & -1 \\ 
               -1 & -1 & 0 & -1 & 3 
              \end{bmatrix}
\end{align*}$$

다음으로 $$\textbf{Lx}$$를 계산하면 각 노드와 해당 노드가 연결된 이웃 노드간의 차이를 계산하게 된다. 

$$\begin{align*}
\textbf{Lx} &= \begin{bmatrix}
               2 & -1 & 0 & 0 & -1 \\ 
               -1 & 3 & -1 & 0 & -1 \\ 
               0 & -1 & 2 & -1 & 0 \\ 
               0 & 0 & -1 & 2 & -1 \\ 
               -1 & -1 & 0 & -1 & 3 
              \end{bmatrix}
              \begin{bmatrix}
              x_1 \\ x_2 \\ x_3 \\ x_4 \\ x_5
              \end{bmatrix} \\
            &= \begin{bmatrix}
               \sum_{i \in [2,3]} (x_1 - x_i) \\
               \sum_{i \in [1,3, 3]} (x_2 - x_i) \\
               \sum_{i \in [2,4]} (x_3 - x_i) \\
               \sum_{i \in [3,5]} (x_4 - x_i) \\
               \sum_{i \in [1,2,4]} (x_5 - x_i) 
               \end{bmatrix}
\end{align*}$$

이때 $$k=1$$인 $$\textbf{Lx}$$의 의미는 자신의 노드와 이웃 노드의 정보만 사용하겠다는 뜻이다. 이제 다음으로 $$k=2$$일 때 $$\textbf{L}(\textbf{Lx})$$를 계산해보자.

$$\begin{align*}
\textbf{L}(\textbf{Lx}) &= \begin{bmatrix}
                            2 & -1 & 0 & 0 & -1 \\ 
                            -1 & 3 & -1 & 0 & -1 \\ 
                            0 & -1 & 2 & -1 & 0 \\ 
                            0 & 0 & -1 & 2 & -1 \\ 
                            -1 & -1 & 0 & -1 & 3 
                            \end{bmatrix}
                            \begin{bmatrix}
                            \sum_{i \in [2,3]} (x_1 - x_i) \\
                            \sum_{i \in [1,3,5]} (x_2 - x_i) \\
                            \sum_{i \in [2,4]} (x_3 - x_i) \\
                            \sum_{i \in [3,5]} (x_4 - x_i) \\
                            \sum_{i \in [1,2,4]} (x_5 - x_i) 
                            \end{bmatrix} \\
                        &= \begin{bmatrix}
                            2\sum_{i \in [2,3]} (x_1 - x_i) - \sum_{i \in [1,3,5]} (x_2 - x_i) - \sum_{i \in [1,2,4]} (x_5 - x_i) \\
                            -\sum_{i \in [2,3]} (x_1 - x_i) + 3\sum_{i \in [1,3,5]} (x_2 - x_i) - \sum_{i \in [2,4]} (x_3 - x_i) - \sum_{i \in [1,2,4]} (x_5 - x_i) \\
                            -\sum_{i \in [1,3,5]} (x_2 - x_i) + 2\sum_{i \in [2,4]} (x_3 - x_i) - \sum_{i \in [3,5]} (x_4 - x_i) \\
                            -\sum_{i \in [2,4]} (x_3 - x_i) + 2\sum_{i \in [3,5]} (x_4 - x_i) - \sum_{i \in [1,2,4]} (x_5 - x_i) \\
                            -\sum_{i \in [2,3]} (x_1 - x_i) - \sum_{i \in [1,3,5]} (x_2 - x_i) - \sum_{i \in [3,5]} (x_4 - x_i) + 3\sum_{i \in [1,2,4]} (x_5 - x_i)
                           \end{bmatrix}
\end{align*}$$

수식이 조금 지저분해 보이기 때문에 node 1의 정보만 가지고 알아보자. $$\textbf{L}(\textbf{Lx})$$을 통해 나온 node 1의 정보는 다음과 같은 의미를 포함하고 있다. 

$$\textbf{L}(\textbf{Lx})(1) = 2\sum_{i \in [2,3]} (x_1 - x_i) - \sum_{i \in [1,3,5]} (x_2 - x_i) - \sum_{i \in [1,2,4]} (x_5 - x_i)$$  

우선 첫 번째로 우변의 $$2\sum_{i \in [2,3]} (x_1 - x_i)$$은 node 1과 다른 이웃 간의 정보를 나타낸다. 다음으로 $$\sum_{i \in [1,3,5]} (x_2 - x_i)$$와 $$\sum_{i \in [1,2,4]} (x_5 - x_i)$$은 node 2와 5의 이웃 간의 정보를 나타낸다. 즉, node 2와 5의 이웃 간의 정보는 node 1의 이웃의 이웃 정보 (2-hop)을 나타내기 때문에 결론적으로 $$\textbf{L}^k\textbf{x}$$는 k-hop의 이웃 정보를 포함한다고 볼 수 있다. 

## ChebNet

**식 (11)**은 localized feature를 추출할 수 있지만 computation complexity가 높다는 단점이 있다. 이러한 이유로 ChebNet에서는 Chebyshev polynomial of the first kind를 통해 근사하여 문제를 해결하였다. 

Chebyshev polynomial of the first kind는 convolution filter를 **식 (12)**와 같이 정의 할 수 있다

$$\textbf{g}_\theta=\sum_{i=0}^{K}\theta_iT_{i}(\tilde{\Lambda}), \ \ \tilde{\Lambda}=\frac{2\Lambda}{\lambda_{max}}-I_n \ \ (-1 < \tilde{\Lambda} <1 ) \tag{12}$$

여기서 $$\tilde{\Lambda}$$는 $$\Lambda$$를 [-1,1] 범위로 스케일링 한 값이다. $$T(\cdot)$$은 Chebyshev polynomial function이다. Chebyshev polynomial of first kind에서 다항식은 다음 **식 (13)**을 따른다.

$$T_{n+1}(x) = 2xT_n(x) -T_{n-1}(x) \tag{13}$$

여기서 $$T_0(x) =1$$ 이고 $$T_1(x)=x$$이다.  **식 (12)**를 **식 (9)**에 적용하면 **식 (14)**와 같이 정의할 수 있다.

$$\textbf{x}*_{G}\textbf{g}_\theta = \textbf{U}(\sum_{i=0}^{K}\theta_iT_{i}(\tilde{\Lambda}))\textbf{U}^T\textbf{x} \tag{14}$$

이때 $$\mathcal{L}$$는 $$\frac{2\textbf{L}}{\lambda_{max}} - I_n$$으로 나타낼 수 있다. **식 (1)**과 같이 $$T_i(\mathcal{L})$$ 또한 $$\textbf{U} T_i(\tilde{\Lambda}) \textbf{U}^T$$ 로 표현할 수 있다. 즉, **식 (14)**를 다음과 같이 **식 (15)**로 표현 할 수 있다.

$$\textbf{x}*_{G}\textbf{g}_\theta = \sum_{i=0}^{K}\theta_iT_{i}(\mathcal{L})\textbf{x} \tag{15}$$

## GCN

드디어 이제 GCN으로 가기위한 긴 여정이 끝날 시간이다. GCN 에서는 앞선 ChebNet 에서 사용한 방법에 크게 3가지 단계로 변화를 준다[^1]. 

- Step 1 : $$K=1$$ , $$\lambda_{max} = 2$$
- Step 2: $$\theta = \theta_0 = -\theta_1$$
- Step 3: $$\textbf{I}_n + \textbf{D}^{-1/2} \textbf{A} \textbf{D}^{-1/2} \rightarrow \tilde{\textbf{D}}^{-1/2} \mathscr{A} \tilde{\textbf{D}}^{-1/2}$$

**Step 1**

GCN에서는 먼저 ChebNet에서 사용하는 필터를 $$K=1$$ 로 정하고 activation function을 통해 non-linearity를 추가해 layer를 깊게 쌓는 방식으로 만들었다. 또한, $$\lambda_{max} = 2$$ 로 가정했다. 

앞선 가정에 따라 **식 (15)**는 **식 (16)**으로 바뀌게 된다. 

$$\textbf{x} *_{G}\textbf{g}_\theta = \theta_0 \textbf{x} - \theta_1 (\textbf{L} - \textbf{I}_n) \textbf{x} \tag{16} $$

그 다음 $$\textbf{L} - \textbf{I}_n = \textbf{A}$$ 를 normalize하여 $$\textbf{D}^{-1/2} \textbf{A} \textbf{D}^{-1/2}$$로 변환하여 다음과 같이 **식 (17)**로 나타냈다. 

$$\textbf{x} *_{G}\textbf{g}_\theta = \theta_0 \textbf{x} - \theta_1 \textbf{D}^{-1/2} \textbf{A} \textbf{D}^{-1/2} \textbf{x} \tag{17} $$

**Step 2**

다음으로는 $$\theta = \theta_0 = -\theta_1$$ 와 같이 parameter를 줄여서 overfitting을 방지하고 **식 (18)**과 같이 간략하게 만들었다.

$$\textbf{x} *_{G}\textbf{g}_\theta = \theta (\textbf{I}_n + \textbf{D}^{-1/2} \textbf{A} \textbf{D}^{-1/2} ) \textbf{x} \tag{18}$$

**Step 3**

**식 (18)**에서 $$\textbf{I}_n + \textbf{D}^{-1/2} \textbf{A} \textbf{D}^{-1/2}$$의 eigenvalue는 [0,2]의 범위를 가진다(Appendix 1). 이 단계에서는 **renormalization trick**을 적용하여 eigenvalue의 최대값의 크기를 줄여주었다. $$\textbf{I}_n + \textbf{D}^{-1/2} \textbf{A} \textbf{D}^{-1/2}$$을 $$\tilde{\textbf{D}}^{-1/2} \mathscr{A} \tilde{\textbf{D}}^{-1/2}$$로 변환하였고 이때 $$\mathscr{A} = \textbf{A} + \textbf{I}_n$$ 이고 $$\tilde{\textbf{D}}_{ii} = \sum_{j} \mathscr{A}_{ij}$$ 이다. 이는 self-loop를 추가함으로써 oversmoothing을 방지하고 eigenvalue의 범위를 줄임으로써 연산 과정에서 stability를 높이기 위함이다(Appendix 2).

$$\textbf{x} *_{G}\textbf{g}_\theta = \theta \tilde{\textbf{D}}^{-1/2} \mathscr{A} \tilde{\textbf{D}}^{-1/2} \textbf{x} \tag{19}$$

이제 **식 (19)**를 **식 (20)**과 같이 일반화 하여 사용할 수 있다.

$$\textbf{H}^{k} = \sigma(\hat{\textbf{A}}\textbf{H}^{k-1}\Theta^{k}) \tag{20}$$

여기서 $$\hat{\textbf{A}} = \tilde{\textbf{D}}^{-1/2} \mathscr{A} \tilde{\textbf{D}}^{-1/2}$$, $$\hat{\textbf{A}} \in \mathbb{R}^{n \times n}$$ 이고 $$\textbf{H}^0 = \textbf{X}$$, $$\textbf{X} \in \mathbb{R}^{n \times C}$$ 이다. $$C$$ 는 input channel 수(= 여기선 각 노드의 $$C$$ 차원의 feature vector)를 말한다.  $$\textbf{H}^{k} \in \mathbb{R}^{n \times F}$$는 $$k$$ 번째 layer의 output이고 $$F$$는 output의 channel 수 이다. $$\Theta^{k} \in \mathbb{R}^{C \times F}$$ 는 $$k$$ 번째 layer의 가중치 행렬이다. 마지막으로 $$\sigma$$는 activation function을 말한다. 

## Example

Class의 수가 5개인 node classification 문제를 푼다고 하자. 이때 GCN을 아래와 같이 설정하였다.

$$k = 3, \ \ F_1 = 30, F_2 = 20, \ \ \sigma = ReLU$$

입력값은 총 30개의 노드가 있고 40개의 feature가 있다. $$\textbf{X} \in \mathbb{R}^{30 \times 40}$$ 

$$\textbf{H}^{(0)} = \textbf{X} \in \mathbb{R}^{30 \times 5}$$

$$\textbf{H}^{(1)} = ReLU(\hat{\textbf{A}} \textbf{H}^{(0)} \Theta^{(1)}) \in \mathbb{R}^{30 \times 30}$$

$$\textbf{H}^{(2)} = ReLU(\hat{\textbf{A}} \textbf{H}^{(1)} \Theta^{(2)}) \in \mathbb{R}^{30 \times 20}$$

$$\textbf{H}^{(3)} = softmax(\hat{\textbf{A}} \textbf{H}^{(2)} \Theta^{(3)}) \in \mathbb{R}^{30 \times 5}$$

위 예시를 통해 짐작할 수 있듯이 GCN의 단점으로는 mini batch 방식으로 학습할 수 없다는 점이다. 때문에 데이터가 큰 경우 학습하기가 쉽지 않다. 

# 결론

Graph에 convolution을 적용한 방법을 크게 두 가지 관점으로 본 것이 spectral과 spatial이고 여기서 언급한 GCN[^1]은 spectral 관점에서 접근한 graph에 convolution을 적용한 방법이다. 

지금까지 내용들을 다시 정리해 보자면 spectral GCN 이란 SP 관점에서 graph 노드들의 signal에 Fourier transform을 적용하여 convolution 연산을 적용할 수 있게 하였고 이때 convolution의 filter 역할을 하는 것이 $$\textbf{L}$$이다. $$\textbf{L}$$은 $$k$$-th order polynomial parameterization을 통해 localized feature를 고려하도록 변환 하였고 이후 Chebyshev polynomial of the first kind를 통해 근사하여 연산량을 낮추었다. 마지막으로 $$k=1$$로 하고 renormalization trick을 적용하여 layer를 여러개 쌓을 수 있게 만든 것이 GCN[^1]이다. 

중간중간 필요한 배경지식이 많기 때문에 spectral GCN을 이해하기란 쉬운 일은 아니었다. GNN은 다른 분야에 비해 진입장벽이 더 높은 편이라고 생각하는데에 가장 큰 기여를 준게 spectral GCN이 아닌가 싶다. 

Spectral GCN은... 사드세요.



# Appendix

## 1. Normalized Adjacency Matrix의 eigenvalue는 [0,2] 이다.

먼저 $$\textbf{A}$$의 eigenvalue 범위에 대해 알아보고 넘어가자.

**1. $$\textbf{A}$$의 가장 큰 eigenvalue 범위**

$$\textbf{A}$$가 undirected graph $$(G)$$의 adjacency matrix이고 $$\lambda_{1}$$이 $$\textbf{A}$$의 가장 큰 eigenvalue라고 하자. 이때, 아래 식이 성립한다. 

$$d_{avg} \leq \lambda_{1} \leq d_{max}$$

여기서 $$d_{avg}$$는 $$G$$의 평균 degree 이고 $$d_{max}$$는 최대 degree 이다.

**[증명 1]**

$$\textbf{v}_1$$을 eigenvalue $$\lambda_1$$에 대응하는 eigenvector이고 $$j$$ 번째 노드는 모든 $$i$$에 대해 $$v_1(j) \geq v_1(i)$$가 성립한다고 하자.

$$\begin{align*}
\lambda_1 v_1(j) &= (\textbf{A}v_1)(j) \\
                 &= \sum_{(i,j)\in E(G)}v_1(i) \leq \sum_{(i,j)\in E(G)} v_1(j) \\
                 &= deg(j)v_1(j) \leq d_{max}v_1(j)
\end{align*}$$

**[예시]**

**증명 1**에서 $$\sum_{(i,j)\in E(G)}v_1(i) \leq \sum_{(i,j)\in E(G)} v_1(j)$$의 부분이 잘 이해가 되지 않을 수 있기에 간단히 예시를 만들었다.

**그림 (2)**와 같은 $$G$$가 있다고 가정하자. 이때 $$G$$로부터 계산한 $$\textbf{A}$$의 첫 번째 eigenvector가 $$v_1$$이라고 하자. $$j$$를 2라고 가정하면, 2번째 노드는 다른 두 노드에 대해 $$v_1(2) \geq v_1(i), i \in [1,3]$$ 이 성립한다. 

$$ A = 
\begin{bmatrix}
0 & 1 & 1 \\
1 & 0 & 1 \\
1 & 1 & 0 
\end{bmatrix}, 
\textbf{v}_1 = \begin{bmatrix}v_1(1) \\ v_1(2) \\ v_1(3)\end{bmatrix}
$$

위 증명식을 따라 계산하면 아래와 같은 결과를 확인할 수 있다.

$$\sum_{(i,j)\in E(G)}v_1(i) = 1\times v_1(1) + 1\times v_1(3)$$ 

여기서 $$v_1(2)$$는 다른 노드보다 항상 더 크거나 같은 값을 가지기 때문에 다음이 성립한다.

$$\sum_{(i,j)\in E(G)}v_1(i) \leq \sum_{(i,j)\in E(G)} v_1(2)$$

**[증명 2]**

모든 벡터의 원소가 1인 $$\textbf{x}$$에 대해 $$\textbf{A}$$에 대한 레일리 몫(Rayleigh quotient)을 적용하면 다음이 성립한다.

$$\begin{align*}
\mu_1 &= \max_{x} \frac{x^T\textbf{A}x}{x^Tx} \geq \frac{\textbf{1}^T\textbf{A}\textbf{1}}{\textbf{1}^T\textbf{1}} \\
      &= \frac{\sum_{(i,j) \in E(G)}\textbf{A}(i,j)}{n} \\
      &= \frac{\sum_i d(i)}{n} \\
      &= d_{avg}
\end{align*}$$

<br>

**2. Normalized $$\textbf{A}$$의 eigenvalue 범위는 [0,2]이다.**

먼저 normalized $$\textbf{A}$$ $$(\mathscr{A})$$와 normalized $$\textbf{L}$$ $$(\mathscr{L})$$은 다음과 같이 정의할 수 있다.

$$\mathscr{A} = \textbf{D}^{-1/2}\textbf{A}\textbf{D}^{-1/2}$$

$$\mathscr{L} = \textbf{I} - \mathscr{A}$$

여기서 $$\mathscr{D}$$는 다음과 같이 풀어서 볼 수 있다.

$$\begin{align*}
\mathscr{L} &= \textbf{I} - \mathscr{A} \\
                   &= \textbf{D}^{-1/2}(\textbf{D} - \textbf{A})\textbf{D}^{-1/2} \\
                   &= \textbf{D}^{-1/2}\textbf{L}\textbf{D}^{-1/2}
\end{align*}$$

이제 앞서 $$\textbf{A}$$의 eigenvalue의 범위가 $$d_{avg} \leq \lambda_1 \leq d_{max}$$ 인것을 확인했다. $$\textbf{A}$$에 "Normalizing" 한다는 의미는 가장 큰 eigenvalue의 값을 1로 만들겠다는 뜻이다. 즉, normalized를 거친 결과는 아래 내용을 통해 증명할 수 있다.

$$\alpha_1 \geq \cdots \geq \alpha_n$$이 $$\tilde{\text{A}}$$의 eigenvalue이고 $$\lambda_1 \leq \cdots \leq \lambda_n$$은 $$\mathscr{L}$$의 eigenvalue라고 하자. 이때 다음이 성립한다.

$$ 1 = \alpha_1 \geq \cdots \geq \alpha_n \geq -1$$

$$ 0 = \lambda_1 \leq \cdots \leq \lambda_n \leq 2$$

**[증명]**

우선, $$\textbf{x} = \textbf{D}^{-1/2}\textbf{v}$$를 통해 $$\mathscr{L}$$의 eigenvalue가 0이 됨을 나타낼 수 있다. 여기서 $$\textbf{v}$$는 $$\textbf{L}$$의 eigenvector이다.

$$\begin{align*}
\mathscr{L}(\textbf{D}^{1/2}\textbf{v}) &= \textbf{D}^{-1/2}\textbf{L}\textbf{D}^{-1/2}\textbf{D}^{1/2}\textbf{v} \\
&= \textbf{D}^{-1/2}\textbf{L}\textbf{v} \\
&= 0
\end{align*}$$

$$\textbf{v}$$는 0의 값을 갖는 eigenvalue에 대응하는 $$\textbf{L}$$의 eigenvector 이기 때문에 $$\textbf{D}^{1/2}\textbf{v}$$에 해당하는 $$\mathscr{L}$$의 eigenvalue는 0이다. 이 값이 가장 작은 eigenvalue 임을 보이기 위해서는 $$\mathscr{L}$$가 positive semi-definite 임을 알아야 한다. positive semi-definite은 모든 $$\textbf{x} \in \mathbb{R}^n$$에 대해 $$\textbf{x}^T\mathscr{L}\textbf{x}$$의 결과가 항상 0보다 크거나 같은 값을 나타내는 것을 말한다.

$$\begin{align*}
\textbf{x}^T \mathscr{L} \textbf{x} &= \textbf{x}^T(\textbf{I} - \mathscr{A})\textbf{x} \\
                                    &= \sum_{i\in V} x(i)^2 - \sum_{(i,j) \in E} \frac{2x(i)x(j)}{\sqrt{d(i)d(j)}} \\
                                    &= \sum_{(i,j) \in E}(\frac{x(i)}{\sqrt{d(i)}} - \frac{x(j)}{\sqrt{d(j)}})^2 \\
                                    &\geq 0
\end{align*}$$ 
                                                             
이제 $$\mathscr{L}$$가 positive semi-definite이고 $$\lambda_1 = 0$$ 인 것을 보였다. 다음으로 $$\alpha_1 \leq 1$$임을 보이기 위해 positive semi-definite인 $$\mathscr{L}$$를 활용하면 모든 $$\textbf{x} \in \mathbb{R}^n$$에 대해 다음이 성립한다. 

$$\textbf{x}^T(\textbf{I}-\mathscr{A})\textbf{x} \geq 0 \rightarrow \textbf{x}^T\textbf{x} - \textbf{x}^T\mathscr{A}\textbf{x} \geq 0 \rightarrow 1 \geq  \frac{\textbf{x}^T\mathscr{A}\textbf{x}}{\textbf{x}^T\textbf{x}}$$

$$\therefore 1 \geq \alpha_1$$

레일리 몫을 통해 $$\alpha_1$$의 상한선을 나타낼 수 있다. 위 식은 $$\textbf{x} = \textbf{D}^{1/2}\textbf{v}$$을 다시 대입해보면 앞서 언급한 내용을 활용해서 $$\mathscr{A}$$의 가장 큰 eigenvalue가 1임을 알 수 있다. 

$$\textbf{x}^T\mathscr{L}\textbf{x}=0 \rightarrow \textbf{x}^T(\textbf{I} - \mathscr{A})\textbf{x} = 0 \rightarrow \textbf{x}^T\textbf{x} = \textbf{x}^T\mathscr{A}\textbf{x}$$

이와 비슷하게 $$\alpha_n$$의 하한선을 구하기 위해 positive semi-definite 인 $$\textbf{I} + \mathscr{A}$$을 적용해보면 다음과 같이 나타낼 수 있다.

$$\textbf{x}^T(\textbf{I}+\mathscr{A})\textbf{x} \geq 0 \rightarrow \textbf{x}^T\textbf{x} + \textbf{x}^T\mathscr{A}\textbf{x} \geq 0 \rightarrow \frac{\textbf{x}^T\mathscr{A}\textbf{x}}{\textbf{x}^T\textbf{x}} \geq -1 $$

$$\therefore \alpha_n \geq -1$$

마지막으로 $$\textbf{x}^T(\textbf{I}+\mathscr{A})\textbf{x} \geq 0$$을 $$\mathscr{L}$$에 대해 나타내면 레일리의 몫을 통해 $$\lambda_n$$의 상한선에 대해 나타낼 수 있다.

$$-\textbf{x}^T\mathscr{A}\textbf{x} \leq \textbf{x}^T\textbf{x} \rightarrow \textbf{x}^T\textbf{I}\textbf{x} - \textbf{x}^T\mathscr{A}\textbf{x} \leq 2\textbf{x}^T\textbf{x} \rightarrow \frac{\textbf{x}^T\mathscr{L}\textbf{x}}{\textbf{x}^T\textbf{x}} \leq 2$$

$$\therefore \lambda_n \leq 2$$

## 2. Renormalization trick

$$\textbf{I}_n + \textbf{D}^{-1/2} \textbf{A} \textbf{D}^{-1/2}$$를 $$\textbf{S}_{1-order}$$라고 하면 이를 다음과 같이 $$\mathscr{L}$$로 나타낼 수 있다.

$$\textbf{S}_{1-order} = 2\textbf{I} - \mathscr{L}$$

이때 $$\textbf{S}^{K}_{1-order}$$은 filter coefficient가 $$(2I-\lambda_i)^K$$인 값을 가진다. 여기서 $$\lambda_i$$는 $$\mathscr{L}$$의 eigenvalue이다. **그림 (5)**를 보면 $$\lambda_i < 1$$ 에서 K가 커짐에 따라 filter coefficient의 값이 급격히 커짐을 알 수 있다[^2]. 

<p align='center'>
    <img src='https://user-images.githubusercontent.com/37654013/105690922-ec7b1600-5f3f-11eb-8e5f-40421b01d59d.png'><br>그림 5. Simplifying GCN에서 실험한 augmented normalized adjacency의 효과
</p>

이러한 것을 방지하기 위해 eigenvalue의 크기를 줄이려고 사용한 방법이 바로 renormalization trick이다. Renormalization trick을 적용한 결과값은 $$\tilde{\textbf{S}}_{adj} = \tilde{\textbf{D}}^{-1/2}\tilde{\textbf{A}}\tilde{\textbf{D}}^{-1/2}$$ 으로 나타낸다. 여기서 $$\tilde{\textbf{A}} = \textbf{A} + \textbf{I}$$ 이고 $$\tilde{\textbf{D}} = \textbf{D} + \textbf{I}$$ 이다. 이를 통해 $$\tilde{\textbf{L}} = \textbf{I} - \tilde{\textbf{D}}^{-1/2}\tilde{\textbf{A}}\tilde{\textbf{D}}^{-1/2}$$ 으로 나타낼 수 있다. 결과적으로 renormalization trick을 적용한 filter의 coefficient는 $$\hat{g}(\tilde{\lambda_i}) = (1 - \tilde{\lambda_i})^K$$ 이고 여기서 $$\tilde{\lambda_i}$$는 $$\tilde{\textbf{L}}$$의 eigenvalue 이다.

**그림 (5)**에서 가운데는 self-loop를 추가하지 않은 $$\textbf{D}^{-1/2}\textbf{A}\textbf{D}^{-1/2}$$의 filter coefficient를 조사한 결과이고 오른쪽은 self-loop를 추가한 $$\tilde{\textbf{D}}^{-1/2}\tilde{\textbf{A}}\tilde{\textbf{D}}^{-1/2}$$의 filter coefficient를 나타낸 결과이다. 

Self-loop를 추가하게 되면 최대 eigenvalue 값이 줄어들게 되는데 이는 다음과 같은 이유이다. 

**[Theorem 1]**

$$\textbf{A}$$가 undirected graph $$G$$에 대한 adjacency matrix이고 이에 대한 degree matrix는 $$\textbf{D}$$라고 하자. 이 때 augmented adjacency matrix를 $$\tilde{\textbf{A}} = \textbf{A} + \gamma\textbf{I}$$, $$\gamma > 0$$ 로 나타내고 이에 대한 degree matrix는 $$\tilde{\textbf{D}}$$이다. 또한, $$\mathscr{L} = \textbf{I} - \textbf{D}^{-1/2}\textbf{A}\textbf{D}^{-1/2}$$의 가장 작은 eigenvalue와 가장 큰 eigenvalue를 각각 $$\lambda_1$$ 그리고 $$\lambda_n$$로 나타낸다. 이와 같이 $$\tilde{\textbf{L}} = \textbf{I} - \tilde{\textbf{D}}^{-1/2}\textbf{A}\tilde{\textbf{D}}^{-1/2}$$의 가장 작은 eigenvalue와 가장 큰 eigenvalue를 각각 $$\tilde{\lambda_1}$$ 그리고 $$\tilde{\lambda_n}$$로 나타낸다. 이제 다음과 같은 식이 성립한다. 

$$0=\lambda_1=\tilde{\lambda_1} < \tilde{\lambda_n} < \lambda_n$$


# Reference

[^1]: T. N. Kipf and M. Welling, “Semi-supervised classification with graph convolutional networks,” in Proc. of ICLR, 2017.

[^2]: Wu, F., Souza, A., Zhang, T., Fifty, C., Yu, T., & Weinberger, K. (2019, May). Simplifying graph convolutional networks. In International conference on machine learning (pp. 6861-6871). PMLR.

[^3]: Chung, F. R., & Graham, F. C. (1997). Spectral graph theory (No. 92). American Mathematical Soc..