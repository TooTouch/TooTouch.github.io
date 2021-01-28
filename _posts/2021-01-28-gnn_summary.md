---
title: "Graph Neural Network 찍어먹기"
categories: 
    - Research
toc: true
---

*긴 글에 앞서 survey paper를 보고 정리하는 데 도움을 주신 DSBA [윤훈상 선배님](https://hoonst.github.io)께 무한한 감사의 말을 전합니다.*

**찍어먹기란?** 제목과 같이 해당 포스팅은 말그대로 **'찍어먹기'**다. 그동안 graph를 공부만 해야지 생각했다가 드디어 시간이 생겨서! 는 아니지만 드디어 공부하기 시작했다. 먼저 cs224w 수업을 듣다가 아무래도 전반적인 개론과 같은 내용이 머리 속에 그려져야 할 거 같아서 survey paper를 보았다. 깊이 공부하기에는 너무 많고 그렇다고 안보기에는 아쉽고해서 survey paper를 통해 GNN의 발전 과정과 여러 접근 방법들을 전반적으로 보기로 했다. 

# Introduction

내가 본 survey paper는 "[A comprehensive survey on graph neural networks.](https://arxiv.org/pdf/1901.00596.pdf)" 이다. 작년 3월에 IEEE transactions on neural networks and learning systems에 출판되었고 현재는 벌써 인용 수가 1000회를 넘었다. 이 survey paper에서는 왜 graph neural network (GNN)에 관련된 내용이고 그래프는 다른 데이터와 어떤점이 다른지 그리고 적용된 neural network를 크게 4가지로 나누어 설명한다[^1]. 그 안에서도 맥락이 또 나뉘는데 아래에서 분류별로 정리하려 한다. 실제 논문에서는 다양항 학습 과정에서의 관점과 여러 활용 사례들이 있지만 여기서는 학습 방법을 위주로 정리한다. 사실 이미 정리된 걸 또 정리하는 것이기 때문에 단순히 한국어로만 변역해서 쓰는 것이 아닌가 싶은 생각도 있지만 추가적으로 내가 정리하는 내용은 다음과 같다.

- Spectral GCN에 대한 중간에 생략된 설명들
- Survey paper를 통해 소개된 여러 방법들의 연도별 Roadmap
- 각 논문들의 source code

## Background

기존 머신러닝은 *euclidean* data에서는 굉장히 잘 적용된다. 하지만 *non-euclidean* data에는 이와 반대로 적용하기 어려운데 그 이유는 선형회귀와 같은 머신러닝 모델들이 대부분 가정하는 것이 feature가 서로 독립이어야 한다는 것이다. 여기서 *non-euclidean* data란 **그림 (1)**과 같이 graph 또는 tree 구조를 가진 데이터를 말한다. Graph는 연결된 edge들이 방향성(*directed*)이 있거나 없거나(*undirected*) 순환(*cyclic*)되는 구조가 있다. 한편 tree는 방향성(directed)이 있는, 즉 유향 비순환(*acyclic*) 그래프의 한 종류로 생각하면 된다. 단, tree는 root node가 하나로 고정되어 있다. 이러한 *non-euclidean* data는 각 feature 간의 관계(*relation*)가 있기 때문에 기존 머신러닝 방법으로는 접근이 어렵다. 따라서 이 문제를 해결하기 위해 다양한 neural network가 제안되었다. 

<p align='center'>
    <img width='500' src='https://user-images.githubusercontent.com/37654013/104730902-387acd80-577e-11eb-91e8-f153cfa7b5a6.png'><br>그림 1. Tree와 Graph 구조 예시
</p>

Survey paper에서는 GNN을 다음과 같이 크게 4가지 맥락으로 나누었다[^1].

- Recurrent Graph Neural Networks (RecGNNs)
- Convolutional Graph Neural Networks (ConvGNNs)
- Graph AutoEncoders (GAEs)
- Spatial-Temporal Graph Neural Networks (STGNNs)

각 맥락별 핵심은 다음과 같다. **RecGNN**s는 recurrent를 기반으로 iteration을 통해 node를 학습하는 방법을 말한다. **ConvGNNs**은 Computer Vision에서 사용되는 CNN을 기반으로 각 node 또는 edge의 정보를 계산하는 방법을 말한다. ConvGNNs에는 크게 *spectral*과 *spatial*로 나뉘는데 이에 대한 설명은 아래에서 이어하기로 한다. **GAEs**는 기존 AE와 같이 graph data를 embedding하거나 graph를 생성하기 위한 방법을 말한다. 마지막으로 **STGNNs**는 graph의 spatial과 temporal을 함께 고려한 방법이다. 즉, 공간과 순서를 함께 고려했다고 생각하면 이해하기 쉽다. 각 방법에 대한 내용은 **그림 (2)**를 통해 간략하게 볼 수 있다.

<p align='center'>
    <img src='https://user-images.githubusercontent.com/37654013/105831118-ad18fc00-6009-11eb-8312-00f36c50e64a.png'><br>그림 2. 4가지 분류에 따른 Graph Neural Network의 학습 방법
</p>

## Roadmap

Survey paper에서 소개하는 논문은 학습 방법만 따지면 대략 48개의 paper를 소개한다. 물론 각 방법마다 자세하게 설명하는 것은 아니라 핵심 내용만 소개하거나 이전 방법에서 개선된 점만 소개한다. **그림 (3)**은 survey paper에서 분류한 GNN 학습 방법을 연도별 그리고 방법별로 나타낸 그림이다. 

<p align='center'>
    <img src='https://user-images.githubusercontent.com/37654013/106168836-12b7e480-61d2-11eb-8997-178df3effcac.png'><br>그림 3. Survey paper에서 소개하는 방법들에 대한 연도와 분류별 Roadmap
</p>


## Terminology

기본적인 그래프에서 사용하는 notion은 survey paper에서 정리한 **표 (1)**을 보면 된다. 

<p align='center'>
    <img width='500' src='https://user-images.githubusercontent.com/37654013/104730344-39f7c600-577d-11eb-8664-1bcd0d3064f0.png'><br>표 1. GNN에서 사용되는 기본 용어
</p>

하지만 graph 개념이 아예 처음이라면 아래 간단한 개념 정도는 짚고 넘어가야 한다. 하나씩 살펴보기 보다 **그림 (4)**를 통해 한번에 알아보자.

<p align='center'>
    <img src='https://user-images.githubusercontent.com/37654013/105630334-e4fa3500-5e8b-11eb-8495-9df3a8e17331.png'><br>그림 4. Graph에 대한 기본 내용
</p>

- **Node or vertex ($$V$$)** : 그래프에서 꼭지점에 해당하는 부분
- **Neighbor of a node $$v$$** : $$v$$번째 노드의 이웃
- **Edge or link ($$E$$)** : 노드를 연결하는 선
- **Degree** : 하나의 노드가 이웃과 연결된 edge 수
- **Adjacency matrix ($$A$$)** : 인접 행렬이라고 불리며 노드 간 연결된 경우 1 아닌 경우 0인 정방 행렬
- **Degree matrix ($$D$$)** : 대각 원소가 각 노드의 degree로 이루어진 대각 행렬


# Recurrent Graph Neural Networks (RecGNNs)

<p align='center'>
    <img width='400' src='https://user-images.githubusercontent.com/37654013/106140765-877b2680-61b2-11eb-991d-0252e03380c0.png'><br>그림 5. RecGNNs Roadmap
</p>

**1. Learning representations by back-propagating errors (1997)[^2]**

neural network static infromation or temporal sequence structure data 에는 잘 되지만 graph 같은 structure data에는 잘 안된다. 그 이유는 graph의 경우 structure 사이즈가 다양하기 때문이다. 그래서 이러한 사이즈의 변화에도 사용할 수 있는 방법인 `generalized recursive network`를 제안했다. 하지만 computational limitation이 존재한다. 또한, directed graph에만 적용 가능하다는 단점이 있다.

**2.1 A new model for learning in graph domains (2005)[^3]**
**2.2 The graph neural network model, GNN (2008)[^4]**

이후 GNN 이라는 단어를 처음 사용한 논문이 공개 되었다. 97년도에 발표된 `generalized recursive network`와는 달리 directed, undirected, cycle이 있는 그래프 모두 적용 가능하다. 또한, 앞선 문제였던 computational complexity를 $$O(m)$$으로 줄였다. 여기서 $$m$$은 edge 수를 말한다. 즉, edge가 늘어남에 따라 complexity 또한 선형적으로 증가함을 말한다. 이유는 잘 모르겠지만 2005년에 처음 GNN을 소개했는데 앞선 논문[^3]에 더해서 time complextity를 줄이는 내용을 더 디테일하게 소개하며 2008년에 두 명의 저자가 추가되어 또 발표했다. 

학습 방법은 이웃 노드들의 information을 recurrent하게 전달하여 node state를 업데이트 한다. 이 과정을 node state가 수렴할 때까지 iteration을 반복해야 한다.

**3. Graph Echo State Networks, GraphESN (2010)[^5]**

GraphESN을 간단히 설명하자면 encoder와 output layer로 구성되어 있고 encoder를 통해 전체 graph를 나타내는 값으로 변환한 뒤 고정한다. 이름과 같이 고정된 encoder를 echo로서 언급한 듯 하다. encoder를 학습하는 방법은 contractive state transition function을 통해 recurrent하게 노드를 업데이트하며 global graph state가 수렴할 때까지 반복한다.  

- **[Author github (python)](https://github.com/gvisco/GraphESN)**

**4. Gated graph sequence neural networks, GGNN (2015)[^6]**

이전에는 수렴할 때까지 iteration을 반복해야 한다는 단점이 있었지만 GGNN은 step 수를 고정해서 학습하기 때문에 convergence를 위한 iteration parameter가 없다는 것이 장점이다. 또한 이름에서와 같이 recurrent function으로 gated recurrent unit (GRU)을 사용하였다. 이전 GNN과 GraphESN 과의 차이점이라면 back-propagation through time (BPTT)를 사용했다는 것이지만 이러한 방법이 단점이 되기도 한다. Large scale 모델의 경우 모든 node에 대해 intermediate state를 가지고 있어야하기 때문에 memory가 많이 필요하다는 게 흠이다. 

- **[Author github (Lua)](https://github.com/yujiali/ggnn)** 
- **[Unofficial Pytorch](https://github.com/chingyaoc/ggnn.pytorch)** 
- **[Unofficial Tensorflow](https://github.com/microsoft/tf-gnn-samples)** 

**5. Stochastic Steady-state Embedding, SSE (2018)[^7]**

SSE는 앞선 large sacle 모델에 대한 메모리 문제점을 해결할 수 있는 방법이다. Node의 hidden state를 stochastic 방식으로 업데이트 하기 때문에 node를 배치로 샘플링해서 업데이트 할 수 있고 gradient 또한 마찬가지로 배치 단위로 계산한다. 하지만 이러한 배치 단위의 학습이 stability를 낮출 수 있는데 이전 state와 현재 state에 가중 평균을 적용해서 stability를 높였다.

- **[Author github (C++)](https://github.com/Hanjun-Dai/steady_state_embedding)** 

# Convolutional Graph Neural Networks (ConvGNNs)

## Spectral vs Spatial 

일반적으로 Graph data에 Convolution을 적용하기 위해 크게 spectral과 spatial 두 가지 관점으로 접근한다. Spectral 모델은 graph signal processing으로 접근하여 이론적인 기반이 튼튼하다는 장점이 있지만 spatial 모델들에 비해 **효율성(efficiency)**, **일반화(generality)**, 그리고 **유연성(flexibility)**이 떨어진다는 단점이 있다. 이러한 이유로 보통은 spatial 모델을 더 선호한다고 한다. 

첫 번째로 spectral 모델을 사용하기 위해서는 eigenvector를 사용하기 때문에 연산 시간이 많이 소비되고 배치 단위로 학습이 안되기 때문에 사이즈가 큰 데이터는 비효율적이라는 단점이 있다. 두 번째로 spectral 모델은 학습한 데이터의 Fourier basis를 기반으로 하기 때문에 새로운 데이터를 적용했을 때 기존에 가지고 있던 eigenbasis가 변해서 일반화가 떨어진다는 단점이 있다. 마지막으로 spectral 모델은 undirected graph에 제한되어 있기 때문에 다른 graph에는 사용할 수 없다는 단점이 있다. 

종합해보면 결국 spectral은 이론적으로는 훌륭하나 써먹을 곳이 많지 않다는 말이다. 조금 더 다양한 graph에 적용할 수 있고 사이즈가 큰 데이터에도 학습할 수 있는 spatial 모델들이 더 유용한 것 같다. 

## Spectral models

<p align='center'>
    <img width='400' src='https://user-images.githubusercontent.com/37654013/106140805-98c43300-61b2-11eb-9393-4fa6f99e2f98.png'><br>그림 6. Spectral 기반 GCNs Roadmap
</p>

Spectral 모델은 graph signal processing을 기반으로 한다. Fourier transform을 활용하여 convolution 연산을 수행한다. Spectral 모델에 대한 convolution 과정은 "[Spectral GCN 은… 사드세요](https://tootouch.github.io/research/spectral_gcn/)"에 자세히 작성했으니 spectral 과정을 보고 모델들을 보는게 이해가 쉽다.

Spectral 모델은 아래 **식 (1)**과 같이 Fourier transform을 적용하여 convolution 연산하는 것을 기본으로 전제한다.

$$\textbf{x}*_{G}\textbf{g} =\mathscr{F}^-1(\mathscr{F}(\textbf{x})\odot \mathscr{F}(\textbf{g})) = \textbf{U}(\textbf{U}^T \textbf{x} \odot \textbf{U}^T \textbf{g}) \tag{1}$$

**식 (1)**은 **식 (2)**와 같이 정리할 수 있다.

$$\textbf{x}*_{G}\textbf{g}_\theta = \textbf{U}\textbf{g}_\theta \textbf{U}^T\textbf{x}  \tag{2}$$


**1. Spectral Convolutional Neural Network, Spectral CNN (2013)[^8]**

Spectral CNN은 아래 **식 (3)**을 통해 convolution 연산을 한다. 이미지에서 사용되는 convolution과 마찬가지로 pooling을 통해 이웃들 간의 정보를 합하여 layer를 쌓아나간다. 

$$\textbf{H}_{:,j}^{k} = \sigma(\sum_{i=1}^{f_{k-1}} \textbf{U}\Theta_{i,j}^{(k)}\textbf{U}^T\textbf{H}_{:,j}^{(k-1)}), \ (j=1,2,\cdots,f_k) \tag{3}$$

여기서 $$k$$는 layer index이고 $$\textbf{H}^{(k-1)} \in \mathbb{R}^{n \times f_{k-1}}$$은 입력값인 graph signal을 말한다. 이때 초기 입력값은 $$\textbf{H}^{(0)}=\textbf{X}$$이다. $$f_{k-1}$$은 입력값의 채널 수이고 $$f_k$$는 출력값의 채널 수 이다. $$\Theta_{i,j}^{(k)}는 학습 파라미터로 단위 행렬이다. 

Spectral CNN은 크게 세 가지 단점이 있다. 첫 번째로는 만약 학습된 모델에 기존 graph signal을 약간만 perturbation을 적용해도 eigenbasis가 바뀌어 버린다. 이는 spectral 모델의 큰 단점이기도 하다. 두 번째로는 학습한 filter를 다른 graph 구조를 가지고 있는 domain에 적용할 수 없다는 단점이 있다. 마지막으로 eigen-decomposition을 하기위한 연산 시간이 $$O(n^3)$$으로 오래걸린다는 단점이 있다. 

**2. Chebyshev Spectral CNN, ChebNet (2016)[^10]**

ChebNet은 이러한 Spectral CNN 연산시간을 Chebyshev polynomial of the first kind를 통해 filter를 근사하여 시간 복잡도를 $$O(m)$$으로 낮추었다. 또한 filter를 다항식으로 바꿔서 graph 사이즈와는 별개로 local feature를 추출할 수 있다는 장점이 있다. 이 말이 직관적으로 와닿지 않는다면 "[Spectral GCN 은… 사드세요 - Graph Convolution에서 GCN으로의 과정](https://tootouch.github.io/research/spectral_gcn/#graph-convolution에서-gcn으로의-과정)"을 읽어보는 것을 추천한다.

Chebyshev polynomial of the first kind는 convolution filter를 **식 (4)**와 같이 정의 할 수 있다

$$\textbf{g}_\theta=\sum_{i=0}^{K}\theta_iT_{i}(\tilde{\Lambda}), \ \ \tilde{\Lambda}=\frac{2\Lambda}{\lambda_{max}}-I_n \ \ (-1 < \tilde{\Lambda} <1 ) \tag{4}$$

여기서 $$\tilde{\Lambda}$$는 $$\Lambda$$를 [-1,1] 범위로 스케일링 한 값이다. $$T(\cdot)$$은 Chebyshev polynomial function이다. Chebyshev polynomial of the first kind에서 다항식은 다음 **식 (5)**를 따른다.

$$T_{n+1}(x) = 2xT_n(x) -T_{n-1}(x) \tag{5}$$

여기서 $$T_0(x) =1$$ 이고 $$T_1(x)=x$$이다.  **식 (4)**를 **식 (2)**에 적용하면 **식 (6)**과 같이 정의할 수 있다.

$$\textbf{x}*_{G}\textbf{g}_\theta = \textbf{U}(\sum_{i=0}^{K}\theta_iT_{i}(\tilde{\Lambda}))\textbf{U}^T\textbf{x} \tag{6}$$

이때 $$\mathcal{L}$$는 $$\frac{2\textbf{L}}{\lambda_{max}} - I_n$$으로 나타낼 수 있다. $$T_i(\mathcal{L})$$은 $$\textbf{L} = \textbf{U}\Lambda\textbf{U}^T$$와 같이 $$\textbf{U} T_i(\tilde{\Lambda}) \textbf{U}^T$$ 로 표현할 수 있다. 즉, **식 (6)**을 다음과 같이 **식 (7)**로 표현 할 수 있다.

$$\textbf{x}*_{G}\textbf{g}_\theta = \sum_{i=0}^{K}\theta_iT_{i}(\mathcal{L})\textbf{x} \tag{7}$$

- **[Author github (Tensorflow)](https://github.com/mdeff/cnn_graph)**

**3. Graph Convoluional Network, GCN (2017)[^11]**

GCN는 **식 (7)**에서 $$K=1$$로 $$\lambda_{max}=2$$로 정의하였고 $$\theta = \theta_0 = -\theta_1$$로 parameter를 줄이면서 over-fitting을 방지하면서 식을 더 간단하게 정의할 수 있게 되었다. 마지막으로 renormalization trick을 적용하여 최종적으로 **식 (8)**과 같이 나타내었다. 여기서도 중간 과정에 대해 궁금하다면 "[Spectral GCN 은… 사드세요 - GCN](https://tootouch.github.io/research/spectral_gcn/#gcn)"을 참고하면 생략된 과정과 renormalization trick에 대해 이해할 수 있다. 

$$\textbf{H} = \textbf{X} *_G \textbf{g}_\Theta = f(\bar{\textbf{A}}\textbf{X}\Theta) \tag{8}$$ 

여기서 $$\bar{\textbf{A}} = \tilde{\textbf{D}}^{(-1/2)}\tilde{\textbf{A}}\tilde{\textbf{D}}^{-1/2}$$이고 이때 $$\tilde{\textbf{A}} = \textbf{A} + \textbf{I}_n$$ 그리고 $$\tilde{\textbf{D}}_{ii} = \sum_j \tilde{\textbf{A}}_{ij}$$이다. 

Spatial 관점에서 보자면 GCN 또한 layer를 쌓으면서 이웃 노드들의 정보를 aggregation하여 학습하기 때문에 비슷한 방법으로 볼 수 있다. 

- **[Author github (Tensorflow)](https://github.com/tkipf/gcn)** 
- **[Author github (Pytorch)](https://github.com/tkipf/pygcn)** 

**4. Adaptive Graph Convolutional Network, AGCN (2018)[^13]**

AGCN는 adjacency matrix로 표현되지 않은 노드 간의 관계를 학습하기 위해 "residual graph adjacency matrix"라는 것을 활용한다. 입력값으로 들어오는 두 노드의 거리를 학습 파라미터로 사용하여 학습한다. 

- **[Author github (Tensorflow)](https://github.com/uta-smile/Adaptive-Graph-Convolutional-Network)** 

**5. Dual Graph Convolutional Network, DGCN (2018)[^14]**

이름에서 알 수 있듯이 DGCN은 graph convolution 과정을 "parallel"하게 학습한다. 두 개의 layer는 서로 파라미터를 공유하고 $$\bar{\textbf{A}}$$와 positive pointwise mutual information (PPMI)를 사용하여 graph에서 sampling을 통해 random walks를 적용한 후 노드들의 co-occurrence 정보를 계산한다. PPMI는 **식 (9)**와 같이 정의한다.

$$PPMI_{v_1,v_2} = max(log(\frac{count(v_1, v_2)\cdot |\textbf{D}|}{count(v_1)count(v_2)}, 0)) \tag{9}$$

여기서 $$v_1, v_2 \in V$$이고 $$\mid\textbf{D}\mid= \sum_{v_1,v_2} count(v_1, v_2)$$이다. 또한 $$count(\cdot)$$는 node $$v$$와 node $$u$$가 random walks로부터 발생한(co-occur/occur) 빈도수(frequency)를 계산하는 함수이다. 

DGCN은 이러한 dual graph convolutional layers를 통해서 결과를 ensemble하여 layer를 깊게 쌓지 않아도 local 정보와 global 정보를 함께 encoding 할 수 있는 방법이다. 

- **[Author github (Theano)](https://github.com/ZhuangCY/DGCN)** 

## Spatial models

<p align='center'>
    <img width='400' src='https://user-images.githubusercontent.com/37654013/106168751-fd42ba80-61d1-11eb-8708-f708efe5a96f.png'><br>그림 7. Spatial 기반 GCNs Roadmap
</p> 

이미지에서 사용하는 convolution과 유사하게 spatial 기반의 방법들은 노드들의 spatial 관계를 활용하여 학습하는 방법이다. Spetial 관계 라는건 중심 노드와 주변 노드의 관계를 말한다. Spatial 모델의 학습 과정은 convolution 과정을 통해서 중심 노드와 주변 노드의 representation을 학습하여 중심 노드의 representation을 업데이트 하는 방식을 말한다. 또다른 관점으로는 spatial 기반 GCN 모델은 앞서 설명한 RecGNNs의 information propagation 또는 message passing과 같은 개념으로 볼 수 있다. 즉, Spatial 모델 또한 edge를 통해 각 노드의 information을 전달하는 방식이다. 

**1. Neural Network for Graphs, NN4G (2009)[^15]**

NN4G는 spatial 기반으로 GCN을 적용한 첫 논문이다. NN4G는 각 layer 마다 독립적인 parameter를 사용해서 'graph mutual dependency'를 학습하도록 하였다. (논문에서 언급된 'graph mutual dependency'가 정확히 어떤 의미인지는 명확하게 알 수 없었으나 layer 마다 서로 다른 parameter를 사용했기 때문에 layer 마다의 output이 서로 의존한다는 의미로 이해했다.) NN4G에서는 각 노드와 주변 노드 같의 합을 통해 aggregation 하였고 이전 layer의 output을 전달하기 위한 skip connection과 residual connection을 사용하였다. (논문에서 'and' 로 작성되어 있어 두 connection이 서로 차이가 있는걸로 보는거 같지만 어떤 차이인지는 명확하지 않았다. 그냥 기존에 알고있는 skip connection을 적용했다 정도만 이해하고 넘어가도 좋다.)

NN4G에 대한 수식은 **식 (10)**과 같다.

$$\textbf{h}_v^{(k)} = f(\textbf{W}^{(k)^T}\textbf{x}_v + \sum_{i=1}^{k-1} \sum_{u \in N(v)} \Theta^{(k)^T}\textbf{h}_u^{k-1}) \tag{10}$$

여기서 $$f(\cdot)$$은 activation function이고 $$\textbf{h}_v^{(0)}=0$$ 이다. **식 (10)**은 **식 (11)**과 같이 행렬로 나타낼 수 있다.

$$\textbf{H}^{(k)} = f(\textbf{X}\textbf{W}^{(k)} + \sum_{i=1}^{k-1} \textbf{A}\textbf{H}^{(k-1)}\Theta^{(k)}) \tag{11}$$

**식 (11)**을 보면 GCN[^11]과 유사한 것을 알 수 있다. 차이가 있다면 NN4G는 unnormalized adjacency matrix $$(\textbf{A})$$를 사용한다는 것이다. 하지만 $$\textbf{A}$$를 사용하게 되는 경우 hidden state node가 서로 다른 scale이 될 수 있는 단점이 있다. 

- **[Unofficial Pytorch](https://github.com/EmanueleCosenza/NN4G)** 

**2. Contextual Graph Markov Model, CGMM (2018)[^17]**

CGMM은 NN4G의 아이디어를 기반으로 제안한 확률 모델이다. CGMM의 장점으로는 확률적인 해석이 가능하다는 장점이 있다. Survey 논문에서는 CGMM에 대해 크게 다루지 않았으므로 추가적으로 알고자 한다면 논문을 참고하는 것이 좋을 듯 하다.

- **[Author github (Pytorch)](https://github.com/diningphil/CGMM)** 

**3. Diffusion Convolutional Neural Network, DCNN (2016)[^16]**

DCNN은 graph convolution 과정을 "diffusion process"로서 나타낸다. 여기서 diffusion process 란 하나의 노드에서 이웃 노드로 information을 전달할 때 특정 확률을 기반으로 전달하는 방법을 말한다. 이러한 과정을 여러번 하다보면 특정 지점에서 수렴하게 되는 시점이 온다. DCNN은 **식 (12)**와 같이 정의한다. 

$$\textbf{H}^{(k)}=f(\textbf{W}^{(k)}\odot\textbf{P}^k\textbf{X}) \tag{12}$$

여기서 $$f(\cdot)$$은 activation function이고 확률 matrix $$\textbf{P} \in \mathbb{R}^{n \times n}$$은 $$\textbf{P}=\textbf{D}^{-1}\textbf{A}$$로 계산된다. $$\textbf{P}$$에 대해 의미를 생각해보자면 각 노드의 연결된 이웃 노드에 대해 해당 노드의 degree로 나누어 $$1/degree(i)$$로 확률을 계산하였다고 생각할 수 있다. 

여기서 알아야 할 점은 $$\textbf{H}^{(k)}$$는 입력값 matrix인 $$\textbf{X}$$와 같은 차원의 matrix라는 점이고 $$\textbf{H}^{(k-1)}$$로 부터 계산된 hidden representation이 아니라는 것이다. DCNN은 최종 output으로 $$\textbf{H}^{(1)},\textbf{H}^{(2)},\cdots,\textbf{H}^{(K)}$$를 모두 concatenate하여 사용한다. 

**4. Diffusion Graph Convolution, DGC (2018)[^18]**

DGC는 **식 (13)**과 같이 DCNN과 다르게 concatenate 대신 결과 값의 합을 사용하여 output을 나타낸다. 

$$\textbf{H} = \sum_{k=0}^{K}f(\textbf{P}^k\textbf{X}\textbf{W}^{(k)}) \tag{13}$$

여기서 $$\textbf{W}^{(k)} \in \mathbb{R}^{D \times F}$$이고 $$f(\cdot)$$은 activation function이다. 확률 matrix의 승을 사용한다는 의미는 멀리 있는 이웃 노드의 information이 중심 노드에 적게 영향을 주기 위함이라고 볼 수 있다.

- **[Author github (Tensorflow)](https://github.com/liyaguang/DCRNN)** 


**5. PGC-DGCNN (2018)[^19]**

PGC-DGCNN은 shortest path를 기반으로 먼 이웃의 기여도(contribution)을 키우는 방법이다. 여기서 사용되는 shortest path adjacency matrix는 $$\textbf{S}^{(j)}$$로 정의하고 node $$v$$에서 node $$u$$까지의 shortest path가 $$j$$라면 $$\textbf{S}_{v,u}^{(j)}=1$$ 아니면 $$0$$으로 한다. PGC-DGCNN에는 receptive field size를 조정하는 hyperparameter $$r$$이 있다. PGC-DGCNN은 **식 (14)**와 같이 나타낸다.

$$\textbf{H}^{(k)} = \|_{j=0}^{r} f((\tilde{\textbf{D}}^{(j)})^{-1}\textbf{S}^{(j)}\textbf{H}^{(k-1)}\textbf{W}^{(j,k)}) \tag{14}$$

여기서 $$\tilde{\textbf{D}}_{ii}^{(j)}=\sum_{l} \textbf{S}_{i,l}^{(j)}$$이고 $$\textbf{H}^{(0)}=\textbf{X}$$이다. $$\|$$은 vector의 concatenation을 말한다. 

앞서 말한 '먼 이웃의 기여도를 키운다'는 **식 (14)**에서 볼 수 있듯이 receptive field의 크기에 따라 중심 노드로 부터 멀리 떨어진 노드도 함께 고려할 수 있다는 말이다. 하지만 PGC-DGCNN의 단점으로는 $$\textbf{S}^{(j)}$$를 계산하는데 드는 시간 복잡도가 $$O(n^3)$$이라는 점이다.

- **[Unofficial Pytorch](https://github.com/dinhinfotech/PGC-DGCNN)** 


**6. Partition Graph Convolution, PGC (2018)[^20]**

PGC는 PGC-DGCNN과 다르게 단지 shortest path 만을 기준으로 하는 것이 아닌 특정한 기준(criterion)을 정해서 이웃 노드를 $$Q$$개의 그룹으로 나눠준다. 때문에 PGC에서는 $$Q$$ adjacency matrix를 각각의 그룹별로 정의해서 사용한다. 그후 PGC의 연산과정을 앞서 소개한 GCN[^11]과 동일하다. $$Q$$ 그룹 간의 결과값은 합으로 계산하여 산출한다. PGC는 **식 (15)**와 같이 나타낼 수 있다.

$$\textbf{H}^{(k)} = \sum_{j=1}^{Q} \bar{\textbf{A}}^{(j)}\textbf{H}^{(k-1)}\textbf{W}^{(j,k)} \tag{15}$$

여기서 $$\textbf{H}^{(0)} = \textbf{X}$$, $$\bar{\textbf{A}}^{(j)} = (\tilde{\textbf{D}}^{(j)})^{-1/2}\tilde{\textbf{A}}^{(j)}(\tilde{\textbf{D}}^{(j)})^{-1/2}$$ 그리고 $$\tilde{\textbf{A}}^{(j)} = \textbf{A}^{(j)} + \textbf{I}$$이다. 

- **[Author github (Pytorch)](https://github.com/open-mmlab/mmskeleton)**

**7. Message Passing Neural Network, MPNN (2017)[^21]** 

MPNN은 spatial 기반의 ConvGNNs을 메인으로 하면서 graph convolution 과정을 노드에서 다른 노드로 information을 edge를 통해 바로 전달하게 하는 "message passing process"로써 나타낸 방법이다. MPNN은 $$K$$-step message passing을 반복한다. Message passing function는 **식 (16)**과 같이 정의할 수 있다.

$$\textbf{h}_v^{(k)} = U_k(\textbf{h}_v^{(k-1)}, \sum_{u \in N(v)} M_k(\textbf{h}_v^{(k-1)}, \textbf{h}_u^{(k-1)}, \textbf{x}_{vu}^e)) \tag{16}$$

여기서 $$\textbf{h}_v^{(0)} = \textbf{x}_v$$이고 $$U_k(\cdot)$$와 $$M_k(\cdot)$$은 학습 파라미터가 있는 function이다. 각 노드에 대해 hidden representation을 계산한 후에는 이 값에 output layer를 붙여서 node-level prediction 문제를 풀 수도 있고 readout function을 사용해서 전체 graph-level prediction 문제를 풀 수도 있다. 여기서 "readout function"이란 노드의 hidden representation을 통해 전체 graph의 representation을 뽑는 것을 말한다. readout function은 **식 (17)**과 같이 정의할 수 있다.

$$\textbf{h}_G = R(\textbf{h}_v^{(K)}|v\in G) \tag{17}$$

여기서 $$R(\cdot)$$은 학습 파라미터가 있는 readout function을 말한다. MPNN은 $$U_k(\cdot)$$, $$M_k(\cdot)$$, 그리고 $$R(\cdot)$$을 다른 형태로 나타내어 다른 GNNs에 적용할 수 있다. 하지만 미리 학습된 graph embedding 으로 다른 graph structure에 사용할 수 없다는 단점이 있다.

- **[Author github (Tensorflow)](https://github.com/brain-research/mpnn)** 

**8. Graph Isomorphism Network, GIN (2019)[^22]**

GIN은 중심 노드에 학습 parameter인 $$\epsilon^{(k)}$$를 더하여 이 값을 조정하며 MPNN의 단점을 보완하였다. GIN은 **식 (18)**과 같이 나타낼 수 있다.

$$\textbf{h}_v^{(k)} = MLP((1+\epsilon^{(k)})\textbf{h}_v^{(k-1)} + \sum_{u \in N(v)} \textbf{h}_u^{(k-1)}) \tag{18}$$

논문에서는 GIN에 대한 내용은 이게 전부지만 $$\epsilon^{(k)}$$에 대하여 수식을 통해 이해한 바로는 주변 이웃의 수가 많아짐에 따라 중심 노드의 information이 사라질 수 있으니 $$\epsilon^{(k)}$$를 조정하여 중심 노드의 information 값을 키우는 것이 아닌다 싶다.

- **[Author github (Pytorch)](https://github.com/weihua916/powerful-gnns)** 

**9. GraphSAGE (2017)[^23]**

GraphSAGE는 각 노드마다 이웃 노드의 수가 다양하게 존재하는데 모든 이웃 노드를 고려하는 것은 비효율적이기 때문에 일정한 이웃 노드 수를 고정하여 학습하는 방법이다. GraphSAGE의 graph convolution은 **식 (19)**를 통해 정의할 수 있다. 

$$\textbf{h}_v^{(k)} = \sigma(\textbf{W}^{(k)} \cdot f_k(\textbf{h}_v^{(k-1)}, \{ \textbf{h}_u^{(k-1)}, \forall u \in S_{N(v)} \})) \tag{19}$$

여기서 $$\textbf{h}_v^{(0)}=\textbf{x}_v$$이고 $$f_k(\cdot)$$은 aggregation function, 그리고 $$S_{N(v)}$$은 노드 $$v$$의 이웃 노드에 대한 샘플을 말한다. 이때 aggregation function은 mean, sum, 또는 max function과 같이 노드의 순서가 바뀌어도 같은 값이 나오도록 하는 invariant여야 한다. 

- **[Author github (Tensorflow)](https://github.com/williamleif/GraphSAGE)** 

**10. Graph Attention Network, GAT (2017)[^24]**

GAT는 중심 노드에 대한 이웃 노드의 기여도를 계산하는 방법이 이웃 노드의 샘플을 정하는 GraphSAGE[^23]나 GCN[^11]과 같이 미리 고정된(pre-determined) 이웃 노드의 기여도와는 다르다. GAT는 일반적으로 사용되는 attention mechanisms을 사용한다. 단, graph에서는 연결된 두 노드의 상대적인 가중치를 학습하는 방식을 사용한다. GAT는 **식 (20)**과 같이 정의할 수 있다.

$$\textbf{h}_v^{(k)} = \sigma(\sum_{u \in N(v) \cup v} \alpha_{vu}^{(k)} \textbf{W}^{(k)} \textbf{h}_u^{(k-1)}) \tag{20}$$

여기서 $$\textbf{h}_v^{(0)} = \textbf{x}_v$$이다. Attention weight $$(\alpha_{vu}^{(k)})$$는 노드 $$v$$와 이웃 노드 $$u$$와의 connective strength를 **식 (21)**과 같이 계산한다.

$$\alpha_{vu}^{(k)} = softmax(g(\textbf{a}^T[\textbf{W}^{(k)}\textbf{h}_v^{(k-1)} \| \textbf{W}^{(k)}\textbf{h}_u^{(k-1)})) \tag{21}$$

여기서 $$g(\cdot)$$은 LeakyReLU activation function 이고 $$\textbf{a}$$는 학습 parameter이다. Softmax function은 노드 $$v$$의 모든 이웃의 attention weigth 합이 1이 되게하기 위해 사용하였다. GAT는 또한 multi-head attention을 사용하여 모델 성능을 더욱 향상 시킬 수 있고 실제로도 GraphSAGE 보다 node classification 문제에서 더 좋은 성능을 나타내었다. 

- **[Author github (Tensorflow)](https://github.com/PetarV-/GAT)** 
- **[Unofficial Pytorch](https://github.com/Diego999/pyGAT)**

**11. Gated Attention Network, GaAN (2018)[^25]**

GAT에서는 각각의 attention head의 기여도를 동일하게 적용하였지만 GaAN에서는 self-attention mechanism을 추가하여 각 attention head에 추가로 attention score를 계산하였다. 

- **[Author github (MXNet)](https://github.com/jennyzhang0215/GaAN)** 

**12. GeniePath (2019)[^26]**

GeniePath는 LSTM 같은 gate mechanism을 통해 information을 조절하여 사용하는 방법이다. 별다른 소개가 더는 없어서 만약 내용이 궁금하다면 논문을 보는 것이 좋을 듯 하다.

- **[Unofficial Pytorch](https://github.com/shawnwang-tech/GeniePath-pytorch)**

**13. Mixture Model Network, MoNet (2017)[^27]**

MoNet은 노드의 이웃에 각각 다른 가중치를 주는 방법이다. 각 노드와 노드의 이웃들에 대한 상대적인 위치(relative position)를 나타낼 수 있는 'pseudo-coordinates'를 추가하여 위치에 따른 가중치를 부여할 수 있도록 학습하는 방법이다. 두 노드의 상대적인 위치가 주어지면 가중치 function은 각 위치에 맞는 가중치를 매핑하게 된다. 이러한 방식으로 graph filter의 가중치는 서로 다른 위치에 따른 가중치를 공유한다. 

Pseudo-coordinates는 아마 transformer[^52]에서 사용되는 position-embedding 이나 CoordConv[^53]에서 사용되는 coordinate convolution과 같은 의미가 아닐까 싶다.

- **[Unofficial Tensorflow](https://github.com/HeapHop30/graph-attention-nets)** 
- **[Unofficial Pytorch](https://github.com/theswgong/MoNet)** 

**14. PATCHY-SAN (2016)[^28]**

PATCHY-SAN은 서로 다른 위치에 따라 가중치를 주는 방법 중 하나이다. PATCHY-SAN은 특정한 기준과 학습 parameter에 순위를 매겨서 이웃 노드들의 순서를 정하여 서로 다른 가중치를 주는 방법이다. 노드의 순서는 "graph labeling"과 상위 $$q$$개의 이웃을 선정하여 결정한다. 여기서 graph labeling은 노드 degree, centrality, 그리고 Weisfeiler-Lehman color[^54]에 의해 정해진다. Weisfeiler-Lehman Algorithm은 두 그래프가 있을 때 isomorphic인지 아닌지 test하는 방법이기도 하고 또는 naive vertex classification에 사용 되기도 한다. 

각 노드는 graph labeling 을 통해 $$q$$개의 정렬된 이웃 노드를 통해 graph-structure를 grid-structure로 변환할 수 있다. 이제 정렬된 이웃 노드의 순서에 맞게 1D convolutional filter를 사용하여 이웃 노드의 feature information을 aggregate 할 수 있다. 

PATCHY-SAN의 단점으로는 ranking 기준이 graph structure에만 적용된다는 점이고 이는 연산량이 크다는 단점이 있다.

- **[Unofficial Keras](https://github.com/tvayer/PSCN)** 

**15. Large-scale Graph Convolutional Network, LGCN (2018)[^29]**

LGCN은 노드의 feature information을 기준으로 이웃 노드의 순위를 정한다. LGCN의 feature matrix는 이웃 노드로 구성되어 있고 열을 기준으로 정렬하여 상위 $$q$$개를 중심 노드의 입력값으로 사용한다. 

- **[Author github (Tensorflow)](https://github.com/divelab/lgcn)** 

# Graph AutoEncoders (GAEs)

GAEs는 노드를 laten feature space로 encoding 하고 다시 latent space의 representation을 decoding하는 방법이다. GAEs는 network embedding을 학습하거나 새로운 graph를 생각하기 위해 사용된다. 여기서는 Network Embedding과 Graph Generation 크게 두 가지로 나누어 리뷰한다.

## Network Embedding

<p align='center'>
    <img width='400' src='https://user-images.githubusercontent.com/37654013/106140920-bf826980-61b2-11eb-8bdd-7accec619086.png'><br>그림 8. GAEs의 Network Embedding Roadmap
</p>

Network embedding은 노드의 topological information을 가지면서 저차원의 vector representation으로 학습하는 방법이다. GAEs는 encoding을 사용하여 network embedding을 추출하고 다시 decoding을 사용하여 PPMI matrix나 adjacency matrix와 같이 graph의 topological information을 복원한다. 

**1. Deep Neural Network for Graph Representation, DNGR (2016)[^30]**

DNGR은 "stacked denoising autoencoder"를 사용하여 encoding하고 MLP를 통해 PPMI로 decoding하는 방법이다. 

- **[Author github (Tensorflow)](https://github.com/ShelsonCao/DNGR)** 

**2. Structural Deep Network Embedding, SDNE (2016)[^31]**

SDNE는 'node first-order proximity'와 'second-order proximity'를 고려하여 "stacked autoencoder"를 사용한다. 

SDNE는 encoder와 decoder 각각 output에 대한 loss function을 계산한다. Encoder의 loss function은 노드의 network embedding과 이웃 노드의 representation의 차이인 node first-order proximity를 최소화하며 학습한다. Encoder의 loss function $$(L_{1st})$$는 **식 (22)**와 같이 정의한다.

$$L_{1st} = \sum_{(v,u) \in E} \textbf{A}_{v,u} \| enc(\textbf{x}_v) - enc(\textbf{x}_u) \|^2 \tag{22}$$

여기서 $$\textbf{x}_v = \textbf{A}_{v,:}$$이고 $$enc(\cdot)$$은 MLP로 구성된 encoder를 말한다. 

Decoder의 loss function은 노드의 입력값과 reconstruct된 입력값 간의 차이인 node second-order proximity를 최소화하며 학습하게 한다. Decoder의 loss function은 **식 (23)**과 같이 정의한다.

$$L_{2nd} = \sum_{v \in V} \| (dec(enc(\textbf{x}_v)) - \textbf{x}_v) \odot \textbf{b}_v \|^2 \tag{23}$$

여기서 $$\textbf{A}_{v,u} = 0$$인 경우 $$b_{v,u}=1$$,  $$\textbf{A}_{v,u} = 1$$인 경우 $$b_{v,u}=\beta > 1$$이다. $$dec(\cdot)$$은 MLP로 구성된 decoder이다. 

DNGR과 SDNE의 단점은 노드의 information은 무시한 채 두 노드 간의 연결성(connectivity), 즉 노드 구조만 고려한다는 점이다. 

- **[Unofficial Tensorflow](https://github.com/shenweichen/GraphEmbedding)** 

**3. Graph Autoencoder, GAE (2016)[^32]**

GAE$$*$$는 GCN[^11]을 활용하여 노드 구조와 feature information을 모두 고려하여 encoding 한다. GAE$$*$$의 encoder는 **식 (24)**와 같이 두 개의 graph convolutional layer로 구성되어 있다. 

$$\textbf{X} = enc(\textbf{X}, \textbf{A}) = Gconv(f(Gconv(\textbf{A}, \textbf{X}; \Theta_1)); \Theta_2) \tag{24}$$

여기서 $$\textbf{Z}$$는 graph의 network embedding matrix를 말한다. $$f{\cdot}$$은 $$ReLU$$ activation function 그리고 $$Gconv(\cdot)$$은 **식 (8)**에서 정의한 graph convolutional layer 이다. GAE$$*$$의 decoder는 graph의 adjacency matrix를 target으로 하여 embedding 값에서 노드의 relational information을 decoding한다. Decoder는 **식 (25)**와 같이 나타낸다.

$$\hat{\textbf{A}}_{v,u} = dec(\textbf{z}_v, \textbf{z}_u) = \sigma(\textbf{z}_v^T \textbf{z}_u) \tag{25}$$

여기서 $$\textbf{z}_v$$는 노드 $$v$$의 embedding이다. GAE$$*$$는 real adjacency matrix $$(\textbf{A})$$와 reconstructed adjacency matrix $$(\hat{\textbf{A}})$$을 통해 negative cross entropy가 최소화 되는 방향으로 학습한다. 하지만 GAE$$*$$의 단점으로는 overfitting이 잘된다는 것이다.

- **[Author github (Tensorflow)](https://github.com/tkipf/gae)**

**4. Variational Graph Autoencoder, VGAE (2016)[^32]**

VGAE는 data의 분포를 학습하기 위한 GAE에서 변형된 방법이다. VGAE는 **식 (26)**과 같이 variational lower bound $$(L)$$을 최적화 한다.

$$L = E_{q(\textbf{Z} | \textbf{X}, \textbf{A})} [\log p(\textbf{A}|\textbf{Z})] - KL[q(\textbf{Z}|\textbf{X}, \textbf{A})\| p(\textbf{Z})] \tag{26}$$

여기서 $$KL(\cdot)$$은 Kullback-Leibler divergence function을 말한다. KL divergence는 두 분포를 비교하는 function이다. $$p(\textbf{Z})$$는 Gaussian prior이고 $$p(\textbf{Z}) = \prod_{i=1}^{n} p(\textbf{z}_i) = \prod_{i=1}^{n} N(\textbf{z}_i \mid 0, \textbf{I})$$로 정의한다. 다음으로 $$p(A_{ij} = 1 \mid \textbf{z}_i, \textbf{z}_j) = dec(\textbf{z}_i, \textbf{z}_j) = \sigma(\textbf{z}_i^T\textbf{z}_j)$$이고 $$q(\textbf{Z} \mid \textbf{X}, \textbf{A}) = \prod_{i=1}^{n} q(\textbf{z}_i \mid \textbf{X}, \textbf{A})$$이다. 여기서 $$q(\textbf{z}_i \mid \textbf{X}, \textbf{A}) = N(\textbf{z}_i \mid \mu_i, diag(\sigma_i^2))$$이다.

Mean vector $$\mu_i$$는 **식 (24)**에서 정의한 encoder output의 $$i^{th}$$ 행이고 $$\log \sigma_i$$는 $$\mu_i$$와 같이 다른 encoder로 계산된다. **식 (26)**에서 정의하듯이 VGAE는 empirical distribution $$q(\textbf{Z} \mid \textbf{X}, \textbf{A})$$가 prior distribution $$p(\textbf{Z})$$와 근사해지도록 학습한다. 

- **[Author github (Tensorflow)](https://github.com/tkipf/gae)** 

**5. GraphSAGE (2017)[^23]**

GraphSAGE는 spatial 기반 모델에서 설명했지만 network embedding에도 사용된다. GraphSAGE에서는 reconstruction error를 최적화 하는 것이 아닌, negative sampling을 통해 두 노드 간의 relational information을 최적화 한다. GraphSAGE의 embedding loss term은 **식 (27)**과 같이 나타낼 수 있다.

$$L(\textbf{z}_v) = -\log(\sigma(\textbf{z}_v^T \textbf{z}_u)) - QE_{v_n ~ P_n{(v)}}\log(\sigma(-\textbf{z}_v^T\textbf{z}_{v_n})) \tag{27}$$

여기서 노드 $$u$$은 노드 $$v$$의 이웃을 말하고 노드 $$v_n$$은 노드 $$v$$와 멀리 떨어진 노드를 말한다. 노드 $$v_n$$은 negative sampling distribution $$P_n(v)$$에서 추출한다. $$Q$$는 negative sample의 수 이다. $$\sigma(\cdot)$$은 sigmoid function을 말한다. **식 (27)**의 loss function에 대해 해석하자면 유사한 representation을 갖는 노드는 가깝게하고 다른 representation을 갖는 노드는 더 멀게 한다고 볼 수 있다. 다른 말로는 만약 유사한 노드가 서로 다른 representation을 갖게되면 내적값이 작아지게 되어 첫 번째 term의 값을 키우게 되고 서로 다른 노드가 유사한 representation을 갖게 되면 내적값이 커져서 두 번째 term의 값이 커지게 된다.

- **[Author github (Tensorflow)](https://github.com/williamleif/GraphSAGE)**

**6. Adversarially Regularized Variational Graph Autoencoder, ARVGA (2018)[^33]**

ARVGA는 GAN의 학습 방법을 사용한다. GAN의 학습 방식과 같이 ARVGA는 prior distribution $$p(\textbf{Z})$$과 구분하기 어렵도록 empirical distribution $$q(\textbf{Z} \mid \textbf{X}, \textbf{A})$$를 생성하여 encoder를 학습한다. 

- **[Author github (Tensorflow)](https://github.com/Ruiqi-Hu/ARGA)** 

**Deep Recursive Network Embedding, DRNE (2018)[^34]**

앞서 소개한 방법들은 대부분 기본적으로 link prediction 방식으로 network embedding을 학습한다. 하지만 graph가 sparse 한 경우, 노드는 많지만 서로 이웃이 많지 않은 경우 positive node가 negative node에 비해 훨씬 적다. 이러한 sparsity 문제를 풀기 위해 사용하는 방법이 random permutations 또는 random walks를 이용하여 graph를 sequence 형태로 변환하여 학습하는 것이다. Sequence로 데이터를 변환하면 기존 sequence에 적용한 딥러닝 모델을 적용할 수 있는 장점이 있다. 

DRNE는 노드의 network embedding과 노드의 이웃들에 대한 network embedding을 aggregation한 값을 근사하도록 학습한다. 이때 사용되는 방법은 LSTM이다. DRNE의 reconstruction error는 **식 (28)**과 같다.

$$L=\sum_{v \in V} \| \textbf{z}_v - LSTM(\{ \textbf{z}_u \mid u \in N(v)\})\|^2 \tag{28}$$

여기서 $$\textbf{z}_v$$는 노드 $$v$$의 network embedding이고 LSTM network는 노드 $$v$$의 이웃에 대한 random sequence를 이웃 노드의 degree를 기준으로 하여 정렬한다. 하지만 seqeuce 형태이다보니 노드 순서가 바뀌는 경우 LSTM network가 invariant하지 못하다는 단점이 있다.

- **[Author github (Tensorflow)](https://github.com/tadpole/DRNE)**

**7. Network Representations with Adversarially Regularized Autoencoders, NetRA(2018)[^35]**

NetRA는 graph encoder-decoder loss function을 **식 (29)**와 같이 제안하였다.

$$L = - E_{z~P_{data}}(\textbf{z})(dist(\textbf{z}, dec(enc(\textbf{z})))) \tag{29}$$

여기서 $$dist(\cdot)$$은 노드 embedding $$\textbf{z}$$와 reconstructed $$\textbf{z}$$와의 거리를 계산하는 function이다. NetRA의 encoder와 decoder는 노드 $$v \in V$$를 시작으로 하는 random walks를 사용한 LSTM network로 구성되어 있다. ARGVA와 마찬가지로 adversarial 학습을 통해 prior distribution를 기준으로 network embedding을 학습한다. NetRA에서는 LSTM network의 문제점인 permutation variant를 따로 해결하지는 않았지만 실험적으로 NetRA의 효과를 입증했다. 

- **[Author github (Pytorch)](https://github.com/chengw07/NetRA)**

**Deep Graph Infomax, DGI (2019)[^55]**

DGI는 global structural information을 위해 local mutual information을 최대화하여 local network embedding을 유도하는 방법이다. DGI는 node classification 문제에서 기존 graph network embedding 방법보다 좋은 성능을 내는 것 뿐만 아니라 supervised learning 보다 좋은 성능을 내었다.

- **[Author github (Pytorch)](https://github.com/PetarV-/DGI)**

## Graph Generation

<p align='center'>
    <img width='400' src='https://user-images.githubusercontent.com/37654013/106137131-a32ffe00-61ad-11eb-9981-e170c49d7394.png'><br>그림 9. GAEs의 Graph Generation Roadmap
</p>

**1. GrammarVAE (2017)[^37], Chemical-VAE. (2018)[^36], SD-VAE (2018)[^38]**  

이 세 가지 방법은 SMILES(simplifired molecular-input line-entry system)이라는 방법을 통해 encoder와 decoder에 각각 CNN과 RNN을 적용하였다. 이 방법은 분자 구조의 string representation을 생성하기 위한 방법으로 다소 domain에 특화되어 있긴 하다. 하지만 특정 기준에 수렴할때까지 반복적으로 node와 edge를 추가하며 graph를 키워나가는 방식으로 다양한 graph에 적용해 볼 수 있다.

- **[GrammarVAE Author github (Keras)](https://github.com/mkusner/grammarVAE)** 
- **[GrammarVAE Unofficial Pytorch](https://github.com/geyang/grammar_variational_autoencoder)** 

- **[Chemical-VAE Author github (Keras)](https://github.com/aspuru-guzik-group/chemical_vae)**

- **[SD-VAE Author github (Pytorch)](https://github.com/Hanjun-Dai/sdvae
)**


**2. Deep Generative Model of Graphs, DeepGMG (2018)[^39]**

DeepGMG는 먼저 **식 (30)**과 같이 모든 노드의 permutation에 대한 확률 합으로 graph의 확률을 계산한다.

$$p(G) = \sum_{\pi} p(G, \pi) \tag{30}$$

여기서 $$\pi$$는 노드의 순서를 말한다. **식 (30)**의 그래프에 대한 확률은 모든 노드와 edge에 대한 complex joint probability를 나타낸다. DeepGMG는 sequence of decision을 통해 노드를 추가할지 말지, edge를 더할지 말지 그리고 새로운 노드와 노드를 연결할지 말지를 정한다. 여기서 노드와 edge를 만들어내는 과정은 각 노드의 state를 보고 결정한다. Graph를 키워나가기 위한 graph state는 RecGNN을 통해 업데이트 한다.

- **[Unofficial Pytorch](https://github.com/JiaxuanYou/graph-generation)**
 
**3. GraphRNN (2018)[^40]**

GraphRNN에서는 각 노드와 edge를 생성하기 위해 graph-level RNN과 edge-level RNN을 사용한다. 우선 edge-level RNN이 이전 sequence에서 생성된 노드에 새로운 노드를 연결할지 나타내는 binary sequence를 만들고 graph-level RNN은 매번 node sequence에 새로운 node를 추가한다. 

- **[Author github (Pytorch)](https://github.com/JiaxuanYou/graph-generation)** 

**4. Graph Variational Autoencoder, GraphVAE (2018)[^41]**

GraphVAE는 앞서 소개한 방법들과 달리 한방에 graph를 뙇! 하고 생성하는 방법이다. GraphVAE는 각각의 노드와 edge가 독립변수라고 가정한다. Encoder를 통해 생성된 posterior distribution $$q_\phi (\textbf{z} \mid G)$$와 decoder를 통해 생성된 generative distribution $$p_\theta(G \mid \textbf{z})$$를 가정으로 하여 **식 (31)**을 통해 variational lower bound를 최적화 한다. 

$$L(\phi, \theta ; G) = E_{q_\phi(z \mid G)}[-\log p_theta (G \mid \textbf{z})] + KL[q_\phi(\textbf{z} \mid G) \| p(\textbf{z})] \tag{31}$$

여기서 $$p(\textbf{z})$$는 Gaussian prior를 말하고 $$\phi$$와 $$\theta$$는 학습 parameter를 말한다. GraphVAE에서 encoder는 ConvGNN을 사용하고 decoder에서는 MLP를 사용한다. GraphVAE의 output으로는 adjacency matrix, node attributes, 그리고 edge attributes로 세 가지이다. 

GraphVAE는 한번에 graph representation을 만들어내지만 생성된 graph에 대한 graph connectivity, validity, 그리고 node compatibility를 모두 고려하는 건 역시나 어렵고 앞으로 풀어나가야할 문제다.

- **[Unofficial Pytorch](https://github.com/JiaxuanYou/graph-generation)** 

**5. Regularized Graph Variational Autoencoder, RGVAE (2018)[^42]**

RGVAE는 앞서 언급한 challenge 중 validity에 대해 decoder의 output distribution을 정규화하여 graph variational autoencoder에 제약 조건을 추가한 방법이다. 

- **[Unofficial Pytorch](https://github.com/INDElab/rgvae)** 

**6. Molecular Generative Adversarial Network, MolGAN (2018)[^43]**

MolGAN은 ConvGNN[^56], GAN[^57], 그리고 강화 학습 목적식을 추가하여 그래프를 생성하는 방법이다. MolGAN은 generator와 discriminator로 구성되어 있고 서로 번갈아 가면서 학습하며 generator의 authenticity를 높이는 방법이다. 또한 강화 학습 방식을 적용했기 때문에 추가적인 evaluator를 넣어서 이에 맞는 기준을 충족하도록 graph를 생성하기 위해 discriminator와 함께 병렬적으로 reward network를 적용했다. 

- **[Author github (Tensorflow)](https://github.com/nicola-decao/MolGAN)** 
- **[Unofficial Pytorch](https://github.com/yongqyu/MolGAN-pytorch)**

**7. NetGAN (2018)[^44]**

NetGAN은 random-walks를 기반으로 하여 graph를 생성하기 위해 LSTM과 Wasserstein GAN[^58]을 적용하였다. NetGAN은 앞선 방식들과 다르게 generator는 LSTM network를 통해 random walks를 생성하고 discriminator는 생성된 가짜 random walks와 진짜 random walks를 구분하도록 학습된다. 학습이 끝난 후에는 generator로 생성된 random walks를 사용하여 node의 co-occurrence matrix를 normalize하여 새로운 graph를 만들어낸다. 

- **[Author github (Tensorflow)](https://github.com/danielzuegner/netgan)** 

**추가 정보** 

그래프를 순서대로 나열하는 Sequential 방법들은 graph 구조를 sequence로 선형화 하지만 graph 구조의 cycle 때문에 structural information에 손실이 생긴다. 

그래프를 한방에 뙇! 만드는 global 방법들은 단계를 거치지 않고 한번에 만들 수 있지만 사이즈가 큰 데이터에는 시간이 기본적으로 GAE의 output space가 $$O(n^2)$$이기 때문에 적용하기 힘들다는 단점이 있다.

# Spatial-Temporal Graph Neural Networks (STGNNs)

<p align='center'>
    <img width='400' src='https://user-images.githubusercontent.com/37654013/106140985-d5902a00-61b2-11eb-8b0f-c98c511f6300.png'><br>그림 10. STGNNs Roadmap
</p>

STGNNs은 graph의 dynamicity를 나타내기 위한 방법들을 나타내는 말이다. STGNNs의 방법들은 연결된 노드 간의 상호의존성(interdependency)를 가정으로하여 dynamic node 입력값을 모델링 하는 것을 목적으로 한다. 이 말이 무슨말인가 직관적으로 다가오지 않기 때문에 survey 논문에서는 친절하게 예시를 들어 주었다. 

예를 들어 traffic network는 거리에 설치된 속도 감지 센서로 구성되어 있다. 여기서 각 센서 간의 거리는 edge weight를 나타낸다. 어떤 거리의 traffic 상황은 그 거리와 인접한 다른 거리의 상태에 의존하게 된다. 즉, traffic speed를 예측하면서 spatial 정보를 같이 고려해야 한다. 이러한 문제를 풀기 위해 STGNNs은 graph의 sptial과 temperal dependency를 모두 고려한다. STGNNs의 task는 미래의 node value와 label을 예측하거나 spatial-temperal graph label을 예측하는 것이다. 

STGNNs은 RNN 기반의 방법들과 CNN 기반의 방법으로 크게 두 가지 방향성이 있다. RNN 기반 방법들은 graph convolution을 통해 reccurent unit에 입력값과 hidden state를 filtering하여 spatial-temporal dependency를 나타낸다. 간단한 RNN 형태로는 **식 (32)**와 같이 정의할 수 있다.

$$\textbf{H}^{(t)} = \sigma(\textbf{W}\textbf{X}^{(t)} + \textbf{U}\textbf{H}^{(t-1)} + \textbf{b}) \tag{32}$$

여기서 $$\textbf{X}^{(t)} \in \mathbb{R}^{n \times d}$$는 시간 $$t$$에서의 node feature matrix를 말한다. 여기에 graph convolution을 적용하면 **식 (33)**과 같이 나타낼 수 있다. 

$$\textbf{H}^{(t)} = \sigma(Gconv(\textbf{X}^{(t)}, \textbf{A}; \textbf{W}) + Gconv(\textbf{H}^{(t-1)}, \textbf{A}; \textbf{U}) + \textbf{b}) \tag{33}$$

여기서 $$Gconv(\cdot)$$은 graph convolution layer이다. 

## CNN & RNN based model

**1. Graph Convolutional Recurrent Network, GCRN(2018)[^45]** 

GCRN은 ChebNet[^10]과 LSTM network를 결합한 방법이다. 

- **[Author github (Tensorflow)](https://github.com/youngjoo-epfl/gconvRNN)** 
- **[Unofficial Pytorch](https://github.com/benedekrozemberczki/pytorch_geometric_temporal)**

**2. Diffusion Convolutional Recurrent Neural Network, DCRNN(2018)[^46]**  

DCRNN은 diffusion graph convolutional layer인 **식 (13)**과 GRU network를 통합한 방법이다. 또한, DCRNN은 node value의 $$K$$ step을 예측하기 위해 encoder-decoder 방식을 적용하였다. 

- **[Author github (Tensorflow)](https://github.com/liyaguang/DCRNN)** 
- **[Unofficial Pytorch](https://github.com/benedekrozemberczki/pytorch_geometric_temporal)** 


## RNN based model

**1. Structural-RNN (2016)[^47]** 

Structural-RNN은 time step 마다 node label을 예측하기 위해 recurrent framework를 적용했다. 이때 사용된 recurrent framework에는 node-RNN과 edge-RNN이 사용된다. 각 노드와 edge의 temporal information은 node-RNN과 edge-RNN에 따로따로 들어간다. Temporal 이외에 spatial information을 표현하기 위해 edge-RNN의 output은 node-RNN의 입력값으로 들어간다. 단점으로는 각각의 node와 edge가 서로 다른 RNN을 사용하다보니 model complextity가 높아질 수 밖에 없다는 점이다. 하지만 이 문제는 node와 edge를 semantic group으로 나누어 같은 group에 속한 node와 edge는 같은 RNN 모델을 공유하여 연산량을 줄일 수 있다. 

- **[Author github (Theano)](https://github.com/asheshjain399/RNNexp/tree/master/structural_rnn)** 

## CNN based model

RNN 기반의 방법들은 iterative propagation 과정에서 시간이 오래걸린다는 것와 gradients의 vanishing 또는 exploding 한다는 단점이 있다. 하지만 CNN 기반 방법들은 non-recursive 방법을 통해 병렬처리, 안정된 gradients, 그리고 낮은 메모리 소비라는 장점으로 spatial-temporal graph에 사용될 수 있다. 

**그림 (2). d**와 같이 CNN 기반 모델들은 graph convolution layer에 1D-CNN를 사용하여 temporal과 spatial dependency를 각각 학습한다. Spatial-temporal GNN의 입력값이 $$\textbf{X} \in \mathbb{R}^{T \times n \times d}$$라고 가정할 때 1D-CNN layer는 시간 축으로 $$\textbf{X}_{[:,i,:]}$$을 통과시켜서 각 노드에 대한 temporal information을 aggregation한다. 한편 매번 time step 마다는 $$\textbf{X}_{[i,:,:]}$$에 graph convolution layer를 적용하여 spatial information을 aggregate한다. 

**1. CGCN (2018)[^48]** 

CGCN은 1D convolution layer를 ChebNet[^10] 또는 GCN[^11]에 적용한 방법이다. 이때 spatial-temporal block을 각 sequential order에 따라 gated 1D convolutional layer, graph convolutional layer, 그리고 다시 gated 1D convolutional layer를 쌓아서 구성하였다. 

- **[Author github (Tensorflow)](https://github.com/VeritasYin/STGCN_IJCAI-18)** 
- **[Unofficial Pytorch](https://github.com/FelixOpolka/STGCN-PyTorch)**
- **[Unofficial MXNet](https://github.com/Davidham3/STGCN)**

**2. ST-GCN (2018)[^49]** 

ST-GCN은 spatial-temporal block을 PGC-layer **식 (15)**와 1D convolutional layer를 사용하여 구성하였다.

- **[Author github (Pytorch)](https://github.com/open-mmlab/mmskeleton)** 

**3. Graph-WaveNet (2019)[^50]**

앞서 소개한 CNN 기반 방법들은 사전에 정의된 graph structure가 노드 간의 genuine dependency relationship을 반영한다는 것을 가정으로 하였다. 하지만 spatial-temporal 환경에서는 graph data의 snapshots(전체가 아닌 일부분)만으로도 data로부터 latent static graph structure를 알아서 학습하는 것이 가능하다. 즉, 전체 graph에 대한 정보(예를 들면 adjacency matrix)가 없어도 graph structure를 학습할 수 있다.

Graph WaveNet은 graph convolution에 self-adaptive adjacency matrix를 적용하여 이를 구현한 방법이다. Self-adaptive adjacency matrix는 **식 (34)**와 같이 정의한다.

$$\textbf{A}_{adp} = softmax(ReLU(\textbf{E}_1\textbf{E}_2^T)) \tag{34}$$

여기서 $$softmax$$ function은 행마다 적용한다. $$\textbf{E}_1$$은 source node embedding(시작 노드의 embedding)이고 $$\textbf{E}_2$$는 target node embedding(시작 노드로부터 연결된 노드 embedding)을 말한다. 각각은 학습 파라미터를 함께 적용하여 학습한다. $$\textbf{E}_1 \textbf{E}_2^T$$는 두 노드 간의 dependency weight를 말한다. 이러한 과정을 통해 Graph WaveNet은 adjacency matrix가 주어지지 않아도 좋은 성능을 낼 수 있다.

- **[Author github (Pytorch)](https://github.com/nnzhan/Graph-WaveNet)**

## Attention based model

Latent static spatial dependency를 학습하는 것은 network의 서로 다른 노드 간의 해석가능성이나 상관관계를 나타낼 수 있다. 하지만 특정한 상황에서는 latent dynamic spatial dependency를 학습하는 것이 모델의 정확도를 더 향상시키기도 한다. 예를 들어 traffic network에서 두 도로 사이의 travel time은 현재 traffic 상황에 따라 다르다. 

**4. Gated Attention Network, GaAN (2018)[^25]**

GaAN은 spatial 기반 GCN 모델에서도 설명했지만 spatial-temporal data에도 적용할 수 있다. GaAN은 RNN 기반 방법을 사용하여 dynamic spatial dependency를 학습하기 위해 attention mechanism을 적용했다. Attention function은 현재 시점의 입력값이 주어진 상황에서 서로 연결된 두 노드 간의 edge weight를 업데이트 한다. 

- **[Author github (MXNet)](https://github.com/jennyzhang0215/GaAN)** 

**5. ASTGCN (2019)[^51]**

ASTGCN은 CNN 기반의 방법을 적용하여 sptial attention function과 temporal attention function을 적용하여 latent dynamic spatial dependency와 temporal dependency를 학습한다. 하지만 latent spatial dependency를 학습하기 위해서는 연결된 두개의 노드 마다 spatial dependency weight를 계산하기 위해 $$O(n^2)$$만큼의 시간 복잡도가 든다는 단점이 있다. 

- **[Author github (Pytorch)](https://github.com/guoshnBJTU/ASTGCN-r-pytorch)**

# Benchmark Dataset

GNN을 평가하기 위한 benchmark dataset은 **표 (2)**에서 볼 수 있듯이 크게 Citation Network, Bio-chemical Graphs, Social Networks 등등 있다. 

<p align='center'>
    <img src='https://user-images.githubusercontent.com/37654013/105983406-0ea99b00-60dc-11eb-8fa0-9bde4e34e8a3.png'><br>표 2. GNN에서 사용되는 benchmark dataset과 각 dataset 마다 속성값
</p>

이외에도 다양한 benchmark dataset이 있다. Python에서는 graph data를 다루기 위한 여러 framework가 존재하는데 그 중 대표적으로 Pytorch를 기반으로 하는 `Pytorch Geometric` 과 Stanford에서 만든 `SNAP` 이 두 framework는 다양한 dataset를 제공한다.

- `Pytorch` Geometric에서 제공하는 [dataset](https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html)
- `SNAP`에서 제공하는 [Large dataset](https://snap.stanford.edu/data/)
- `SNAP`에서 제공하는 [Web and Blog dataset](https://snap.stanford.edu/data/other.html)
- `SNAP`에서 제공하는 [기타 등등 dataset](https://snap.stanford.edu/data/links.html)

# 맺음말

수 많은 논문을 단지 survey로만 보고 모두 이해할 수는 없지만 graph data를 어떻게 neural network으로 활용하는지 알아볼 수 있는 좋은 시간이였다. 중간중간 단순히 말만으로는 이해하기 힘든 부분이 많지만... 특히 spectral의 경우... 그래도 처음 GNN을 공부하기로 했을 때의 안개 같았던 앞 길이 조금 맑아진 기분이다. 하지만 길의 끝이 안보이는 건 똑같다. 

이제 GNN에 대해 공부하는 아주 샛병아리 수준의 G린이 지만 이제는 graph data도 다뤄볼 수 있다는 사실이 참으로 새롭고 재밌다. 

# Reference
[^1]: Wu, Z., Pan, S., Chen, F., Long, G., Zhang, C., & Philip, S. Y. (2020). A comprehensive survey on graph neural networks. IEEE Transactions on Neural Networks and Learning Systems.
[^2]: A. Sperduti and A. Starita, “Supervised neural networks for the classification of structures,” IEEE Transactions on Neural Networks, vol. 8, no. 3, pp. 714–735, 1997.
[^3]: M. Gori, G. Monfardini, and F. Scarselli, “A new model for learning in graph domains,” in Proc. of IJCNN, vol. 2. IEEE, 2005, pp. 729–734.
[^4]: F. Scarselli, M. Gori, A. C. Tsoi, M. Hagenbuchner, and G. Monfardini, “The graph neural network model,” IEEE Transactions on Neural Networks, vol. 20, no. 1, pp. 61–80, 2009.
[^5]: C. Gallicchio and A. Micheli, “Graph echo state networks,” in IJCNN. IEEE, 2010, pp. 1–8.
[^6]: Y. Li, D. Tarlow, M. Brockschmidt, and R. Zemel, “Gated graph sequence neural networks,” in Proc. of ICLR, 2015.
[^7]: H.Dai,Z.Kozareva,B.Dai,A.Smola,andL.Song,“Learning steady - states of iterative algorithms over graphs,” in Proc. of ICML, 2018, pp. 1114–1122.
[^8]: J. Bruna, W. Zaremba, A. Szlam, and Y. LeCun, “Spectral networks and locally connected networks on graphs,” in Proc. of ICLR, 2014.
[^10]: M. Defferrard, X. Bresson, and P. Vandergheynst, “Convolutional neural networks on graphs with fast localized spectral filtering,” in Proc. of NIPS, 2016, pp. 3844–3852.
[^11]: T. N. Kipf and M. Welling, “Semi-supervised classification with graph convolutional networks,” in Proc. of ICLR, 2017.
[^12]: R. Levie, F. Monti, X. Bresson, and M. M. Bronstein, “Cayleynets: Graph convolutional neural networks with complex rational spectral filters,” IEEE Transactions on Signal Processing, vol. 67, no. 1, pp. 97–109, 2017.
[^13]: R. Li, S. Wang, F. Zhu, and J. Huang, “Adaptive graph convolutional neural networks,” in Proc. of AAAI, 2018, pp. 3546–3553.
[^14]: C. Zhuang and Q. Ma, “Dual graph convolutional networks for graph - based semi-supervised classification,” in WWW, 2018, pp. 499–508.
[^15]: A. Micheli, “Neural network for graphs: A contextual constructive approach,” IEEE Transactions on Neural Networks, vol. 20, no. 3, pp. 498–511, 2009.
[^16]: J. Atwood and D. Towsley, “Diffusion-convolutional neural networks,” in Proc. of NIPS, 2016, pp. 1993–2001.
[^17]: D.Bacciu, F.Errica, and A.Micheli, “Contextual graph markov model: A deep and generative approach to graph processing,” in Proc. of ICML, 2018.
[^18]: Y. Li, R. Yu, C. Shahabi, and Y. Liu, “Diffusion convolutional recurrent neural network: Data-driven traffic forecasting,” in Proc. of ICLR, 2018.
[^19]: D. V. Tran, A. Sperduti et al., “On filter size in graph convolutional networks,” in SSCI. IEEE, 2018, pp. 1534–1541.
[^20]: S. Yan, Y. Xiong, and D. Lin, “Spatial temporal graph convolutional networks for skeleton-based action recognition,” in Proc. of AAAI, 2018
[^21]: J. Gilmer, S. S. Schoenholz, P. F. Riley, O. Vinyals, and G. E. Dahl, “Neural message passing for quantum chemistry,” in Proc. of ICML, 2017, pp. 1263–1272.
[^22]: K. Xu, W. Hu, J. Leskovec, and S. Jegelka, “How powerful are graph neural networks,” in Proc. of ICLR, 2019.
[^23]: W. Hamilton, Z. Ying, and J. Leskovec, “Inductive representation learning on large graphs,” in Proc. of NIPS, 2017, pp. 1024–1034.
[^24]: P. Velickovic, G. Cucurull, A. Casanova, A. Romero, P. Lio, and Y. Bengio, “Graph attention networks,” in Proc. of ICLR, 2017.
[^25]: J. Zhang, X. Shi, J. Xie, H. Ma, I. King, and D. -Y. Yeung, “GaAN: Gated attention networks for learning on large and spatiotemporal graphs,” in Proc. of UAI, 2018. 
[^26]: Z.Liu, C.Chen, L.Li, J.Zhou, X.Li, and L.Song, “Genie path: Graph neural networks with adaptive receptive paths,” in Proc. of AAAI, 2019
[^27]: F. Monti, D. Boscaini, J. Masci, E. Rodola, J. Svoboda, and M. M. Bronstein, “Geometric deep learning on graphs and manifolds using mixture model cnns,” in Proc. of CVPR, 2017, pp. 5115–5124.
[^28]: M. Niepert, M. Ahmed, and K. Kutzkov, “Learning convolutional neural networks for graphs,” in Proc. of ICML, 2016, pp. 2014–2023.
[^29]: H.Gao, Z.Wang, and S.Ji, “Large-scale learnable graph convolutional networks,” in Proc. of KDD. ACM, 2018, pp. 1416–1424.
[^30]: S. Cao, W. Lu, and Q. Xu, “Deep neural networks for learning graph representations,” in Proc. of AAAI, 2016, pp. 1145–1152.
[^31]: D. Wang, P. Cui, and W. Zhu, “Structural deep network embedding,” in Proc. of KDD. ACM, 2016, pp. 1225–1234.
[^32]: T. N. Kipf and M. Welling, “Variational graph auto-encoders,” NIPS Workshop on Bayesian Deep Learning, 2016.
[^33]: S. Pan, R. Hu, G. Long, J. Jiang, L. Yao, and C. Zhang, “Adversarially regularized graph autoencoder for graph embedding.” in Proc. of IJCAI, 2018, pp. 2609–2615.
[^34]: K.Tu, P.Cui, X.Wang, P.S.Yu, and W.Zhu,“Deep recursive network embedding with regular equivalence,” in Proc. of KDD. ACM, 2018, pp. 2357–2366.
[^35]: W. Yu, C. Zheng, W. Cheng, C. C. Aggarwal, D. Song, B. Zong, H. Chen, and W. Wang, “Learning deep network representations with adversarially regularized autoencoders,”inProc.ofAAAI. ACM,2018, pp. 2663–2671.
[^36]: R. Go ́mez-Bombarelli, J. N. Wei, D. Duvenaud, J. M. Herna ́ndez- Lobato, B. Sa ́nchez-Lengeling, D. Sheberla, J. Aguilera-Iparraguirre, T. D. Hirzel, R. P. Adams, and A. Aspuru-Guzik, “Automatic chemical design using a data-driven continuous representation of molecules,” ACS central science, vol. 4, no. 2, pp. 268–276, 2018.
[^37]: M. J. Kusner, B. Paige, and J. M. Herna ́ndez-Lobato, “Grammar variational autoencoder,” in Proc. of ICML, 2017.
[^38]: H. Dai, Y. Tian, B. Dai, S. Skiena, and L. Song, “Syntax-directed variational autoencoder for molecule generation,” in Proc. of ICLR, 2018.
[^39]: Y. Li, O. Vinyals, C. Dyer, R. Pascanu, and P. Battaglia, “Learning deep generative models of graphs,” in Proc. of ICML, 2018.
[^40]: J. You, R. Ying, X. Ren, W. L. Hamilton, and J. Leskovec, “Graphrnn: A deep generative model for graphs,” Proc. of ICML, 2018.
[^41]: M. Simonovsky and N. Komodakis, “Graphvae: Towards generation of small graphs using variational autoencoders,” in ICANN. Springer, 2018, pp. 412–422.
[^42]: T. Ma, J. Chen, and C. Xiao, “Constrained generation of semantically valid graphs via regularizing variational autoencoders,” in Proc. of NeurIPS, 2018, pp. 7110–7121.
[^43]: N. De Cao and T. Kipf, “MolGAN: An implicit generative model for small molecular graphs,” ICML 2018 workshop on Theoretical Foundations and Applications of Deep Generative Models, 2018.
[^44]: A. Bojchevski, O. Shchur, D. Zu ̈gner, and S. Gu ̈nnemann, “Netgan: Generating graphs via random walks,” in Proc. of ICML, 2018.
[^45]: Y. Seo, M. Defferrard, P. Vandergheynst, and X. Bresson, “Structured sequence modeling with graph convolutional recurrent networks,” in International Conference on Neural Information Processing. Springer, 2018, pp. 362–373.
[^46]: Y. Li, R. Yu, C. Shahabi, and Y. Liu, “Diffusion convolutional recurrent neural network: Data-driven traffic forecasting,” in Proc. of ICLR, 2018.
[^47]: A. Jain, A. R. Zamir, S. Savarese, and A. Saxena, “Structural-rnn: Deep learning on spatio-temporal graphs,” in Proc. of CVPR, 2016, pp. 5308–5317.
[^48]: B. Yu, H. Yin, and Z. Zhu, “Spatio-temporal graph convolutional networks: A deep learning framework for traffic forecasting,” in Proc. of IJCAI, 2018, pp. 3634–3640.
[^49]: S. Yan, Y. Xiong, and D. Lin, “Spatial temporal graph convolutional networks for skeleton-based action recognition,” in Proc. of AAAI, 2018.
[^50]: Z. Wu, S. Pan, G. Long, J. Jiang, and C. Zhang, “Graph wavenet for deep spatial-temporal graph modeling,” in Proc. of IJCAI, 2019.
[^51]: S. Guo, Y. Lin, N. Feng, C. Song, and H. Wan, “Attention based spatial- temporal graph convolutional networks for traffic flow forecasting,” in Proc. of AAAI, 2019.
[^52]: Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. arXiv preprint arXiv:1706.03762.
[^53]: Liu, R., Lehman, J., Molino, P., Such, F. P., Frank, E., Sergeev, A., & Yosinski, J. (2018). An intriguing failing of convolutional neural networks and the coordconv solution. arXiv preprint arXiv:1807.03247.
[^54]: Weisfeiler, Boris and Lehman, AA. A reduction of a graph to a canonical form and an algebra arising during this reduction. Nauchno-Technicheskaya Informatsia, 2(9): 12–16, 1968.
[^55]: P. Veliˇckovi´c, W. Fedus, W. L. Hamilton, P. Li`o, Y. Bengio, and R. D. Hjelm, “Deep graph infomax,” in Proc. of ICLR, 2019.
[^56]: M. Schlichtkrull, T. N. Kipf, P. Bloem, R. van den Berg, I. Titov, and M. Welling, “Modeling relational data with graph convolutional networks,” in ESWC. Springer, 2018, pp. 593–607.
[^57]: I. Gulrajani, F. Ahmed, M. Arjovsky, V. Dumoulin, and A. C. Courville, “Improved training of wasserstein gans,” in Proc. of NIPS, 2017, pp. 5767–5777.
[^58]: M. Arjovsky, S. Chintala, and L. Bottou, “Wasserstein gan,” arXiv preprint arXiv:1701.07875, 2017.
[^59]: Zhou, J., Cui, G., Zhang, Z., Yang, C., Liu, Z., Wang, L., ... & Sun, M. (2018). Graph neural networks: A review of methods and applications. arXiv preprint arXiv:1812.08434.