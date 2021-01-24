---
title: "Spectral Clustering에서 왜 2nd eigenvector를 사용하는가?"
categories: 
    - Research
toc: true
--- 

Spectral clustering에서는 clustering을 하기위해 Laplacian Matrix ($$L$$)의 eigenvector 중 두 번째eigenvector 부터 사용한다. 이번 포스팅은 왜 첫 번째 eigenvalue에 해당하는 eigenvector는 사용을 하지 않는지에 대한 내용이다.

이번 포스팅에 대한 핵심 내용은 아래 두 가지 이다.

1. Graph에서 $$L$$은 positive semi-definite matrix이다. 즉, eigenvalue가 모두 0과 같거나 크다.
2. $$L$$의 모든 행의 합은 0이다. 따라서 모든 원소가 1인 eigenvector의 eigenvalue는 0이다. 

먼저 **positive semi-definite matrix** 가 무엇인지 알아야한다. positive semi-definite matrix 는 식 (1)을 만족하는 행렬이다. 식 (1)의 결과는 scalar 값이 나오는데 이 scalar 값이 항상 0보다 크거나 같다는 말이다. 만약 0을 포함하지 않는다면 그때 $$M$$은 positive definite matrix 라고 부른다.

$$\textbf{x}^TM\textbf{x} \geq 0 \tag{1}$$

Graph에서 $$L$$는  positive semi-definite matrix 다. 즉, 식 (1)을 만족하는 행렬이다. 이때 $$L$$에 대한 eigenvalue 를 구하면 식 (2)를 식 (3)과 같이 전개할 수 있다. 이때 $$L$$은 symmetric matrix 이기 때문에 순서를 바꾸어도 문제없다. $$L$$가 symmetric matrix 인 이유는 아래에서 설명한다. 결과적으로 식 (1)의 성질에 따라 식 (2)를 만족하는 $$\textbf{x}$$ (eigenvector)에 대해서 $$L$$의 eigenvalue 는 모두 0과 같거나 큰 값을 가진다.

$$L\textbf{x} = \lambda \textbf{x} \tag{2}$$

$$\textbf{x}^TL\textbf{x}=\textbf{x}^T\lambda\textbf{x} = \lambda\textbf{x}^T\textbf{x} = \lambda \tag{3}$$

$$L$$에 대한 eigenvalue는 graph signal 에서 spectrum 을 의미한다. 일반적으로 사용되는 PCA의 eigen decomposition 을 통해 사용하는 eigenvalue 는 내림차순으로 정렬하여 정보량이 많이 있는 eigenvector 를 사용하는 반면에 graph에서 $$L$$의 eigenvalue 이 나타내는 spectrum 은 노드 간의 similarity 를 말하기 때문에 차이가 적은(=eigenvalue 가 작은) 순서대로 오름차순 정렬하여 사용한다. 왜 $$L$$의 eigenvalue가 노드 간의 similarity를 나타내는지는 [여기](https://www.notion.so/Graph-Laplacian-Operator-df12eda3b6c34d33b5abe4b124b22134)에서 설명한다. $$L$$에 대한 공식은 식 (4)와 같다.  

$$L = D - A \tag{4}$$

여기서 $$A$$는 adjacency matrix 를 말하고 $$D$$는 degree matrix 를 말한다. $$A$$는 노드 간의 edge ($$E$$)가 있는 경우 1 아닌 경우는 0인 symmetric matrix 이고 $$D$$는 대각 원소가 각 node ($$N$$)에 연결되어 있는 $$E$$의 수를 나타내는 diagonal matrix 이다. 따라서 그 결과인 $$L$$ 또한 앞서 말한 것처럼 symmetric matrix 가 된다. 그림 1 과 같이 간단하게 예시를 들어 보일 수 있다. 

<p align='center'>
    <img src='https://user-images.githubusercontent.com/37654013/105630334-e4fa3500-5e8b-11eb-8495-9df3a8e17331.png'><br>그림 1. Laplacian matrix 계산 예시
</p>



이제 마지막으로 spectral clustering 할 때 **왜 첫 번째 eigenvalue 값에 대한 eigenvector 를 사용하지 않는지**를 설명할 차례이다. 그림 1과 같이 $$L$$의 **모든 행의 합은 0**이다. 즉, 모든 원소가 1인 eigenvector ($$e$$) $$= [1,\dotsb,1]^T$$ 에 대응하는 eigenvalue는 0이다. 식 (1)에 따라 $$L$$의 eigenvalue 는 0과 같거나 0보다 큰 값을 갖는다. 가장 작은 값을 기준으로 정렬하기 때문에 자연스럽게 eigenvalue 0이 첫 번째로 오고 그에 대한 모든 원소가 1(또는 일정한 상수)인 eigenvector가 온다. 

Spectral clustering은 그림 2와 같이 $$L$$의 eigenvector를 기준으로 clustering 하기 때문에 모든 값이 상수인 첫 번째 eigenvalue의 eigenvector가 아닌, 결론적으로 **두 번째 eigenvalue의 eigenvector(이를 Fiedler vector라 부른다)부터 사용한다.** 

<p align='center'>
    <img src='https://user-images.githubusercontent.com/37654013/105630336-e62b6200-5e8b-11eb-99db-b2b124eaf171.png'><br>그림 2. spectral clustering 예시<br>출처 : cs224w 2019 Fall
</p>


