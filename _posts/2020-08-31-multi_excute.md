---
title: "Python 실행시간 2배 이상 줄이는 방법"
categories: 
    - Setting
toc: true
---

신도림 어느 한 카페. 그 곳은 바로 [Gecko](https://map.naver.com/v5/entry/place/1483163346?c=14125408.655867513,4509512.727182969,13,0,0,0,dh&placePath=%2Fhome%3Fentry%3Dplt)라는 곳이다. 갑자기? 라고 생각할 수 있지만 갑자기가 맞다. 

카페에 앉아 영어 공부를 하느냐 마느냐 내 머리 속에서 씨름을 벌이던 어느 저녁, 교수님께서 연락이 오셨다. 

<p align='center'>
    <img width='300' src='https://user-images.githubusercontent.com/37654013/91682584-febd7780-eb8c-11ea-88f3-cda88505453b.jpeg'>
</p>

음..  처음 본 순간 들었던 생각은 **'동시'**실행이 무엇일까.. 그냥 한 줄로 실행하는 것일까 생각해서 간단하게 코드를 짜보고 보여드렸다.

<p align='center'>
    <img width='300' src='https://user-images.githubusercontent.com/37654013/91682741-7db2b000-eb8d-11ea-9e54-be219f7c8a0d.jpeg'>
</p>

마치 "이 도끼가 너의 도끼냐", "아니오.."와 같은 금도끼 은도끼를 시전하며 티키타카를 주고받은 후 이제서야 어떤 의미인지 알게되는데,,

<p align='center'>
    <img width='300' src='https://user-images.githubusercontent.com/37654013/91682574-f6653c80-eb8c-11ea-98a3-f6117c4ed759.jpeg'>
</p>

예전에 잠시 전처리가 너무 오래걸려서 python의 multiprocessing으로 해결한적이 있어서 비슷한 키워드로 검색했더니 역시 갓택오버플로(라 읽고 stackoverflow라 쓴다)에는 없는게 없었다.

<p align='center'>
    <img width='300' src='https://user-images.githubusercontent.com/37654013/91682502-c0c05380-eb8c-11ea-8eae-f8b220ed8958.jpeg'>
</p>

그리하여 만들게된 예시..!

검색해보니 이전에 내가 써봤던건 multiprocessing이고 thread라는 모듈이 또 있었다.

파이썬은 기본적으로 연산을 하게될때 하나의 쓰레드(thread)만 잡고 사용하게된다. 나는 입이 12개인데 1개만 써서 밥을 먹고 있다는 것이다. 입이 12개라면 36분 먹을 음식을 3분만에 먹을 수 있는거 아닌가? 매우 시간을 효율적으로 사용할 수 있는 방법이다. 그러나 어느누가 맛있는 음식을 빨리 먹고싶겠는가. 내 눈 앞에 스테이크가 있다면 한 입 한 입 소중히 먹을거다. 아마 파이썬도 같은 심정이었지 않을까싶다.

# 실험

각설하고 다시 실험해보았던 예시에 대해 얘기해보자. 우선 예제 코드를 만들기전 목적을 분명히 하기로했다. 나의 목적은 **'그래서 시간이 줄어 안줄어?'**이기 때문에 실험 방법은 아래와 같이 세 가지로 진행했다.

1. 각 함수를 실행
2. 멀티 쓰레드로 동시 실행
3. 멀티 프로세스로 동시 실행

파이썬에는 **threading**과 **multiprocessing**이라는 모듈이 있다. 설치는 기본적으로 되어있을거라 생각된다. 아래는 간단하게 사용해본 예시 코드이다. 

실험 환경에서 사용된 CPU나 RAM은 따로 언급하지 않아도 될까싶다. 왜냐하면 중요한건 '내 CPU는 좋아서 이것밖에 시간이 안들지롱'이 아니라 '시간이 그래서 줄었니'이기 때문이다.

## 사용 모듈

우선 실험에 사용한 모듈들. 간단한 예제 돌려보는데 뭐가 이렇게 많이 필요하나 싶지만 다 계획이 있다. 대충 예상이 간다면 코드로 고고!

```python
from threading import Thread 
from multiprocessing import Process

from rich.progress import track

import pandas as pd
import argparse
```

우선 파이썬 코드를 병렬처리하기 위한 두 가지 모듈인 threading과 multiprocessing이다. 사용할 클래스는 각각 Thread와 Process이지만 사용 방법은 동일하다. 이후 예시에서 확인하길

```python
from threading import Thread 
from multiprocessing import Process
```

다음은 다른 부차적인 모듈이다. 각각 주로 실험할 때 주로 사용하는 모듈이다. 하나씩 설명하자면, 

우선 rich! rich는 비교적 최근에 공개된 UI 모듈(?)이다. 기존에 칙칙했던 터미널 화면에서 벗어나게 해줄 구세군같은 존재이다. 사용 방법도 굉장히 간단하므로 [rich github](https://github.com/willmcgugan/rich)에서 확인하면 된다. 

다음은 pandas이다. 아마 python을 사용하면서 pandas를 모르는 사람은 없을거라 생각된다. 여기서 pandas를 사용한 이유는 I/O 차이를 실험해 보기 위함이었다. 이후 예시에서 얘기해보기로 한다.

그 외 argparse는 실행명령어와 함께 인자(argument)를 넘기기위해 사용했고 여기서 인자는 for문을 얼마나 돌릴건지와 어떤 방법으로 실험할 것인지이다. 


```python
from rich.progress import track

import pandas as pd
import argparse
```

## 코드

우선 실험에 사용한 함수를 동일한 내용으로 두 개의 함수를 만들었다. 간단하게 두 개의 리스트의 연산 결과를 받아서 리스트에 추가하는 내용이다. 그리고 연산이 끝나면 결과 리스트를 데이터프레임으로 변환하여 저장하도록 했다.

```python
def func1(n):
    x = range(n)
    y = range(n)
    result = []
    for i in track(range(n), description='func1'):
        result.append(x[i] + y[i])
    df = pd.DataFrame(result)
    df.to_csv('result1.csv')
    print('Save result1.csv')

def func2(n):
    x = range(n)
    y = range(n)
    result = []
    for i in track(range(n),description='func2'):
        result.append(x[i] - y[i])
    df = pd.DataFrame(result)
    df.to_csv('result2.csv')
    print('Save result2.csv')
```

실험 실행 코드는 아래와 같다. 간단하게 인자를 받아줄 argparse와 그 아래는 각각 세 가지 방법별로 조건문에 따라 나눠주었다. 아마 코드 실행 시간에 대한 불만까지 왔다면 어느정도 코딩을 할줄 아는 사람이라는 생각이 들기에 자세한 설명은 생략한다..

```python
if __name__=='__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--iter',type=int,default=100,help='The number of iterations')
    parse.add_argument('--thread',action='store_true')
    parse.add_argument('--process',action='store_true')
    args = parse.parse_args()

    funcs = [func1, func2]
    results_list = [[],[]]

    if args.thread:
        print('Multi-threading')
        threads = [Thread(target=funcs[i], args=(args.iter,)) for i in range(len(funcs))]
        for thread in threads:
            thread.start()
    elif args.process:
        print('Multi-processing')
        processes = [Process(target=funcs[i], args=(args.iter,)) for i in range(len(funcs))]
        for process in processes:
            process.start()
    else:
        print('None')
        for i in range(len(funcs)):
            funcs[i](args.iter)
```

아래 코드를 보면 알겠지만 위에서 언급했다싶이 Thread와 Process의 사용 방법은 이름만 다르지 동일하다는 것을 알 수 있다. 그래서 상황에 맞게 바꿔쓰기가 아주 편하다는 점! 

```python
threads = [Thread(target=funcs[i], args=(args.iter,)) for i in range(len(funcs))]
processes = [Process(target=funcs[i], args=(args.iter,)) for i in range(len(funcs))]
```

위 코드를 실행하기 위해 간단한 쉘(shell) 스크립트를 만들었다. 왜? 라고 물어본다면 대답해주는게 인지상정. 코드 실행 시간을 확인하기 위함이다. 파이썬 스크립트에서도 충분히 할 수 있지 않을까 싶지만 이렇게 하는것에는 다 이유가 있었다.. 랄까?

멀티 쓰레드를 사용하는 경우 이상하게도 코드가 순서대로 실행되긴하지만 다른 쓰레드도 같이 활용해서 여러 함수를 실행하기 때문인지 시간을 확인하기 위한 코드가 먼저 실행되어버려서 정확히 함수 실행이 끝난 시간을 알기 어려웠다. 때문에 쉘 스크립트를 통해 파이썬 명령문의 시작과 끝난 시간을 계산해서 실행 시간을 확인했다.

```bash
STARTTIME=$(date +%s)
python test.py --iter=$1 $2 
ENDTIME=$(date +%s)

echo "It takes $(($ENDTIME - $STARTTIME)) seconds to complete this task..."
```


## 결과

실험 결과를 얘기하기전 자꾸 얘기가 다른 곳으로 세나 싶지만 이 글을 읽는 사람들에게 도움이 될까 싶어 던져본다.

이번에 터미널의 결과 화면을 녹화하고 싶은데 어떻게 할까 고민하던 찰나에 알게된 [asciinema](https://asciinema.org/) ! 한국어로 잘 설명해준 포스팅은 여기를 [참고](https://rsec.kr/?p=721)하면 된다.

**결과 확인 방법**

다시 본론으로 돌아와서 실행 결과를 비교해보자. 결과 비교를 위해서 전체 코드가 **실행된 시간**과 그리고 **cpu의 thread가 병렬적으로 잘 실행되고 있는지** 비교하기 위해 [htop](https://ko.wikipedia.org/wiki/Htop)을 통해 확인했다. 

**htop**란? 아래와 같이 터미널 환경에서 cpu가 일을 잘~하고 있는지 확인할 수 있는 명령어이다. 없다면 설치는 간단하니 설치하자. 

해당 실험에서 반복할 iteration 수는 10,000,000회로 하였다.

```bash
sudo apt install htop
```

<p align='center'>
    <img width='400' src='https://upload.wikimedia.org/wikipedia/commons/thumb/b/b1/Htop.png/600px-Htop.png'><br>출처: 위키백과
</p>


**1. 각 함수를 따로 실행**

우선 각 함수를 따로 실행했을 때 소요된 시간이다. 전체 실행 시간은 **36초**가 소요되었다. 실행 당시 CPU 상황을 보면 하나의 쓰레드만 신나게 사용되다가 함수 하나 계산이 끝나면 이번엔 나~ 하면서 다른 CPU가 신나게 일하는 것을 확인할 수 있다.

<p align='center'>
    <img src='https://user-images.githubusercontent.com/37654013/91686024-24e81500-eb97-11ea-9d7d-2074424ee15f.png'>
</p>

동영상은 모바일에서 보니 옆에가 짤려서 나온다.. 가로 비율이 높은 화면에서 보면 전체화면이 나오니 궁금하신분은 데스크톱에서 보시길..

<p align='center'>
    <script id="asciicast-32izQtxa6jceRe5V4A8WFgxC5" src="https://asciinema.org/a/32izQtxa6jceRe5V4A8WFgxC5.js" async></script><br>코드 실행 화면
</p>

<p align='center'>
    <script id="asciicast-cybN7rzW1EOc0KvNuJ7pDkxnx" src="https://asciinema.org/a/cybN7rzW1EOc0KvNuJ7pDkxnx.js" async></script><br>CPU 화면
</p>

**2. 멀티 쓰레드로 동시 실행**

다음은 멀티 쓰레드를 사용해서 동시 실행하는 화면이다. 동시 실행이라 progress bar가 동시에 생겨나는 것을 볼 수 있다. CPU도 위의 실험과는 다르게 여러 쓰레드가 동시 다발적으로 열심히 일하는 모습을 볼 수 있다. 실행 결과는 **35초**이다. 응? 응 35초이다. 왜 35초일까? 

바로 멀티 쓰레드는 여러 함수를 여러개의 쓰레드를 활용해서 계산하는 것이지 병렬적으로 연산을 처리하는 것은 아니다. 이게 무슨 말일까 싶지만 하나의 쓰레드와 멀티 쓰레드의 가장 큰 차이라고 한다면 I/O 시간을 줄일 수 있다는 것이다. 하나의 쓰레드만 사용하게 되는경우 I/O까지 끝난 후 다름 코드를 실행하게 된다. 그러나 멀티 쓰레드를 사용하게되면 I/O를 기다리지 않고 다른 쓰레드에서 코드를 실행할 수 있다. 라고 한다.

조금 더 뚜렷한 차이를 보려면 더 큰 연산을 실험해볼 필요가 있지만,, 솔직히 귀찮다. 

<p align='center'>
    <img src='https://user-images.githubusercontent.com/37654013/91686139-755f7280-eb97-11ea-838a-a9d2e721987e.png'>
</p>

<p align='center'>
    <script id="asciicast-934qsobznLyHFVnulLfMDCmvo" src="https://asciinema.org/a/934qsobznLyHFVnulLfMDCmvo.js" async></script><br>코드 실행 화면2
</p>

<p align='center'>
    <script id="asciicast-JXvL7R6E3WByYM1dG5hSdIMJ0" src="https://asciinema.org/a/JXvL7R6E3WByYM1dG5hSdIMJ0.js" async></script><br>CPU 화면2
</p>


**3. 멀티 프로세스로 동시 실행**

마지막으로 멀티 프로세스를 사용해서 동시 실행하였다. 결과는 당연히 위 두 방법보다 연산 시간이 줄어야한다. 실행 화면을 보면 func1과 func2가 거꾸로 실행된 것처럼 보이지만 동시에 실행됐지만 func2가 먼저 연산이 끝나서 순서가 뒤집어 졌다. 결과는 **20초**이다. 위 두 방법에 비해 거의 2배 정도 연산 시간이 줄었다. 

<p align='center'>
    <img src='https://user-images.githubusercontent.com/37654013/91686666-d20f5d00-eb98-11ea-855e-140eff83c76e.png'>
</p>

<p align='center'>
    <script id="asciicast-SyusuNzqOoN1371JKbDqI8yev" src="https://asciinema.org/a/SyusuNzqOoN1371JKbDqI8yev.js" async></script><br>코드 실행 화면3
</p>

<p align='center'>
    <script id="asciicast-oYuQYcKHWu7RfEX9tOOJKE3TL" src="https://asciinema.org/a/oYuQYcKHWu7RfEX9tOOJKE3TL.js" async></script><br>CPU 화면3
</p>

# 결론

간단한 코드말고 실제 전처리에서 어느정도 효과적으로 시간을 줄일 수 있을지도 비교해보려 했지만 그렇게 하게되면 코드에 대한 설명만 부차적으로 늘어나게 될거같아서 누구나 보고 이해하기 쉽도록 간단한 예제를 동반해서 설명했다. 

그 동안 한참을 전처리 시간을 기다리며 지루했던 당신! 이 방법을 사용해서 효율적으로 시간을 줄여보자. 그러나 나는 사용하지 않을 수도 있다. 코드를 돌려놓고 산책을 다녀오는 그 시간만큼은 양보할 수 없기 때문이다. 
