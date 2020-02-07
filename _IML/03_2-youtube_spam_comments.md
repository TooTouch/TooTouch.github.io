---
title:  "3.2 Youtube Spam Comments (Text Classification)"
permalink: /IML/youtube_spam_comments/
---

텍스트 분류에 대한 예제들로 5개의 유튜브 동영상들에 있는 1,956개의 댓글을 사용합니다. 스팸 분류에 대해 다룬 논문에서 이 데이터를 사용한 저자들 덕분에 무료도 사용할 수 있습니다(Alberto, Lochter 그리고 Almeida (2015[^1])).

댓글들은 2015년도 1분기에 조회수 상위 10개 중 5개를 유튜브 API를 통해서 수집됐습니다. 5개 모두 음악영상입니다. 그들 중 하나는 한국 가수인 싸이의 "강남 스타일"이고 다른 가수들은 Kate Perry, LMFAO, Eminem 그리고 Shakira입니다.

몇몇 댓글들을 확인해봅시다. 이 댓글들은 다 손수 스팸인지 아닌지 분류해놨습니다. 스팸은 "1", 스팸이 아닌것은 "0"으로 분류되어 있습니다.

내용 | 클래스
---|---
Huh, anyway check out this you[tube] channel: kobyoshi02 | 1
Hey guys check out my new channel and our first vid THIS IS US THE MONKEYS!!! I’m the monkey in the white shirt,please leave a like comment and please subscribe!!!! |	1
just for test I have to say murdev.com| 1
me shaking my sexy ass on my channel enjoy ^_^| 1
watch?v=vtaRGgvGtWQ Check this out . | 1
Hey, check out my new website!! This site is about kids stuff. kidsmediausa . com	| 1
Subscribe to my channel | 1
i turned it on mute as soon is i came on i just wanted to check the views…| 	0
You should check my channel for Funny VIDEOS!!	| 1
and u should.d check my channel and tell me what I should do next!	| 1

유튜브에서도 이 댓글들을 확인할 수 있습니다. 단, 한번 들어갔다가 유튜브에 빠지지 않길 바랍니다. 결국 해변에서 원숭이가 관광객들의 칵테일을 훔쳐마시는 영상을 보고있게 될수도 있습니다. 구글 스팸 탐지기는 2015년 이후로 많이 변경됐을 것입니다.

[파격적인 조회수를 나타내는 "강남 스타일"을 여기서 확인해보실 수 있습니다. ](https://www.youtube.com/watch?v=9bZkp7q19f0&feature=player_embedded)

이 데이터를 통해서 무언가 해보고 싶으시다면 이 책의 Github repository에서 몇가지 간편한 함수들이 있는 [RData 파일](https://github.com/christophM/interpretable-ml-book/blob/master/R/get-cervical-cancer-dataset.R)과 [R-script](https://github.com/christophM/interpretable-ml-book/blob/master/data/cervical.RData)를 확인해보실 수 있습니다.

--- 

[^1]: Alberto, Túlio C, Johannes V Lochter, and Tiago A Almeida. “Tubespam: comment spam filtering on YouTube.” In Machine Learning and Applications (Icmla), Ieee 14th International Conference on, 138–43. IEEE. (2015).