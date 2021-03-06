#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import re

# 韓國瑜受邀訪美為哪樁？
new1_content = '高雄市長韓國瑜受哈佛大學費正清中國研究中心邀請，即將前往美國訪問，這份來自美國的演講邀請，在台灣引起媒體和政界人士議論，因為大家猜想美國政府和學界已經預期，韓國瑜將爭奪總統大位，所以要先做準備，可是，美國真的有那麼了解台灣政局發展嗎？與其說邀請韓國瑜訪美，是華府和美國學界認為韓國瑜將會參選總統，所以想對未來台灣可能的領導人進行面試，不如說美方驚覺他們對台灣民意認識不足，因此採取亡羊補牢的措施，想重新加強對台灣政情的掌握。事實上在兩零一八年台灣九合一選舉前夕，美國學界和華府都還一面倒的認爲，就算民進黨不會選得很好，但國民黨也不可能有翻身的機會，而這也是為什麼主流媒體如紐約時報在選後，會用「驚人」一詞來形容台灣選舉的結果。從台灣看世界，媒體和民眾總認為全世界都覺得台灣在世界上有關鍵的角色，不過，真實的情況是，除了台灣自己，恐怕不會有哪一國真的把台灣看得很重要，而美國政府和學界對台灣的認識自然也相當有限。以華府的政治人物們來說，他們對台灣的認識，除了台灣駐外代表處的官方正式管道之外，許多訊息都來自在美國長期支持台灣民主運動，爭取台灣獨立的海外台灣社團。這些旅美台灣社團透過選舉募款和生意往來，與眾多美國國會議員建立良好的關係，也因此在川普上台決定要對中國採取強硬手段，使得美中關係日益緊張之際，成功遊說美國國會通過幾項對台灣有利的法案。然而，透過這些海外台灣同鄉組織獲得台灣相關資訊，美國政治人物和學界對台灣的認識，究竟真的是台灣的主流民意，還是只獲得了片面的消息？從美國媒體和學者專家都認為台灣主流民意是要跟中國切割，認定兩岸之間的九二共識，已經不再被大多數台灣人所接受來看，他們對台灣的認識恐怕真的不算全面。話說回來，相較於為了捍衛台灣主權而向美國靠攏的主張，在當前一心想要制衡崛起中國的美國眼中，想與中國保持經貿關係穩定的想法，反而成了阻礙美國以台灣箝制中國的阻力。加上泛藍陣營在美國沒有花心思建立交流和溝通的管道，導致美方認為在台灣支持擴大與中國交流的民意只是少數，並且對兩岸和平的主張越來越摸不著頭緒。在世界兩強之間求生存，要不要選邊站值得台灣仔細思考。理論上來說，因為對民主價值的認同，選擇跟美國親近本是再自然不過的事，但是轉過頭面對現實，中國的市場商機和就業機會，也的確可以滿足當前台灣許多人民的需求。不管是政治人物，學者專家，還是商界人士，台灣的社會菁英同溫層裡，不論主張的是親美還是和中，其結果都對身為菁英的同溫層沒有太大影響。可是，不同的選擇卻會對絕大多數台灣人民的生活帶來不同的衝擊。簡單來說，台灣人民不是不懂民主的價值，而是多數人口袋裡根本沒有本錢，可以跟同溫層裡的菁英站在一起。邀請韓國瑜訪美，不是美國人熟悉台灣，而是想要知道為什麼，台灣政府和旅美菁英們所提供的資訊，會跟選舉結果有這麼大的差異。說穿了，韓國瑜訪美行，根本不是去接受美國面試，而是要去給美國人上一堂台灣民主課。'

new1_review = '原來冥進洞豬霸爸被綠豬兒給騙了!這是衝3小?，2018年台灣九合一選舉→大多數人要顧八豆，還有少部份人也餓醒了。想要知道為什麼，台灣政府和旅美菁英們所提供的資訊，會跟選舉結果有這麼大的差異?，因為台灣政府和所謂綠營旅美菁英們給的都不是真實的,韓國瑜應告訴美國,台灣政府和旅美菁英們是多數人民認證的詐騙執政黨,美國真的被騙而不知。92共識或許不是全台灣人民都有共識，但想跟對岸有經貿往來的，就一定要認同92共識，所以有問題的是那些在台灣說沒92共識卻去對岸撈錢的鬼島人!!。所以選前綠蛆連美國爸爸都敢騙?。九二共識 一中各表,是國民黨一向的主張,從2008年馬總統上台就任總統開始, 便是在此一基礎上展開海峽兩岸交流,進一步擴大兩岸 通商 文化 教育 觀光等 各方面,除了開放大三通外，更在2011年開放大陸學位生來台就讀,衛福部更以觀察員出席世界衛生大會,在馬總統任職之八年當中除非洲甘比亞與我斷交 而欲與老共建交部程之外,未再減少邦交國,兩岸關係穩定,期間台灣雖逢世界經濟海嘯,但仍能度過難關,迄2016年第二次政黨輪替,臺灣在中華民國的保護下有少哪一塊土地有少哪一分權益,反觀小英執政兩年餘兩岸外交內政各方面慘不忍睹。2018年九合一大選民進黨大敗,在韓流引領下,討厭民進黨成為台灣最大黨,因此老美政府學界及專家覺得應近身直接從韓國瑜身上做第一手瞭解臺灣得主流民意為何,作為爾後對台政策知正確參考不再是綠媒提供知片面偏頗之訊息。親美仇中與親美和中 那一種對台灣有利 這是膝蓋動一動的問題。2018 韓國瑜熱愛中華民國的九二共識 ! ，旗捲翻轉全台灣 ! ，親中保護中華民國 ! ，親中和平共存保台灣 ! ，親中保護台灣自由民主 ! ，這些事美國鬼迷心竅反中思想被屏蔽? 2018台灣人民已徹底全面覺醒 ! 唾棄這些操弄反中的政黨 !。哇這篇文章真的一針見血，那些名嘴都是米國人有米國中央情報局饌養，不愁吃穿 一般升斗小民卻要為張羅3餐，沒錢可賺煩惱，根本不接地氣，現代版晉惠帝何不食糜肉。美國人不知道豬鼻蔡是扶不起的阿斗?。韓國瑜去給美國佬上一堂「美國人不是台灣人的爸爸」的民主課也好……。其實AI早就預測正確了，只是美國高官不相信罷了。韓國瑜參選我才會回台投票。說白了問題出在美國,保不住台灣加入國際組織例如WHA,不跟台灣簽FTA,強賣農產品賣高價次級軍購,沒承諾要護台。支持92共識一中各表。韓國瑜穩贏!。CIA為了摸底佈局,如果交出自已黑資料把柄,美國人才會放心,將來選舉時才不會整他。很簡單，就是人民受夠雙重標準的詐騙集團統治，所以一舉推翻傲慢的東廠。讓老美了解台灣真正民意。為什麼一個所謂主權獨立國家政客,要接受美國人摸底?'
# 垃圾」風波市議會要求道歉 柯文哲：社會自有公評
new2_content = '台北市長柯文哲23日展開4天訪日行程，上午出發前被媒體問及對於20日在台北市議會小聲地罵了民進黨市議員王世堅「垃圾」，市議會要求柯文哲道歉，否則27日就拒絕柯到議會專案報告。對此，柯文哲表示，「我們有時候話講了就講了，但相信社會上自有公評。」柯文哲20日赴台北市議會專案報告，他在王世堅質詢結束後，小聲地罵了「垃圾」，對此，藍綠議員22日在議會大會砲轟柯不尊重議會監督職權，要求柯文哲必須道歉，否則27日就不讓柯到議會專案報告。將展開4天訪日行程的柯文哲，23日上午在松山機場被媒體問及此事，他表示，「求取公平就好，議員每天謾罵官員都沒事，官員講一句就要道歉，議員每天謾罵官員要不要道歉，所以在社會上求取公平。」媒體追問27日是否會道歉？柯文哲先說「再講啦」，接著又說，「我們有時候講話講了就講了，但相信社會上自有公評，我們還是求取一個公平正義的社會就好。」似乎不願對這件風波道歉。被問到此次的訪日目的，柯文哲表示，日本對台灣來講，不管在文化、地理、社會都滿接近，所以很多東西都可以跟日本學習。他說，第一點，台日在產業上來往密切，這次主要是台日觀光論壇，日人來台觀光人數190萬人次，台人到日460萬人次，日本人口是台灣人口5倍，明顯台人去日本的多很多，「所以要研究如何來鼓勵日人來台觀光，這也是謝長廷大使特別拜託我去幫他支援，促進台日觀光交流。」第二點，柯文哲指出，台北市長本來就很多城市外交，其實過去只要是日本訪問團來台灣，他幾乎都有會見，這次去東京都知事、首相弟弟岸信夫或者是自民黨幹事長、日本美國特使河井克行，過去也常常有見面，都見面好幾次，順便到日本去一次把城市外交該處理的處理完。而被問及此行有許多不公開行程，柯文哲回應，外交很多事不需要拿到檯面上敲鑼打鼓；媒體追問是否日方考量柯可能參選總統必須要謹慎、小心？柯則說，日本還是大國，處理外交關係上有一定的SOP。(法駐聯合國代表團祝台同性伴侶合法結婚)'

new2_review = '市長受議員監督，市長受議員監督，市長受議員監督，很難理解?。看看那嘴臉,噁心的投機份子, 這就是中華民國首都的市長?，天佑台灣。垃圾市長...誰投出來的...。挺柯P等於挺貪扁。社會自有公評，議會是讓議員質詢市府作為的，所以議員有言論免責權，不是拿來給市長耍脾氣發洩情緒的保貪就醫、指定分包、剪身分證、拔韓插吳、放任違建火災、搞爛蛋，這不是垃圾應該連核廢料都不如！。噁心的小丑市長。現在就如此傲慢，如果選上總統，豈不直接將罵他的人關押。一張爛嘴走天下！？，柯府教養真是好！？。對你很失望。我就是看柯文哲是政治小丑。那我評你是拉基。沒錯!問政歸問政，憑什麼議員可以辱罵他人，很扯。柯只是突然想到家裡垃圾沒倒，恰吉自己就對號入座了。世間情世堅人，質詢垃圾市長變垃圾議員，情何以堪？。韓國瑜面對黑韓也一定心裏不舒服，但也不口出惡言，和柯P比起來，高貴了多，這就心態問題！，公僕任勞任怨，但不是生來就是要給人超，但絕對還會保持一定的風範，柯P真性情，很好！不過那是你不是公僕的身份的時候，現在有公職，你算老幾 ? 。雖然一個不小心洩漏的內心，但人家要你道歉，怎麼還敢大辣辣say no ,很顯然票多托大！。'
# 台灣同婚合法化寫下亞洲第一 法國駐聯合國副代表公開讚揚
new3_content = '台灣總統蔡英文22日簽署立法院日前三讀通過的《司法院釋字第748號解釋施行法》，而在17日立院通過該法的當天，除了英國外交大臣杭特、加拿大外交部均推文祝賀台灣人民，法國常駐聯合國副代表古根也在聯合國國際反恐同日慶祝活動中，恭賀台灣LGBTI伴侶終於有權依法結婚。此外，法國人權大使克羅凱特也說，台灣同婚合法化是國際反恐同日最美麗的象徵。每年5月17日是國際反恐同日，因為世界衛生組織（WHO）在1990年的這天，正式把同性戀傾向從疾病分類表中除名，自此同性戀不再是心理或精神疾病。聯合國也在這天舉行慶祝活動「正義與保障人人平等」，法國駐聯合國副代表古根公開讚揚台灣同婚合法化。「每個往正確方向踏出一步的時刻，我們都應同感歡欣鼓舞」古根表示，「從這個角度來看，我要祝賀所有台灣LGBTI伴侶，從今（5月17日）開始他們有權依法結婚了」。此外，2019年2月曾訪台的克羅凱特（François Croquette）也在17日貼文祝賀：「今天，台灣向前邁進一大步，賦予所有人結婚權利，這是國際反恐同日最美麗的象徵。」台灣成為亞洲首個同婚合法化的國家引起國際關注，英國外相杭特（Jeremy Hunt）推文稱：「恭喜台灣人民通過同婚！這是台灣LGBT社群的大好消息，也是亞洲LGBT平權向前邁進的一大步。」加拿大外交部也推文寫道：「祝福台灣人民，亞洲同婚合法化首例。」奧地利台北辦事處處長陸德飛（Roland Rudorfer）也說，此舉確立台灣在促進與保護人權的領先角色。'
new3_review = '謝謝民進黨 全世界肛台灣。好噁心，資紀黨還說愛無差別，對資進黨來說，走前門走後門沒差別，好噁心好難看。表揚亂倫，不如表揚蔡母。'

# 友邦超給力！WHA全數再為台灣發聲
new4_content = '台灣連續三年，遭中國杯葛打壓缺席世界衛生大會，外交部表示22日共9國友邦接力位台灣發聲，尼加拉瓜副總統穆麗優更透過其國內電視台發表廣播聲明力挺台灣。外交部表示「21日巴拉圭、瓜地馬拉、宏都拉斯及帛琉等4友邦於「世界衛生大會」(WHA)全會上為我未獲邀出席事仗義執言後， 5月22日續有吉里巴斯、史瓦帝尼、聖克里斯多福、聖露西亞、聖文森、諾魯、貝里斯、海地及吐瓦魯等9國友邦代表在全會中為我執言；索羅門群島及馬紹爾群島等2友邦則在A委員會(Committee A)為我強力發聲」。尼加拉瓜副總統穆麗優（Rosario MURILLO）則於同日在尼國國家電視台發表廣播聲明，大力聲援我國。另繼美、英、法、德、日、加、澳等7國於全會中為我聲援後，紐西蘭亦於5月22日的全會中發表友我言論。外交部指出，衛福部陳部長此行在日內瓦迄（22）日為止，計已舉行71場雙邊會談，超過去年之60場。截至23日為止，本屆WHA共有16個友邦為我仗義執言，而梵諦岡教廷因為非世俗國家，向例不就政治性議題表達立場；另有8個理念相近國家發表友我言論。'

new4_review = '尼國：錢已收到，明年繼續！。友邦發聲不用給錢有人相信嗎?，發聲給錢還是參與不了這錢跟扔到海裡有什麼差別'
# 美國將被取代？川普：我任內不可能
new5_content = '美國總統川普祭出禁令，讓全球企業爭相取消與華為的合作，日前他接受專訪，被問到貿易戰的相關話題時，親口坦承中國的確想取代美國，成為世界強權領導者，不過川普也霸氣的說「在我任內絕不可能」！川普日前接受他最喜愛的媒體《福斯新聞》專訪，內容包含施政、外交、經濟等議題，期間主播希爾頓（Steve Hilton）問及貿易戰的問題時，川普自信的說「我們額外進帳了數十億的關稅，相反的中國的狀況就不太好」，並自稱從他上任以來，美國已經賺進超過10兆美元（約新台幣315兆元）。除此之外，川普親口坦承中國一直虎視眈眈，妄想擊敗美國成為第一強權，認為中國確實非常具有野心，且中國有許多聰明的人、古老的文化傳承，不過川普話鋒一轉，也說「但是在我任內是不可能發生的事情！」'

new5_review = '加速到達的首席戰犯!。搖頭美國民主可憐耍流氓。美國不會被取代但是會成為歷史名詞...。自大自私的美國人。換個角度想 如果美國被川普玩掛，世界重新洗牌也不錯，是丕是就可以不買死老美的過期武器了。'

# 賴清德糗了！這區里長竟全數挺蔡英文
new6_content = '前行政院長賴清德角逐2020民進黨總統提名，多次宣稱年初為立委郭國文輔選時，多次聽聞地方不滿總統蔡英文表現，因而毅然決定參選，不過位於郭國文立委選區的台南麻豆，共20位里長23日連署表態，力挺蔡英文連任。根據自由時報報導，台南市麻豆區共20位里長，23日發表連署全力支持蔡英文競選總統連任，其中身兼當地重要信仰宮廟良皇宮主委的中興里長吳振和表示，2年多前蔡英文曾贈匾給良皇宮，將力邀總統前來揭匾。吳振和表示，麻豆以台灣最堅強、最團結的民主聖地為榮，希望固守台灣價值，發揮中流砥柱的力量，力挺蔡英文總統競選連任，不過吳振和也提到，未來無論誰在民調中勝出，都會力挺出線者角逐2020總統大位。'

new6_review = '這些民進黨的里長為了自己的(錢))途做出明智的決定嗎?賴清德應該感嘆自己的一念知仁，讓蔡英文充容佈署回擊，歷史上這類的事重複上演，現在賴清德最好的一步，不是去當蔡英文副手---而是蓄積實力躲著不出來既不輔選也不爭鋒。睞埤枸千萬要挺住， 那操支襬已經奧步盡出，本村全力支持你。果然有夠綠，西瓜偎大邊。縱使天下人不支持我......我連表達的權利都沒有嗎????低級的冥禁洞。郭國文抖了一下，還好現在不是立委補選，不然就要輸給龍介仙了。'

# 觀察站／賴清德險招 盼逼出延長賽終點
new7_content = '民進黨中執會昨天討論總統初選時程，蔡賴陣營僵持不下，最後民進黨主席卓榮泰宣布散會，身為參選人的行政院前院長賴清德突然現身黨中央並提出新方案，大動作讓黨內驚訝。從賴清德最近突然展現強勢風格，再到昨天主動出招來看，策略其實都是期盼逼出這場延長賽的終點，讓自己有公平參加初選的機會。賴清德昨天提出的新方案，可能發生四種狀況，第一種狀況是蔡贏韓國瑜，賴無條件退出，也就是即便賴清德也贏韓國瑜，仍支持蔡英文連任。第二種狀況是蔡賴都輸韓國瑜，即便賴輸比較少，賴也退出。第三種狀況是只有蔡贏韓國瑜，但賴輸韓，賴退出。唯有賴贏韓國瑜，但蔡輸韓，民進黨則由賴清德代表出馬競選總統。賴清德提出的方案明顯是「險招」，因為四種狀況中，只有一種狀況可讓自己出線，且依據聯合報本周一公布的民調來看，無論賴蔡都無法贏過韓國瑜，因此若依賴新方案，賴必須退出的可能性並不小。但若以「逼出終點」的策略來看，賴清德出這一招仍有展現決心的意義。不但把球再丟給蔡英文陣營，也向外界證明自己無論處於多艱困的環境都「只求一戰」。因此蔡英文若不肯接招，繼續讓初選延長下去，如此一來蔡英文將面對更多基層壓力。但蔡英文接受了，等於接受「有條件被禮讓」，無論結果如何，免不了「正當性」遭質疑。民進黨總統初選時程從四月結束延到現在，支持者早就失去耐心。賴陣營近日對手機民調比例及對比式對象不願退讓，最大原因是與對手早已沒有互信，不相信對方口中的「公平」。相對而言，蔡陣營也不願意接受賴所提的新方案，雙方都不願被對方主導，可以預見，這場初選仍很難有任何共識，只能競爭決勝負，只是誰也不知要如何收拾殘局，讓對決上演。'

new7_review = '這正是以DDP贏得總統選戰為目標的初衷!初選賴若輸韓出來沒意義，兩個都贏那麼賴也不必出來! 所以只有賴贏蔡輸情況下，由賴出線打選戰才有機會保住繼續執政!也才合理，高招!我看中執會，會使出霸王條款，直接沒收初選。初選何時開始， 何時結束， 還沒定論， 歹戲繼續拖棚吧。空心菜卡八田癩德，大家有目共睹！無論是菜還是賴會對自己有利的機制才等於公平。耍賴而已，人家都出陰招，你出險招沒用的。實實在在有夠爛的政治人物,政黨,竟然名調還會上升??????????????????。瘟神戰蔡媽祖。反正誰出來都是敗選，爭甚麼呢？除非是想爭2020敗選後的黨主席？民進黨這齣戲就跟古早的黑道電影劇情一模一樣。果然是個幫派啊。民進黨被你亡完了，國民黨應該感謝你。如果到時真的賴贏蔡不管有沒贏韓空心菜拉的下臉出來選嘛?這賴皮很瞭解空心蔡愛吃假小心的千金大小姐傲驕心理啊。不是看政治新聞，而是在看社會新聞，詐騙集團內鬨也時有耳聞！。賴的表現差勁低級到沒救。這不是險招，因為賴清德清楚，藍粉們會假性支持賴清德讓蔡英文難堪的。哈哈哈。'

# 【監察院爭議事件】監察院的手可以伸入司法嗎？
new8_content = '監察院近期因發布前立委陳朝龍案件調查報告，指責高等法院判決不符社會經驗、屬於有罪推定一事，在網路上引發熱烈討論，也讓人開始思考監察院這個組織究竟是在做什麼的。在討論監察院的職權問題前，先讓我們來談談究竟監察院在憲法上的地位吧！相信大家一定都聽過孟德斯鳩跟他著名的學說：「三權分立」吧！所謂的三權分立，指的是把政府的權能拆成「行政」、「立法」、「司法」三個權力，並且分別由不同的機關負責、彼此互相制衡，避免單一機關擁有過大的權力。但我國並非採取此種三權分立模式，而是採取(挪抬)孫中山提出的「五權分立」架構，將「考試權」、「監察權」另外獨立成兩權。但現在大多數的學者認為，「考試權」實際上就是「行政權」的一部分；而「監察權」其實就是「立法權」的一部分，因此基本上還是在三權分立的架構內，也有人主張應該要讓考試權及監察權回歸行政權及立法權，走向真正的三權分立。而有關監察權的部分，從組織架構來看，已經從過去由各個省市議會選出監察委員的「間接選舉」模式(憲法第91條參照)，演變成現在的由總統提名、立法院同意的「提名模式」(憲法增修條文第7條參照)。另外從監察委員的組成我們也可以發現，原先監察院的組成其實比較偏向是一個「民意機關」，因此在修憲前，我國一共有立法院、監察院及國民大會等3個民意機關，形成「類似」美國的上下議院、日本的參眾議院的多民意機關模式。但是在憲法修正以後，現在的監察院已經不是民意機關了。監察院的職權方面，在憲法修正之後也有了不同。在憲法第90條中說到，監察院的職權包括了「同意」、「彈劾」、「糾舉」及「審計」4個權利；但在憲法增修條文第7條第1項中，「同意權」已經從監察院的職權中刪除，而僅剩下「彈劾」、「糾舉」、「審計」三權。彈劾權的部分，指的是對公務人員的違法失職行為所做的「拔官」處置；糾舉權則同樣針對公務人員的違法失職，所先行的停職或其他急速處分；而糾正權則是針對行政機關的行政行為的問題，所做出的糾正。此外，監察院除了上述這3項權力以外，還可以行使「調查權」，但監察院的調查權究竟是什麼樣的權利呢？目前仍有爭議，大家可以參考我們在《黨產條例釋憲說明會–監察院聲請解釋的條件是什麼？》一文中的說明。監察院究竟可不可以監督司法機關呢？這必須要區分成兩個層次來討論。首先，是可不可以的問題。基於五權分立而制衡的我國憲法原則，司法權必須受到其他機關的監督。因此監察權在必要的範圍內，應該是可以監督司法權的。這是第一個層次。但是，監察權對司法權的制衡也不是漫無目的。如果監察院監督的是法官的言行表現，例如法官喝花酒、法官利用上班時間招妓、收賄等等，應該是可以進行監督的。但是，如果今天監察權的行使是針對「審判權的核心事項」，例如法官的認事用法、法律見解等等，則不適宜行使監察權，否則就會出現監察權凌駕於司法權的問題。監察院究竟該不該存在，是一直以來都有的問題，而現在也是開始思考這個問題的時候了。首先，監察院的監察權其實是從立法權分割出來的，而過去監察權也確實類似於一個和立法院類似的民意機關。在修憲以後，監察權已經不再是民意機關，而同時從他國的經驗來看，立法權又可以涵括監察院的職權，也可省去思考監察院可不可以監督立法委員的問題理論上有沒有必要再多一個監察院，是我們可以思考的。其次，我們目前的監察院的職權行使是透過監察法，但是監察法的最後一次修正是在1992年，也就是26年前，有些條文的一句甚至已經不存在，例如監察法第5條提到的「憲法增修條文第15條」就不在憲法增修條文裡。而現在監察院的職權行使，反而是透過監察法施行細則在規範。這種以監察院自己訂的「行政規則」來解釋已經不合時宜的「法律」的立法模式究竟好不好呢？也值得我們思考。但是，不論要不要留下監察院，因為監察院的職權是以憲法規定的，所以如果我們想要把監察院拔掉、或更改監察院的職權範圍，勢必得走到修改憲法這一條路，那又一定會是一起大工程。究竟未來監察院何去何從，就讓我們繼續看下去吧！'

new8_review = '該做不做，不該做的一意孤行，這是什麼爛政府丫！。監察院，當然虛造存在!!!!!!!!!!!!! 需要更公正設計的監察院!!!。現在所有的政府機構都以東廠化了，乾脆就廢掉政府，台灣才有機會大破大立。監察院已變東廠 廢了 早該廢了。總統的手都可以伸進監獄了，讓民進黨貪腐前總統陳水扁保外就醫還能手腳靈活到處趴趴走，那為何總統找出來的東廠鷹犬不能插手司法呢？。大家異口同聲痛恨藍綠惡鬥，政客因此有了制衡，行政、立法都可以攤在陽光下，司法才是台灣最落伍黑暗的角落。既然連監察委員都可以被監督了，由監察院來制衡司法濫權，只要設計好監委產生方法與任期，可能也是利大於弊。原來是專欄 我還以為我國突然出現良心記者了。'

# 周錫瑋告知五人小組 國民黨2020沒把握能贏
new9_content = '國民黨2020黨內總統初選五人小組今(23)日拜會前台北縣長周錫瑋，周錫瑋告知五人小組，最近陸續出現黨內同志互相攻訐，他認為民進黨現在執政這麼差，但國民黨在2020仍沒把握能贏，他呼籲黨中央務必阻止黨內互打。國民黨副主席曾永權、郝龍斌，組發會主委李哲華今天到新北市拜會周錫瑋，周錫瑋告訴五人小組，他觀察最近黨內同志已在互相批評，呼籲黨中央未來務必制止。他舉例韓國瑜曾在競爭立委的黨內初選期間批評過同志，當時就被處分，沒有例外，因此大家不要互相批評，要欣賞彼此的優點，將對方意見納入勝選後的政見。周錫瑋說，他希望所有參加總統初選的參選人最後能組成一個競選團隊，大家一條心。他指出，2020大選局勢其實非常艱難，要贏的機率「一半、一半」，民進黨現在做得這麼差，但國民黨仍沒有把握能贏，尤其柯文哲最後可能成為重大影響因素，因此大家不要讓彼此變成敵人、仇人，應該彼此尊重。周錫瑋說，國民黨唯有真正團結，努力贏得人民信心，才是贏得2020的唯一關鍵。他當場承諾兩點，第一、他在初選時絕對不批評同志，若違反願受最嚴厲處分；第二、如果他勝出，請大家支持他，如果他不是第一名，黨可以分派他去任何地方、做任何事，他願意貢獻國民黨這次的選戰。'

new9_review = '一個不服一個,都認為自己最行,從不反省,這就是國民黨問題。有側翼走狗批評就好!完全與本人無關!這錢花得真是值得!。國民黨最大錯誤就是讓步韓，引起諸多糾紛與黨的失格後果，黨必須堅持社會道義，韓背棄誠信如何還能被特徵，理性選民看的下去嗎？。套一句王世堅議員說的給國民黨輸柯賴配當選。別搞知名度了！你有幾兩重。'

new10_content = '資深媒體人黃光芹22日深夜臉書撰文批高雄市長韓國瑜「花天酒地」，並指韓國瑜睡前都要來上一大杯高粱。韓國瑜今（23）日回應時說，自己是有反省能力的人，並駁斥他睡覺前的活動是念彿經，不是喝酒。對此，黃光芹於中午再臉書撰文反擊，字字句句十分嗆辣。韓國瑜也再度有所回應，說他在私生活方面，「我自己心中有一把尺，我不是一個亂來的人。」媒體提問，對黃光芹的指控會有甚麼樣的動作？這是過去的事情嗎？韓國瑜表達，「請放心，這是二十多年前，我當民意代表有些場合，喝酒、聊天、喧嘩等等，但是放心我的私生活方面，我自己知道這一把尺在哪裡。」韓國瑜說，他剛跟說希望黃光芹能找到內心的寧靜與快樂，但是不能一直這樣抹黑，「沒有任何證據把我變成好像十惡不赦之徒，我覺得這樣子是不好的。」韓國瑜又說他也時常鼓勵年輕人，30歲以前不要害怕，六十歲以後不要後悔，人生就是一個階段接著一個階段，再強調了一次「我是一個會反省的人」，年輕的時候，喝酒聊天朋友之間，但是放心，「我自己心中有一把尺，我不是一個亂來的人。」至於是否會對黃光芹提出法律行動？韓國瑜回應，不會動不動就要提法律，這樣要告的人很多，有什麼意義呢？浪費司法資源，他並稱對於這些指控不要說某些立法委員說什麼？某一個人說甚麼？他呼籲黃光芹能直接明確的講出來，並自問自答「我喝酒，我手放哪裡？我當然放酒杯上啊！」'

new10_review = '看看韓國瑜二十幾年前的面相在看看他現在的面相就知道他說的是真話，在看看弘一和尚李叔同是否有幾分神似。 人怕出名,豬怕肥...這幾個月蹭韓的人太多了, 黃小姐蹭過請排隊好嗎??人的心顯於他的面相。'


newfilename = 'new.csv'
f = csv.writer(open(newfilename, "w"))
f.writerow(["sentences", "label_hubert", "label_shaoyuan"])

sentences1 = re.split('，|。', new1_content)
for s in sentences1:
	f.writerow([s])
sentences2 = re.split('，|。', new1_review)
for s in sentences2:
	f.writerow([s])

sentences3 = re.split('，|。', new2_content)
for s in sentences3:
	f.writerow([s])
sentences4 = re.split('，|。', new2_review)
for s in sentences4:
	f.writerow([s])

sentences5 = re.split('，|。', new3_content)
for s in sentences5:
	f.writerow([s])
sentences6 = re.split('，|。', new3_review)
for s in sentences6:
	f.writerow([s])

sentences7 = re.split('，|。', new4_content)
for s in sentences7:
	f.writerow([s])
sentences8 = re.split('，|。', new4_review)
for s in sentences8:
	f.writerow([s])

sentences9 = re.split('，|。', new5_content)
for s in sentences9:
	f.writerow([s])
sentences10 = re.split('，|。', new5_review)
for s in sentences10:
	f.writerow([s])

sentences11 = re.split('，|。', new6_content)
for s in sentences11:
	f.writerow([s])
sentences12 = re.split('，|。', new6_review)
for s in sentences12:
	f.writerow([s])

sentences13 = re.split('，|。', new7_content)
for s in sentences13:
	f.writerow([s])
sentences14 = re.split('，|。', new7_review)
for s in sentences14:
	f.writerow([s])

sentences15 = re.split('，|。', new8_content)
for s in sentences15:
	f.writerow([s])
sentences16 = re.split('，|。', new8_review)
for s in sentences16:
	f.writerow([s])

sentences17 = re.split('，|。', new9_content)
for s in sentences17:
	f.writerow([s])
sentences18 = re.split('，|。', new9_review)
for s in sentences18:
	f.writerow([s])

sentences19 = re.split('，|。', new10_content)
for s in sentences19:
	f.writerow([s])

sentences20 = re.split('，|。', new10_review)
for s in sentences20:
	f.writerow([s])



