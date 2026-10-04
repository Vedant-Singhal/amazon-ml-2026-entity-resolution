# 03 · Blocking recall vs candidate volume (full train)


## Recall and volume per key family and block-size cap

`cap` = drop blocks with more than this many S2/S3 records. `pair_recall` = share of GT pairs sharing at least one kept block. `cands_per_s1` = candidate volume / #S1 (upper bound).

| key | cap | pair_recall | cand_pairs | cands_per_s1 | s1_keyed | other_keyed |
|---|---|---|---|---|---|---|
| name_tok | none | 0.8403 | 67567274401 | 30617.4694 | 0.9923 | 0.9917 |
| name_tok | 20000 | 0.7429 | 15090525002 | 6838.1282 | 0.9923 | 0.9917 |
| name_tok | 5000 | 0.5642 | 2428924592 | 1100.6441 | 0.9923 | 0.9917 |
| name_tok | 1000 | 0.4144 | 283449692 | 128.4425 | 0.9923 | 0.9917 |
| name_tok | 300 | 0.3387 | 105060918 | 47.6074 | 0.9923 | 0.9917 |
| name_tok | 100 | 0.1832 | 21151596 | 9.5846 | 0.9923 | 0.9917 |
| name_tok_pair | none | 0.6150 | 2684829447 | 1216.6050 | 0.8248 | 0.7992 |
| name_tok_pair | 20000 | 0.6112 | 1613469471 | 731.1284 | 0.8248 | 0.7992 |
| name_tok_pair | 5000 | 0.6058 | 489972327 | 222.0263 | 0.8248 | 0.7992 |
| name_tok_pair | 1000 | 0.5914 | 184144214 | 83.4432 | 0.8248 | 0.7992 |
| name_tok_pair | 300 | 0.5761 | 138859756 | 62.9230 | 0.8248 | 0.7992 |
| name_tok_pair | 100 | 0.4908 | 41092115 | 18.6205 | 0.8248 | 0.7992 |
| name_prefix4 | none | 0.8043 | 16266323931 | 7370.9304 | 0.9782 | 0.9814 |
| name_prefix4 | 20000 | 0.7089 | 9310007161 | 4218.7414 | 0.9782 | 0.9814 |
| name_prefix4 | 5000 | 0.4497 | 1122008878 | 508.4277 | 0.9782 | 0.9814 |
| name_prefix4 | 1000 | 0.3019 | 233181985 | 105.6642 | 0.9782 | 0.9814 |
| name_prefix4 | 300 | 0.1851 | 47828035 | 21.6728 | 0.9782 | 0.9814 |
| name_prefix4 | 100 | 0.1051 | 8345571 | 3.7817 | 0.9782 | 0.9814 |
| name_prefix6 | none | 0.7369 | 9124614551 | 4134.7325 | 0.9322 | 0.9464 |
| name_prefix6 | 20000 | 0.6866 | 5722952367 | 2593.3016 | 0.9322 | 0.9464 |
| name_prefix6 | 5000 | 0.5239 | 641133748 | 290.5237 | 0.9322 | 0.9464 |
| name_prefix6 | 1000 | 0.4529 | 235253575 | 106.6029 | 0.9322 | 0.9464 |
| name_prefix6 | 300 | 0.3545 | 74326906 | 33.6805 | 0.9322 | 0.9464 |
| name_prefix6 | 100 | 0.2455 | 18087750 | 8.1963 | 0.9322 | 0.9464 |
| name_metaphone | none | 0.8801 | 90349747912 | 40941.1311 | 0.9874 | 0.9837 |
| name_metaphone | 20000 | 0.7135 | 16462185686 | 7459.6833 | 0.9874 | 0.9837 |
| name_metaphone | 5000 | 0.4856 | 2819866401 | 1277.7957 | 0.9874 | 0.9837 |
| name_metaphone | 1000 | 0.2466 | 253521445 | 114.8808 | 0.9874 | 0.9837 |
| name_metaphone | 300 | 0.1337 | 45870278 | 20.7857 | 0.9874 | 0.9837 |
| name_metaphone | 100 | 0.0621 | 8027851 | 3.6377 | 0.9874 | 0.9837 |
| addr_num+tok | none | 0.6771 | 3460433846 | 1568.0628 | 0.9584 | 0.9015 |
| addr_num+tok | 20000 | 0.6769 | 3321917538 | 1505.2954 | 0.9584 | 0.9015 |
| addr_num+tok | 5000 | 0.6744 | 1833946273 | 831.0354 | 0.9584 | 0.9015 |
| addr_num+tok | 1000 | 0.6666 | 542875807 | 245.9990 | 0.9584 | 0.9015 |
| addr_num+tok | 300 | 0.6569 | 195207088 | 88.4562 | 0.9584 | 0.9015 |
| addr_num+tok | 100 | 0.6434 | 80066458 | 36.2814 | 0.9584 | 0.9015 |
| addr_tok_pair | none | 0.7647 | 35141280209 | 15923.9377 | 0.9184 | 0.9008 |
| addr_tok_pair | 20000 | 0.7564 | 9357230014 | 4240.1400 | 0.9184 | 0.9008 |
| addr_tok_pair | 5000 | 0.7450 | 3306111038 | 1498.1329 | 0.9184 | 0.9008 |
| addr_tok_pair | 1000 | 0.7194 | 907913591 | 411.4124 | 0.9184 | 0.9008 |
| addr_tok_pair | 300 | 0.6925 | 306081203 | 138.6978 | 0.9184 | 0.9008 |
| addr_tok_pair | 100 | 0.6522 | 112496261 | 50.9766 | 0.9184 | 0.9008 |


## Union of all key families at cap=1000

Pair recall (union): **0.9702**

| country | src | pairs | recall | only_name_tok | only_name_tok_pair | only_name_prefix4 | only_name_prefix6 | only_name_metaphone | only_addr_num+tok | only_addr_tok_pair |
|---|---|---|---|---|---|---|---|---|---|---|
| India | S2 | 1480545 | 0.9568 | 0.0018 | 0.0059 | 0.0007 | 0.0013 | 0.0000 | 0.0465 | 0.0975 |
| India | S3 | 1579298 | 0.9521 | 0.0034 | 0.0160 | 0.0013 | 0.0033 | 0.0001 | 0.0486 | 0.0759 |
| US | S2 | 2213074 | 0.9802 | 0.0006 | 0.0343 | 0.0007 | 0.0022 | 0.0000 | 0.0327 | 0.0317 |
| US | S3 | 2365448 | 0.9813 | 0.0005 | 0.0323 | 0.0006 | 0.0020 | 0.0000 | 0.0328 | 0.0311 |

**Greedy key selection** (each step adds the family with the largest recall gain):

| step | add | recall | cands_per_s1_ub |
|---|---|---|---|
| 1 | addr_tok_pair | 0.7194 | 411.4124 |
| 2 | name_tok_pair | 0.8886 | 494.8556 |
| 3 | addr_num+tok | 0.9500 | 740.8547 |
| 4 | name_prefix6 | 0.9639 | 847.4576 |
| 5 | name_tok | 0.9693 | 975.9001 |
| 6 | name_prefix4 | 0.9701 | 1081.5643 |
| 7 | name_metaphone | 0.9702 | 1196.4452 |

Non-singleton S1 entities with ALL matches recoverable: 0.9190; mean fraction recoverable: 0.9699


## Sample of GT pairs missed by every key family

| s1_name | s1_addr | o_name | o_addr |
|---|---|---|---|
| blue foundation | pn 473 besment vidyut nagar a ajmer road jaipur rajasthan | bluu phaaunddeshn | block e 159 pn 473 jaipur rajasthan |
| jones s associates | 3611 rosedale avenue university park tx | jones s associates | 3611 rosedle ave po box 3175 dalllas city tx |
| universal services private limited | kadel house near sunrise english medium school babubasa debidanga p o champasari anch al siliguri west bengal | universal seolvices private limited | h no 689 kadel hguse siliguri al west bengal |
| south hospitality pvt ltd | c 343 new ashok nagar vasundhra enclave east delhi delhi | saauth honspittailittii praa li | door no a 111 c 343 east delhi दिल्ली |
| anilita brothers | lgf 16 a south ex part 2 ndse ii south delhi d 15 delhi new delhi | drexnovisyn | d 15 south delhi new delhi दिल्ली |
| jay developers private limited | first floor mulratna 334 narshi natha street masjid west mumbai mumbai city maharashtra | jy ddevlprs praaivett limittedd | firt floor mumbai mumbai city महाराष्ट्र |
| w a s sunshine inc | 2501 fair meadows drive unit 140 springfield oh | w a s |  |
| sg mohan private limited | 104 block 16 suncity apartment sarjapur outer ring road junction ibblu r bangalore karnataka | sg mlnhhn private limited | ಕರ್ನಾಟಕ 411 104 block 16 suncity apartment sarjapur outer ring road junction ibblu r bangalore |
| laxmi estate private limited | 6 3 249 5 a road no 1banjarahills hyderabad hyderabad telangana | lkssmii estteett praiveett limittedd | h no 6 3 249 5 a road no 1banjarahills hyderabad hyderabad తెలంగాణ |
| drift it pvt ltd | b 1409 13f 7th avenue gaur city 1 gh 01 sector ghaziabad uttar pradesh | dotft it pvt ltd | plot 470 b 1409 13f ghaziabad uttar pradesh |
| southern york llc | 7755 rd 37 mansfield oh | southernyork com | rd 37   mansfield oh |
| pediatric dental care associates llc | 5616 bernadette avenue frisco tx | pediatric dental care care |  |
| an trading pc | 571 highway 394 unit 20 blountville tn | an trading pc | 571 highway 394 pmb 4561 blountille tn |
| fortune shyam investments private limited | second floor main rohtak road h no b 37 delhi north west delhi | phonrcyuun shyaam investtmentts praaivett limittedd | delhi h no b 37 delhi north west delhi |
| global investment private limited | no 1083 2nd floor prince arcade 5th cross 12th main indiranagar bangalore north bangalore karnataka | gloobl investtmentt praiveett limittedd | no 4083 2nd floor bangalore h a l second stage karnataka |
| internal medicine physicians associates | 120 ashley lane powell tn | internal medicine physicians associates ltd |  |
| orthopedic clinic | 7879 university avenue unit apartment 206 mn spring lake park | orthopedic center |  |
| secure polska llc | 2902 san ygnacio road albuquerque nm | secure llc center | 2902 san ygacio road nm south valley |
| bari meeks d d s p c | cincinnati 4332 8th street oh | meeksddsbari com | 4332 eighth street oh cincinnti city |
| urban cafe p c | 24 k street montague ma | urban cfhade p c | turners falls city 24 28 k street massachusetts |
| c x a total | ct fairfield 1032 valley road | c x total total a |  |
| gold properties private limited | sh f 6 rainbow busithane pl 33 sanpada new mumbai thane maharashtra | dr gold private limited partners | sh f 6   navi mumbai region thane mh |
| lotech inc | 570 5th avenue new rochelle ny unit apartment b5 | l0tech inc | 0257 5th ave new rochelle new york |
| dean capital partners | 27 kensington circle unit 208 wheaton il | dean partners center |  |
| alpha constructions | g 30 sec 7 extn dwarka new delhi delhi | alphaa knsttrkshns | pdot 995 g 30 new delhi dl |
| raj ventures private limited | c o miryala venkat reddy santhosh nagar nalgonda telangana | dr raj vetnueees private limited |  |
| ironclad pc | nc 823 center street hickory | ironclad pc |  |
| premier business partners inc | 31021 172nd avenue auburn wa | premier business partners inc | 31021 172rd ave washington aubunr |
| one constructions private limited | no 7 1 4th cross jaijavan nagar c banasawadi road bangalore karnataka | on knsttrkssns praiveett limittedd | door no 7 1 bengaluru bangalore ka |
| ember llc | unit apt 3e ak anchorage 800 m street | ember l l c | 800 m st unit apt 3e anchorage alaska |
| federal materials industries llc | 50 grayhawk way silver spring township pa | federal material5 industries llc | 0050 grayhawk way mechanicsbuurg pennsylvania |
| ace energy private limited | c o shri raju a kalmegh chausala anjangaonsurji anjangaon amravati maharashtra | es enrjii praaivett limittedd | c o shri raju a kalmegh amravati mh |
| rq solutions partners | shree shankeshwar properties shop no 9 10 15 16 b wing ground floor mahalaxmi heights surv pune city pune maharashtra | rq solutions partners corp | 464 shree shankeshwar properties pune city pune mh |
| krishna technology private limited | delhi s f office no 227 delhi north west delhi dda cc rg complex p1 s 14 rohini | sri krishna technofogy private limited |  |
| consultancy india industries private limited | jaipur rajasthan sanjay nagar b plot no 262 jaipur | consultancy india industries private ltd |  |
| sides mountain ampercap pc | 2610 berry road lucas oh | msampercap com | berry rd monroe twp ohio |
| stewart and willingham llc | 324 10 street sartell mn | stewartwillingham | 00324 10 street sartell city minnesota |
| cancer center iii | 2238 31 needham in | cancer center | 002238 31   needham indiana |
| complete construction global llc | 1363 big bill road arnold mo | complete global llc center |  |
| foundry llc | 2103 lee drive el campo tx | the foundry llc | 2104 lee drive el campo texas |


## Test set: candidate volume per S1 by country (cap=1000)

| key | US | France | India |
|---|---|---|---|
| name_tok | 148.8 | 133.9 | 150.4 |
| name_tok_pair | 90.5 | 83.5 | 59.6 |
| name_prefix4 | 100.9 | 89.7 | 116.1 |
| name_prefix6 | 91.6 | 79.3 | 126.1 |
| name_metaphone | 125.1 | 146.0 | 143.5 |
| addr_num+tok | 21.1 | 520.8 | 578.0 |
| addr_tok_pair | 112.3 | 930.2 | 855.7 |

| key | US | France | India |
|---|---|---|---|
| name_tok | 0.999 | 0.997 | 0.983 |
| name_tok_pair | 0.899 | 0.899 | 0.713 |
| name_prefix4 | 0.996 | 0.992 | 0.952 |
| name_prefix6 | 0.972 | 0.981 | 0.873 |
| name_metaphone | 0.996 | 0.995 | 0.974 |
| addr_num+tok | 0.996 | 0.995 | 0.902 |
| addr_tok_pair | 0.864 | 1.000 | 0.998 |
