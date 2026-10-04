# 01 · Data profile


## Field quality and noise markers (fraction of rows)

Per split/source/country. `name_non_latin` = contains a non-Latin script letter; `name_domain` = looks like a web domain; `addr_5digit`/`addr_6digit` = US ZIP / India PIN candidates; `addr_near` = landmark-style references.

| split | source | country | rows | name_empty | addr_empty | name_len_p50 | name_tokens_mean | addr_len_p50 | addr_parts_mean | name_non_ascii | name_non_latin | addr_non_latin | name_all_caps |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| train | S1 | India | 883188 | 0.000 | 0.000 | 27.000 | 3.741 | 76.000 | 5.659 | 0.000 | 0.000 | 0.000 | 0.000 |
| train | S1 | US | 1323633 | 0.000 | 0.000 | 22.000 | 3.419 | 34.000 | 3.158 | 0.000 | 0.000 | 0.000 | 0.000 |
| train | S2 | India | 2017799 | 0.000 | 0.029 | 27.000 | 3.810 | 68.000 | 4.868 | 0.279 | 0.235 | 0.237 | 0.149 |
| train | S2 | US | 3016817 | 0.000 | 0.037 | 23.000 | 3.471 | 32.000 | 2.998 | 0.067 | 0.000 | 0.000 | 0.215 |
| train | S3 | India | 2115547 | 0.000 | 0.031 | 27.000 | 3.794 | 58.000 | 4.731 | 0.185 | 0.132 | 0.225 | 0.026 |
| train | S3 | US | 3170056 | 0.000 | 0.035 | 23.000 | 3.528 | 39.000 | 3.118 | 0.068 | 0.000 | 0.000 | 0.032 |
| test | S1 | France | 259452 | 0.000 | 0.000 | 19.000 | 3.112 | 48.000 | 3.039 | 0.157 | 0.000 | 0.000 | 0.000 |
| test | S1 | India | 809986 | 0.000 | 0.000 | 27.000 | 3.740 | 76.000 | 5.664 | 0.000 | 0.000 | 0.000 | 0.000 |
| test | S1 | US | 663106 | 0.000 | 0.000 | 22.000 | 3.417 | 34.000 | 3.158 | 0.000 | 0.000 | 0.000 | 0.000 |
| test | S2 | France | 703378 | 0.000 | 0.031 | 20.000 | 3.192 | 40.000 | 2.639 | 0.245 | 0.000 | 0.000 | 0.208 |
| test | S2 | India | 2312565 | 0.000 | 0.023 | 28.000 | 3.918 | 68.000 | 4.893 | 0.276 | 0.236 | 0.238 | 0.141 |
| test | S2 | US | 1871330 | 0.000 | 0.029 | 24.000 | 3.591 | 32.000 | 3.014 | 0.062 | 0.000 | 0.000 | 0.204 |
| test | S3 | France | 731615 | 0.000 | 0.029 | 20.000 | 3.188 | 41.000 | 2.657 | 0.239 | 0.000 | 0.000 | 0.056 |
| test | S3 | India | 2405000 | 0.000 | 0.025 | 28.000 | 3.904 | 59.000 | 4.749 | 0.182 | 0.133 | 0.229 | 0.025 |
| test | S3 | US | 1945701 | 0.000 | 0.028 | 24.000 | 3.636 | 39.000 | 3.133 | 0.064 | 0.000 | 0.000 | 0.032 |

| split | source | country | name_domain | name_compact_no_space | name_has_digit | name_paren | name_dba | name_junk_prefix | addr_null_token | addr_5digit | addr_6digit | addr_has_number | addr_po_box | addr_near |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| train | S1 | India | 0.000 | 0.000 | 0.001 | 0.051 | 0.000 | 0.001 | 0.000 | 0.003 | 0.000 | 0.913 | 0.000 | 0.133 |
| train | S1 | US | 0.000 | 0.000 | 0.026 | 0.000 | 0.000 | 0.001 | 0.000 | 0.108 | 0.001 | 1.000 | 0.000 | 0.000 |
| train | S2 | India | 0.034 | 0.031 | 0.031 | 0.096 | 0.000 | 0.017 | 0.022 | 0.011 | 0.000 | 0.913 | 0.000 | 0.113 |
| train | S2 | US | 0.044 | 0.047 | 0.064 | 0.065 | 0.000 | 0.025 | 0.029 | 0.101 | 0.014 | 0.902 | 0.033 | 0.000 |
| train | S3 | India | 0.036 | 0.034 | 0.034 | 0.101 | 0.014 | 0.018 | 0.021 | 0.010 | 0.000 | 0.901 | 0.000 | 0.086 |
| train | S3 | US | 0.042 | 0.046 | 0.063 | 0.066 | 0.022 | 0.024 | 0.028 | 0.102 | 0.013 | 0.913 | 0.031 | 0.000 |
| test | S1 | France | 0.000 | 0.000 | 0.008 | 0.080 | 0.000 | 0.001 | 0.000 | 0.004 | 0.000 | 0.996 | 0.000 | 0.000 |
| test | S1 | India | 0.000 | 0.000 | 0.001 | 0.052 | 0.000 | 0.001 | 0.000 | 0.003 | 0.000 | 0.913 | 0.000 | 0.132 |
| test | S1 | US | 0.000 | 0.000 | 0.026 | 0.000 | 0.000 | 0.001 | 0.000 | 0.108 | 0.001 | 1.000 | 0.000 | 0.000 |
| test | S2 | France | 0.035 | 0.040 | 0.011 | 0.102 | 0.000 | 0.009 | 0.000 | 0.005 | 0.000 | 0.931 | 0.000 | 0.000 |
| test | S2 | India | 0.027 | 0.025 | 0.029 | 0.095 | 0.000 | 0.015 | 0.022 | 0.010 | 0.000 | 0.930 | 0.000 | 0.114 |
| test | S2 | US | 0.036 | 0.039 | 0.062 | 0.062 | 0.000 | 0.023 | 0.029 | 0.103 | 0.014 | 0.921 | 0.034 | 0.000 |
| test | S3 | France | 0.034 | 0.040 | 0.011 | 0.100 | 0.012 | 0.009 | 0.000 | 0.005 | 0.000 | 0.934 | 0.000 | 0.000 |
| test | S3 | India | 0.030 | 0.027 | 0.032 | 0.101 | 0.011 | 0.016 | 0.021 | 0.010 | 0.000 | 0.919 | 0.000 | 0.085 |
| test | S3 | US | 0.035 | 0.037 | 0.061 | 0.065 | 0.018 | 0.023 | 0.028 | 0.104 | 0.014 | 0.929 | 0.032 | 0.000 |


## Script mix (fraction of rows containing each script)

| split | source | field | country | Latin | Devanagari | Bengali | Gujarati | Gurmukhi | Tamil | Telugu | Kannada | Malayalam | Oriya | Arabic |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| train | S1 | business_name | India | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| train | S1 | business_address | India | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| train | S2 | business_name | India | 0.774 | 0.134 | 0.015 | 0.015 | 0.003 | 0.017 | 0.019 | 0.018 | 0.009 | 0.004 | 0.000 |
| train | S2 | business_address | India | 0.971 | 0.138 | 0.016 | 0.016 | 0.004 | 0.017 | 0.017 | 0.019 | 0.008 | 0.003 | 0.000 |
| train | S3 | business_name | India | 0.885 | 0.075 | 0.009 | 0.009 | 0.002 | 0.009 | 0.011 | 0.010 | 0.005 | 0.002 | 0.000 |
| train | S3 | business_address | India | 0.969 | 0.130 | 0.015 | 0.015 | 0.003 | 0.017 | 0.016 | 0.018 | 0.007 | 0.003 | 0.000 |
| test | S1 | business_name | France | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| test | S1 | business_name | India | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| test | S1 | business_address | France | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| test | S1 | business_address | India | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| test | S2 | business_name | France | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| test | S2 | business_name | India | 0.773 | 0.134 | 0.015 | 0.015 | 0.003 | 0.017 | 0.020 | 0.019 | 0.010 | 0.004 | 0.000 |
| test | S2 | business_address | France | 0.969 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| test | S2 | business_address | India | 0.977 | 0.138 | 0.016 | 0.016 | 0.004 | 0.018 | 0.017 | 0.019 | 0.008 | 0.003 | 0.000 |
| test | S3 | business_name | India | 0.883 | 0.075 | 0.009 | 0.009 | 0.002 | 0.009 | 0.011 | 0.011 | 0.005 | 0.002 | 0.000 |
| test | S3 | business_name | France | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| test | S3 | business_address | India | 0.975 | 0.133 | 0.015 | 0.015 | 0.003 | 0.017 | 0.016 | 0.019 | 0.007 | 0.003 | 0.000 |
| test | S3 | business_address | France | 0.971 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |


## Most frequent name tokens (train S1 vs S2 vs S3, and test France)

**train S1**

| country | tokens |
|---|---|
| India | limited(59.1%), private(48.9%), ltd(16.5%), pvt(13.8%), india(6.9%), llp(4.4%), services(2.7%), solutions(2.4%), brothers(2.3%), trading(2.3%), co(2.2%), technologies(1.9%), international(1.7%), foundation(1.6%), global(1.6%), tech(1.5%), industries(1.5%), enterprises(1.4%), consultants(1.4%), technology(1.4%), consultancy(1.3%), developers(1.3%), ventures(1.3%), traders(1.3%), care(1.3%) |
| US | llc(26.9%), inc(18.0%), and(4.3%), s(4.2%), c(4.1%), l(3.7%), care(3.2%), of(2.8%), associates(2.7%), center(2.4%), group(2.4%), partners(2.2%), p(2.2%), d(2.1%), corp(2.1%), pc(2.0%), health(2.0%), clinic(1.6%), pllc(1.6%), medicine(1.3%), lp(1.2%), pediatric(1.2%), dental(1.0%), family(0.9%), global(0.9%) |

**train S2**

| country | tokens |
|---|---|
| India | limited(27.0%), private(27.0%), ltd(15.3%), pvt(9.4%), लिमिटेड(9.4%), प्राइवेट(7.8%), india(5.8%), services(3.7%), com(3.4%), llp(3.2%), center(2.9%), co(2.2%), industries(1.9%), enterprises(1.9%), group(1.8%), brothers(1.8%), प्रा(1.8%), लि(1.8%), public(1.8%), ventures(1.7%), solutions(1.7%), లిమిటెడ్(1.6%), ಲಿಮಿಟೆಡ್(1.6%), trading(1.5%), exports(1.5%) |
| US | llc(17.6%), inc(13.8%), l(5.1%), center(4.8%), partners(4.4%), com(4.4%), corp(4.1%), and(4.0%), c(3.9%), s(3.8%), group(3.6%), co(3.3%), ltd(3.1%), care(2.6%), of(2.6%), services(2.5%), holdings(2.4%), associates(2.2%), d(1.7%), health(1.7%), lp(1.6%), p(1.5%), corporation(1.4%), the(1.4%), clinic(1.3%) |

**train S3**

| country | tokens |
|---|---|
| India | limited(32.9%), private(31.2%), ltd(16.3%), pvt(10.5%), india(5.7%), लिमिटेड(5.0%), services(4.3%), प्राइवेट(4.2%), com(3.7%), llp(3.6%), center(3.6%), co(2.2%), enterprises(2.1%), industries(2.1%), trading(1.9%), ventures(1.9%), solutions(1.8%), group(1.8%), brothers(1.8%), public(1.7%), exports(1.6%), s(1.4%), m(1.4%), mr(1.4%), sri(1.4%) |
| US | llc(18.2%), inc(13.8%), center(4.9%), l(4.6%), partners(4.4%), com(4.2%), and(3.9%), corp(3.9%), s(3.8%), c(3.7%), group(3.5%), co(3.2%), ltd(2.7%), care(2.6%), of(2.6%), services(2.5%), holdings(2.3%), a(2.2%), associates(2.2%), d(2.1%), health(1.7%), the(1.6%), p(1.5%), lp(1.5%), service(1.3%) |

**test S2**

| country | tokens |
|---|---|
| France | sarl(20.9%), sas(14.1%), france(10.0%), club(7.4%), s(7.1%), de(6.1%), eurl(5.9%), ecole(5.1%), sa(4.9%), amicale(4.8%), comite(4.7%), sasu(4.4%), a(4.0%), sci(3.9%), groupe(3.8%), du(3.7%), maison(3.6%), com(3.5%), developpement(3.4%), centre(3.4%), fils(3.1%), union(3.0%), des(2.7%), sportive(2.7%), cie(2.6%) |
| India | limited(28.1%), private(26.8%), ltd(15.2%), pvt(9.5%), लिमिटेड(9.4%), प्राइवेट(7.8%), india(5.9%), llp(3.6%), services(3.6%), com(2.8%), center(2.6%), industries(2.6%), enterprises(2.6%), group(2.5%), public(2.4%), ventures(2.4%), co(2.2%), exports(2.1%), holdings(2.0%), infratech(1.8%), brothers(1.8%), overseas(1.8%), प्रा(1.8%), लि(1.8%), solutions(1.7%) |
| US | llc(17.1%), inc(13.8%), partners(5.6%), corp(5.1%), group(4.7%), l(4.7%), center(4.7%), co(4.3%), ltd(4.1%), and(4.0%), s(3.8%), c(3.7%), com(3.6%), holdings(3.5%), care(2.8%), of(2.6%), services(2.5%), associates(2.2%), d(1.8%), health(1.7%), lp(1.5%), p(1.4%), corporation(1.4%), the(1.3%), clinic(1.3%) |


## Most frequent address tokens

**train S1**

| country | tokens |
|---|---|
| India | no(50.4%), delhi(36.5%), road(25.0%), maharashtra(21.8%), nagar(18.9%), floor(18.3%), mumbai(14.5%), west(14.0%), pradesh(13.2%), 1(13.0%), new(11.6%), c(11.4%), a(10.0%), plot(9.9%), bangalore(9.5%), 2(9.4%), b(9.1%), uttar(8.6%), o(8.5%), city(8.3%), karnataka(8.0%), tamil(7.2%), nadu(7.2%), pune(6.6%), gujarat(6.6%) |
| US | street(20.7%), road(18.6%), drive(16.7%), unit(15.0%), avenue(13.7%), tx(10.1%), ny(7.7%), nc(7.2%), lane(7.1%), oh(6.4%), city(5.7%), il(5.7%), tn(4.5%), va(4.5%), court(4.4%), ma(4.3%), az(4.3%), in(3.7%), county(3.5%), wa(3.2%), md(3.1%), ca(2.8%), of(2.7%), al(2.5%), wi(2.4%) |

**train S2**

| country | tokens |
|---|---|
| India | no(55.4%), delhi(24.3%), road(21.5%), floor(15.9%), maharashtra(15.9%), nagar(15.1%), c(12.2%), 1(11.5%), a(11.4%), pradesh(10.8%), new(9.5%), mumbai(9.4%), h(9.4%), west(9.0%), plot(9.0%), b(8.6%), 2(8.6%), o(8.0%), bangalore(6.3%), uttar(6.3%), 3(6.0%), karnataka(5.9%), s(5.8%), kolkata(5.7%), door(5.6%) |
| US | st(10.6%), rd(9.9%), tx(9.7%), dr(9.0%), street(8.8%), road(8.2%), ny(7.5%), ave(7.4%), drive(7.1%), nc(6.9%), city(6.8%), oh(6.2%), avenue(5.9%), il(5.5%), va(4.3%), tn(4.3%), ma(4.1%), az(4.1%), ln(3.8%), ct(3.6%), in(3.6%), wa(3.1%), lane(3.1%), md(2.9%), null(2.9%) |

**train S3**

| country | tokens |
|---|---|
| India | no(50.7%), delhi(21.1%), road(16.7%), mh(15.2%), nagar(13.5%), mumbai(12.8%), floor(12.5%), c(11.5%), a(10.4%), 1(10.3%), dl(9.5%), new(9.5%), h(8.7%), bangalore(8.2%), plot(7.9%), b(7.8%), 2(7.6%), o(7.6%), city(6.7%), west(6.5%), up(5.8%), pune(5.8%), ka(5.7%), 3(5.4%), south(5.4%) |
| US | st(10.1%), rd(9.5%), street(9.4%), texas(9.3%), new(8.8%), road(8.7%), dr(8.6%), north(7.7%), drive(7.6%), york(7.2%), ave(7.1%), carolina(6.8%), city(6.8%), avenue(6.3%), unit(6.1%), ohio(5.9%), virginia(5.5%), illinois(5.3%), tennessee(4.2%), massachusetts(4.0%), arizona(4.0%), washington(3.9%), ln(3.7%), 1(3.5%), indiana(3.5%) |


## Ground truth: matches per S1 entity

| country | entities | singleton_rate | mean_matches | p90 | max | mean_S2 | mean_S3 | no_S2 | no_S3 |
|---|---|---|---|---|---|---|---|---|---|
| India | 883188 | 0.056 | 3.465 | 6.000 | 11 | 1.676 | 1.788 | 0.130 | 0.121 |
| US | 1323633 | 0.056 | 3.459 | 6.000 | 11 | 1.672 | 1.787 | 0.131 | 0.121 |

| k | US | India |
|---|---|---|
| 0 | 73896 | 49351 |
| 1 | 71689 | 47468 |
| 2 | 225285 | 149927 |
| 3 | 318876 | 211965 |
| 4 | 290446 | 193669 |
| 5 | 192618 | 129339 |
| 6 | 98406 | 66462 |
| 7 | 38299 | 25669 |
| 8 | 11217 | 7463 |
| 9 | 2534 | 1671 |
| 10 | 347 | 187 |
| 11 | 20 | 17 |


## Ground truth: consistency checks

- GT pairs: 7,638,365

- S2/S3 ids matched to more than one S1: 0 (0 ⇒ clusters are disjoint)

- GT ids missing from source files: 0

- GT pairs with country mismatch: 0 (0 ⇒ country can be a hard blocking constraint)

- GT pairs whose S2/S3 record has an empty address: 4.4%


## Distractors: S2/S3 records that match nothing

| src | country | records | unmatched_rate |
|---|---|---|---|
| S2 | India | 2017799 | 0.266 |
| S2 | US | 3016817 | 0.266 |
| S3 | India | 2115547 | 0.253 |
| S3 | US | 3170056 | 0.254 |


## Exact duplicates

- train S1: duplicate folded names 31.09%, duplicate name+address 0.00%

- train S2: duplicate folded names 21.55%, duplicate name+address 1.61%

- train S3: duplicate folded names 20.71%, duplicate name+address 1.18%

- test S1: duplicate folded names 29.15%, duplicate name+address 0.00%

- test S2: duplicate folded names 19.49%, duplicate name+address 1.45%

- test S3: duplicate folded names 18.78%, duplicate name+address 1.08%


## Train → test drift

Country mix of S1 and the S2+S3 : S1 ratio (candidate pool density).

| split | country | s1 | others | ratio |
|---|---|---|---|---|
| test | France | 259452 | 1434993 | 5.531 |
| test | India | 809986 | 4717565 | 5.824 |
| test | US | 663106 | 3817031 | 5.756 |
| train | India | 883188 | 4133346 | 4.680 |
| train | US | 1323633 | 6186873 | 4.674 |
