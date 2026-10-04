# 02 · Pair analysis: matches vs hard negatives


## What true matches look like (positives only)

| country | src | pairs | core_name_exact | name_tset_ge90 | name_tset_lt50 | other_domain | other_non_latin | other_addr_empty | first_num_eq | addr_tset_ge80 |
|---|---|---|---|---|---|---|---|---|---|---|
| India | S2 | 26823 | 0.559 | 0.586 | 0.053 | 0.046 | 0.231 | 0.038 | nan | 0.975 |
| India | S3 | 28844 | 0.627 | 0.658 | 0.060 | 0.047 | 0.132 | 0.040 | nan | 0.896 |
| US | S2 | 39842 | 0.712 | 0.793 | 0.030 | 0.059 | 0.000 | 0.047 | nan | 0.933 |
| US | S3 | 42954 | 0.722 | 0.803 | 0.030 | 0.053 | 0.000 | 0.045 | nan | 0.822 |

**Name-dissimilar matches (n_tset < 50):** how often does the address rescue them?

| country | n | addr_empty | addr_tset_ge80 | first_num_eq | domain | non_latin |
|---|---|---|---|---|---|---|
| India | 3144 | 0.000 | 0.938 | nan | 0.288 | 0.268 |
| US | 2459 | 0.001 | 0.869 | nan | 0.364 | 0.000 |


## Transliteration effect (positives whose other name is non-Latin)

10,002 pairs. Mean fuzz.ratio raw = 13.1, after transliteration = 70.9, core tf-idf cos = 18.4


## Per-feature separability (AUC; NaN rows dropped)

| feature | country | auc_vs_hard | auc_vs_random | pos_p10 | pos_p50 | hard_p50 | hard_p90 |
|---|---|---|---|---|---|---|---|
| a_tset | ALL | 0.969 | 0.999 | 79.167 | 94.444 | 38.095 | 62.963 |
| a_partial | ALL | 0.960 | 0.999 | 68.493 | 89.600 | 41.270 | 63.158 |
| a_ratio | ALL | 0.943 | 0.985 | 52.778 | 84.615 | 36.257 | 53.488 |
| a_num_jacc | ALL | 0.905 | 0.910 | 0.000 | 1.000 | 0.000 | 0.000 |
| a_first_num_eq | ALL | 0.889 | 0.894 | 0.000 | 1.000 | 0.000 | 0.000 |
| n_ratio | ALL | 0.791 | 0.976 | 63.636 | 87.879 | 65.116 | 88.889 |
| n_tsort | ALL | 0.791 | 0.974 | 60.000 | 90.000 | 65.625 | 89.286 |
| core_compact_ratio | ALL | 0.783 | 0.978 | 55.556 | 100.000 | 62.069 | 100.000 |
| n_jw | ALL | 0.766 | 0.976 | 75.180 | 94.783 | 84.558 | 95.294 |
| n_ratio_raw | ALL | 0.750 | 0.914 | 50.000 | 87.805 | 65.116 | 88.889 |
| core_tfidf_cos | ALL | 0.718 | 0.985 | 37.813 | 100.000 | 62.019 | 100.000 |
| n_partial | ALL | 0.708 | 0.982 | 72.414 | 100.000 | 80.000 | 100.000 |
| core_compact_partial | ALL | 0.686 | 0.987 | 72.727 | 100.000 | 82.353 | 100.000 |
| n_tset | ALL | 0.662 | 0.975 | 66.667 | 100.000 | 86.364 | 100.000 |
| core_tset | ALL | 0.648 | 0.975 | 56.000 | 100.000 | 80.000 | 100.000 |
| is_s3 | ALL | 0.545 | 0.503 | 0.000 | 1.000 | 0.000 | 1.000 |
| a_other_empty | ALL | 0.503 | 0.505 | 0.000 | 0.000 | 0.000 | 0.000 |
| len_diff | ALL | 0.263 | 0.229 | 0.000 | 2.000 | 7.000 | 18.000 |

| feature | India | US |
|---|---|---|
| n_ratio | 0.726 | 0.827 |
| n_ratio_raw | 0.632 | 0.827 |
| n_partial | 0.634 | 0.749 |
| n_tsort | 0.731 | 0.825 |
| n_tset | 0.563 | 0.722 |
| n_jw | 0.706 | 0.800 |
| core_tset | 0.569 | 0.700 |
| core_compact_ratio | 0.718 | 0.820 |
| core_compact_partial | 0.622 | 0.727 |
| core_tfidf_cos | 0.632 | 0.775 |
| a_ratio | 0.925 | 0.953 |
| a_tset | 0.970 | 0.970 |
| a_partial | 0.959 | 0.960 |
| a_num_jacc | 0.934 | 0.889 |
| a_first_num_eq | 0.852 | 0.914 |
| a_other_empty | 0.501 | 0.503 |
| len_diff | 0.294 | 0.243 |
| is_s3 | 0.540 | 0.548 |


## Hardest positives (lowest name similarity)

| country | src | business_name | business_address | business_name_o | business_address_o | n_tset | a_tset |
|---|---|---|---|---|---|---|---|
| US | S2 | Bay Fund LLC | 11210 Nolen Boulevard, Grand Bay, AL | Veovio | 11210 NOLEN BOULEVARD, GRAND BAY, AL | 0 | 100 |
| India | S2 | Gt & Sons | Chauki No. 2 (311) Musepur, Rewari, Haryana | Yumawex | REWARI, हरियाणा, PRATAPGARH, CHAUKI NO. ##2 (311) MUSEPUR | 0 | 88 |
| US | S2 | Cypress Inc. | 2400 Hat Bender Loop, TX, Round Rock | Lumlum | TX, ROUND ROCK, 2400 HAT BENDER LOOP | 0 | 100 |
| India | S2 | Singhania Sangh | A 342 Manav Apartmentspaschim Vihar, New Delhi, North Delhi, Delhi | Fluxecto | दिल्ली, null, NORTH DELHI, A 342 MANAV APARTMENTSPASCHIM VIHAR, NEW DELHI | 0 | 100 |
| US | S3 | Summit Fund Inc. | 5950 Fairview Road, Unit 550, Charlotte, NC | Halozephwex | 5950 Fairview Road, Unit 550, Charlotte, NC | 0 | 100 |
| India | S3 | Ambay & Co | 786/3, Surbhi Colony, Shegaon, Buldhana, Maharashtra | Gildzeph | Buldhana, 786, Surbhi Colony, Shegaon, Buldhana, MH | 0 | 96 |
| India | S3 | Jatadhari Sangh | D-196, Second Floor Saket, New Delhi, Delhi | Lumlum | D-196, New Delhi, Delhi, Second Floor Saket | 0 | 100 |
| US | S3 | Opto LLC | 1 Macneil Court, Bolingbrook, IL | Nexjax | 1 Macneil Ct, Bolingbrook, Illinois | 0 | 86 |
| India | S3 | Mahakaushal & Co Co | Villa No 248, Adarsh Palm Retreat, Devarabisanahalli, Bellandur Post, Bangalore South, Bangalore, Karnataka | Nexiri | Villa No 248, Bangalore, Belandur, KA | 0 | 79 |
| India | S3 | Future New Business Private Limited | No.123, 5Th Main Road, Chamrajpet, Bangalore, Karnataka | Caloxylo | No.123, 5Th Main Road, Chamrajpet, Bangalore, Karnataka | 5 | 100 |
| US | S3 | Keystone Intelligence Worldwide Inc | 11431 Heritage Oak Court, Fairfax County, VA | Jaxyuma | 11431 Heritage Oak Ct, Reston, Virginia | 5 | 71 |
| India | S3 | Chatterjee Traders Private Limited | U-2, 3Rd Floor, Green Park Main, Delhi, South Delhi, Delhi | *** Onyxwex | U-02, 3Rd Floor, Green Park Main, Delhi, South Delhi, दिल्ली | 5 | 98 |
| India | S3 | Ramya Recruitment Private Limited | West Bengal, Burdwan, Bardhaman, Feeder Rd Durgapur Burdwanold Court G T Road Durgapur | Kelosol | Feeder Rd Durgapur Burdwanold Court G T Road Durgapur, Burdwan, Bardhaman, West Bengal | 5 | 100 |
| US | S3 | Beshears and Feltner American LLC | NY, Amherst, 259 Randwood Drive | Quoevo | 259 Randwood Drive, Amherst, NY | 5 | 100 |
| India | S2 | Ski Brothers Private Limited | Bhive Premium, No. 148, Bbmp No. 170/148, 5Th Main Road, Sector 6, Bangalore South, Bangalore, Karnataka | Caloonyx | BHIVE PREMIUM, BANGALORE SOUTH, SARJAPURA, ಕರ್ನಾಟಕ | 6 | 75 |
| India | S3 | Proximo Healthcare Private Limited | 402, Atlantis Building No.4, W E Highway Mahajan Wadi, Near Thakur Mall, Mira Roa, D (E), Mumbai, Mumbai City, Maharashtra | PH | 402, Atlantis Building No.4, W E Highway Mahajan Wadi, Near Thakur Mall, Mira Roa, D (E), Mumbai, Mumbai City, Maharashtra | 6 | 100 |
| India | S3 | Swastik Healthcare Private Limited | 0, Dubey Compound, Mainpuri, Uttar Pradesh | SH | 0, Dubey Compound, Mainpuri, Uttar Pradesh | 6 | 100 |
| US | S2 | Dallas Family Association PC | 1006 Canty Street, Dallas, TX | Zephnex | 1006 CANTY ST, null, DALLAS, TX | 6 | 93 |
| India | S3 | Dynamic Agro Private Limited | Gat No. 537/1B/1, Guriha, Behind Akashwani Quarters, Miraj, Sangli, Maharashtra | Wexflux | Gat No. 537/1B/1, Guriha, Behind Akashwani Quarters, Miraj, Sangli, Maharashtra | 6 | 100 |
| US | S3 | Pumphrey Development Ltd | 138 Elm Street, Unit Unit 2, East Rochester, NY | Ariajaxavi | 138 Elm Street, Unit Unit 2, East Rochester, NY | 6 | 100 |
| India | S3 | Sunrise Producer Private Limited | Cc-48/1648 Mamangalam, Cochin, Idukki, Kerala | SP | Cc-48/1648 Mamangalam, Cochin, Idukki, Kerala | 6 | 100 |
| US | S3 | Twisted Bike Shop Clinic | 584 San Marcos Trail, Dayton, TX | Yumaumbra | #584 San Marcos Trl, Texas, Dayton | 6 | 85 |
| India | S3 | Fortune Complex Private Limited | Plot No 47, 48, 49, Street No 1, 2Nd Avenue, Patrika Nagar, Sy No 77/P, Madhapur, Hyderabad, Telangana | FC | Madhapur, Telangana, Plot No 47, 48, 49, Street No 1, 2Nd Avenue, Patrika Nagar, Sy No 77/P, Hyderabad | 6 | 100 |
| US | S3 | Legacy Electronics Holdings | 4421 Del Rio Road, Unit 2, Sacramento, CA | Jaxkor | 4421 Del Rio Road, Unit 2, Sacramento, California | 6 | 96 |
| US | S2 | Griffith & Wilson Israel LP | 601 Greenwood Avenue, Akron, OH | Dovajax | 601 1/2 GREENWOOD AVENUE, AKRON, OH | 6 | 100 |
| India | S3 | Shiva Projects Private Limited | S No. 38/1/2/1 Shinde Wasti, Mulshi, Pune Marunji, Pune, Maharashtra | SP | S No. 38/1/2/1 Shinde Wasti, Mulshi, Pune Marunji, Pune, Maharashtra | 6 | 100 |
| India | S3 | Red Foundation Private Limited | C/O Skilman Technology Private Limited, Skill Man Road, Gt Road, Phagwara, Kapurthala, Punjab | RF | C/O Skilman Technology Private Limited, Skill Man Road, Gt Road, Phagwara, Kapurthala, Punjab | 6 | 100 |
| India | S3 | East Business Private Limited | Flat No-A/51, Vijay Shree Jagat Apartment Pani Tanki Road, Kankarbagh, Patna, Bihar | EB | Flat No-A/51, Vijay Shree Jagat Apartment Pani Tanki Road, Kankarbagh, Patna, Bihar | 6 | 100 |
| India | S3 | Sunita Pharma Private Limited | A-201-202, West Gate Nr. Ymca Club, S. G. Highway, Ahmedabad, Gujarat | SP | A-201-202, Ahmedabad, Gujarat, West Gate Nr. Ymca Club, S. G. Highway | 6 | 100 |
| India | S3 | Zeal Brothers Private Limited | 601, Bhakti Park, Rhb Road, Mulund (West), Mumbai, Mumbai City, Maharashtra | ZB | 601, Bhakti Park, Rhb Road, Mulund (West), Mumbai, Mumbai City, Maharashtra | 6 | 100 |


## Most confusable hard negatives (highest name similarity)

| country | src | business_name | business_address | business_name_o | business_address_o | n_tset | a_tset |
|---|---|---|---|---|---|---|---|
| US | S2 | Zeque | 109 5th Avenue, Biwabik, MN | Zeque LLC | 226 FLAGSTONE DRIVE, EVANSVILLE CITY, IN | 100 | 35 |
| US | S2 | Zeque | IN, 1079 975, Richland | Zeque LLC | 226 FLAGSTONE DRIVE, EVANSVILLE CITY, IN | 100 | 34 |
| US | S2 | Bia | 14 Paxton Street, Rock Hill, SC | Bia Horse-Ltd Star | 001419 SKYWATER ROAD, GRAVOIS MILLS, MO | 100 | 45 |
| US | S2 | Tiva | 1276 Broadway Avenue, Burns, OR | TIVA CENTRAL LLC | 107 BELAIR STREET, BROCKTON, MA | 100 | 41 |
| US | S2 | Tique | 86 Ricci Road, Marbletown, NY | The Tique Permian |  | 100 | nan |
| US | S2 | Riliex Inc | 1735 Greentree Boulevard, Clarksville, IN | Ríliex | 2518 Plymouth Rd, JOHNSON CITY, TN | 100 | 28 |
| US | S2 | Vavue Inc | NY, Brooklyn, 807 Manhattan Avenue | Inc. Vavue | 324 Worth St, CLEVELAND, TN | 100 | 35 |
| US | S2 | Veify | 1129 Stage Coach Road, Prescott, AZ | VEIFY PARTNERS [LP] | 2859 CLEVELAND BLVD, LORAIN, OH | 100 | 35 |
| US | S2 | Canet Bancshares LLC | 24253 Ridge Road, Beavercreek, OR | Canet LLC | ##132 RAPTOR BEAK WAY, CEDAR CREEK, TX | 100 | 55 |
| US | S2 | Mark Veterinary Clinic LLC | 88 Masonic Home Road, Unit H203, Charlton, MA | Veterinary Mark Clinic | 1100 SOUTH GABRIEL DRIVE, LEANDER, TX | 100 | 31 |
| US | S2 | Cevue | 9 Elmwood Lane, Asheville, NC | Cevue Holdings East | 2401 LITTLE RACER DRIVE, RALEIGH, NC | 100 | 43 |
| US | S2 | Tobye Suarez Ventures | 9520 Glacier Highway, City And Borough Of Juneau, AK | Tobye Suarez Ventures LP | 88 N3328, CONCHO, AZ | 100 | 24 |
| US | S2 | Maum | 708 Bagley Road, Middlesex, NY | Maum Inc [Cargo] | 705 RIM RD, LOS ALAMOS, NM | 100 | 50 |
| US | S2 | Tois P.C. | 5 Mission Street, Unit Apartment 2, Glens Falls, NY | Tois |  | 100 | nan |
| US | S2 | Buix | 28742 Iris Road, Ossian, IA | Buix  Bankshares LLC | CONWAY, 16 GOLD LAKE CLUB RD, AR | 100 | 36 |
| US | S2 | Buix LLC | TX, 1016 Llano Way, Poolville | Buix  Bankshares LLC | CONWAY, 16 GOLD LAKE CLUB RD, AR | 100 | 39 |
| US | S2 | Enis Inc. | AZ, Unit 82, 9250 75th Avenue, Peoria | Enis | 3100 WHTIE CEDAR DRIVE, OKLAHOMA CITY, OK | 100 | 30 |
| US | S2 | Gedyne Tech LLC | 58 Main Street, Adams, NY | GEDYNE TECH | 28 CIRCLE LOOP, STONY POINT, NC | 100 | 35 |
| US | S2 | Zeque | 109 5th Avenue, Biwabik, MN | Zeque-Incorporated |  | 100 | nan |
| US | S2 | Zeque | IN, 1079 975, Richland | Zeque-Incorporated |  | 100 | nan |
| US | S2 | Valaria's Manufacturing Group | 500 Agmart Drive, Luverne, AL | Valaria's Manufacturing Group Corp | 509  AGMART DR, LUVERNE, AL | 100 | 90 |
| US | S2 | Neix | 1662 Harlan Avenue, Fl 0, Wichita, KS | Neix Valley Sciences Inc | 001702 BROOKSHIRE DR, SIERRA VISTA, AZ | 100 | 40 |
| US | S2 | Gowin Generation LLC | 449 Corte Majorca, Vacaville, CA | Gowin Generation South LLC | #451 CORTE MAJORCA, VACAVILLE CDP, CA | 100 | 93 |
| US | S2 | Lio | 1139 Autumn Leaf Lane, Taylorsville City, UT | Lio Hardbor Imaging (Inc) | 634 LITTRELL CIR, MOULTON, AL | 100 | 43 |
| US | S2 | Golson Veterinary Clinic LP | 1530 Windrose Bend, Angleton, TX | Golson Veterinary Clinic East Lp | 1535 WINDROSE BEND, ANGLEETON, TX | 100 | 95 |
| India | S2 | Ketan & Co | D 7/7371, Vasant Kunj, New Delhi, South Delhi, Delhi | Sri Ketan & Co  Enterprises | D 7/7373, VASANT KUNJ, NEW DELHI, दिल्ली | 100 | 82 |
| US | S2 | Thoresen Supreme Can Inc | 198 Belgrade Avenue, Boston, MA | Thoresen Supreme Can | 16601 75TH AVE, PEORIA, AZ | 100 | 38 |
| US | S2 | Gea Advisors | 3216 St Luke Road, Shenandoah County, VA | GEA | 21102 APPALOOSA DRIVE, OMAHA, NE | 100 | 38 |
| US | S2 | Mcconnell Interstate Markets Inc | 1838 Sterling Oaks Drive, St Peters, MO | The Mcconnell Interstate Markets Inc. | KS, 106 LAKESHORE DRIVE, BRURTON | 100 | 48 |
| India | S2 | Vee Sangh Ltd | Bangalore, No.50, 4Th Main Road Saroja Chambers, Nehrunagar, Bangalore, Karnataka, 4Th Floor | Vee  Sangh | Madhya Pradesh, WARD##34 BEHIND NEW HOUSING, BORD COLONY MARUTI NAGAR, DATIA | 100 | 49 |


## LightGBM baseline on these features

OOF AUC all = 0.9991; pos vs hard = 0.9984

| feature | gain_share |
|---|---|
| a_tset | 0.725 |
| a_num_jacc | 0.064 |
| a_ratio | 0.047 |
| a_partial | 0.040 |
| core_compact_ratio | 0.023 |
| n_tsort | 0.017 |
| a_first_num_eq | 0.013 |
| core_tfidf_cos | 0.013 |
| len_diff | 0.011 |
| n_jw | 0.009 |
| n_partial | 0.008 |
| n_tset | 0.006 |
| n_ratio | 0.006 |
| n_ratio_raw | 0.006 |
| core_tset | 0.006 |
| core_compact_partial | 0.003 |
| is_s3 | 0.001 |
| a_other_empty | 0.000 |

Macro F0.5 over the sampled S1 entities, treating pos+hard+random as each entity's candidate set. **Optimistic**: assumes blocking finds every true match and the only confusers are these hard negatives.

| threshold | macro_f05 | India | US |
|---|---|---|---|
| 0.1000 | 0.9598 | 0.4185 | 0.5974 |
| 0.1500 | 0.9653 | 0.4206 | 0.6007 |
| 0.2000 | 0.9689 | 0.4220 | 0.6030 |
| 0.2500 | 0.9715 | 0.4231 | 0.6044 |
| 0.3000 | 0.9735 | 0.4240 | 0.6055 |
| 0.3500 | 0.9753 | 0.4249 | 0.6065 |
| 0.4000 | 0.9770 | 0.4256 | 0.6074 |
| 0.4500 | 0.9780 | 0.4261 | 0.6079 |
| 0.5000 | 0.9790 | 0.4265 | 0.6085 |
| 0.5500 | 0.9799 | 0.4270 | 0.6089 |
| 0.6000 | 0.9809 | 0.4275 | 0.6095 |
| 0.6500 | 0.9815 | 0.4278 | 0.6098 |
| 0.7000 | 0.9818 | 0.4280 | 0.6099 |
| 0.7500 | 0.9819 | 0.4282 | 0.6098 |
| 0.8000 | 0.9812 | 0.4282 | 0.6091 |
| 0.8500 | 0.9798 | 0.4280 | 0.6079 |
| 0.9000 | 0.9764 | 0.4270 | 0.6054 |
| 0.9500 | 0.9666 | 0.4236 | 0.5991 |


## Singletons: how confident is the best (wrong) candidate?

| country | singleton | n | best_neg_p50 | best_neg_p90 | best_pos_p10 | best_pos_p50 |
|---|---|---|---|---|---|---|
| India | False | 15169 | 0.000 | 0.278 | 1.000 | 1.000 |
| India | True | 912 | 0.000 | 0.410 | None | None |
| US | False | 22589 | 0.000 | 0.229 | 1.000 | 1.000 |
| US | True | 1330 | 0.000 | 0.380 | None | None |
