# Diagnosis of run `v1b`

## A. Candidate recall (blocking) by country × source × noise type

| country | src | pairs | recall |
|---|---|---|---|
| India | S2 | 1480545 | 0.9720 |
| India | S3 | 1579298 | 0.9715 |
| US | S2 | 2213074 | 0.9865 |
| US | S3 | 2365448 | 0.9889 |

| country | type | pairs | share | recall | misses | miss_share |
|---|---|---|---|---|---|---|
| India | empty_addr | 119289 | 0.0390 | 0.6153 | 45891 | 0.5304 |
| India | native_script | 551240 | 0.1802 | 0.9594 | 22373 | 0.2586 |
| India | name_unrelated | 103794 | 0.0339 | 0.9182 | 8492 | 0.0982 |
| India | house_no_diff | 487183 | 0.1592 | 0.9912 | 4286 | 0.0495 |
| India | name_typo | 332407 | 0.1086 | 0.9917 | 2775 | 0.0321 |
| India | clean | 1465930 | 0.4791 | 0.9982 | 2702 | 0.0312 |
| US | empty_addr | 216931 | 0.0474 | 0.8072 | 41819 | 0.7447 |
| US | name_unrelated | 124354 | 0.0272 | 0.8922 | 13401 | 0.2386 |
| US | name_typo | 396118 | 0.0865 | 0.9988 | 473 | 0.0084 |
| US | house_no_diff | 483835 | 0.1057 | 0.9994 | 297 | 0.0053 |
| US | clean | 3357284 | 0.7333 | 1.0000 | 164 | 0.0029 |

**Examples of missed pairs**

| country | src | type | name_a | addr_a | name_b | addr_b |
|---|---|---|---|---|---|---|
| India | S2 | empty_addr | Hari Vidyalaya | E-146, K K Colony Bikaner, Bikaner, Rajasthan | Sri Hari Vidyalaya Corp |  |
| US | S2 | name_unrelated | Anderea Richards, D.O., P.C. | 87 Front Street, Groveport, OH | Lyravantagedova | 87- FRONT STREET, GROVEPORT, OH |
| India | S2 | house_no_diff | Great Constructions Private Limited | 293, Flat 319, C Block, 3Rd Floor, Mbr Steeple Apts Begur Devarachikkanahalli Main Road, Beg, Ur, Bangalore, Karnataka | Límited Great Private Services | HN 47 293, BENGALURU, BANGALORE, Karnataka |
| US | S2 | empty_addr | Arts Association LLC | 21304 Cherry Hollow Court, Leander, TX | Arts Assfborialmion LLC |  |
| US | S2 | name_unrelated | Raven LLC | 7305 Perkins Road, Stillwater, OK | Drexvio | PERKINS ROAD, STILLWATER CITY, OK |
| US | S2 | empty_addr | Grand Technical Group | 325 10th Street, Unit Space 3, Cottonwood, AZ | GRAND TECHNICAL |  |
| India | S3 | house_no_diff | Mumbai Manpower Pvt Ltd | Mumbai, 41-P, Building No. Y-3, A-Wing, Andheri, Near Apna Ghar, 4 Bunglow, Mumbai City, 702, Maharashtra | Mumbai Pvt Ltd [Services] | Hn 57 702, Mumbai City, MH |
| India | S3 | empty_addr | Dream Infra Private Limited | 7-1-27/6, Behind Green Park Hotel, Green Lands, Am, Eerpet, Hyderabad, Telangana | Dream Inra Private Limited |  |
| US | S2 | empty_addr | Oncology Group | 253 Crestview Acres Road, Statesville, NC | ONCOLOGY GROUP Enterprises |  |
| India | S2 | empty_addr | Shreeji Knowledge Private Limited | Lg60 Lg62 Lg89 Lg90 Lg91, Lg, Artha Mart Plot No 21, Bishrakh, Gautam Buddha Nagar, Uttar Pradesh | Shreeji Private Limited Center |  |
| US | S2 | name_unrelated | Novex Spade Inc | 250 Harlan Road, Atlanta, GA | Zephorbi | GA, ATL, 250 HARLAN RD |
| India | S2 | empty_addr | Shree Centre Private Limited | D 45 Industrial Estate, Aligarh, Uttar Pradesh | SHREE CENTRE LÍMITED PARTNERS |  |
| India | S2 | empty_addr | Delhi Techno Private Limited | North West, Delhi, 3Rd Floor Blk-C, Pkt-1 Sector 16, Rohini, Delhi, Plot No 17-18 | Delhi Private Techno Limited |  |
| US | S2 | empty_addr | Clifford Fidelity | Katy, TX, 24511 Blane Drive | CLIFFORD FÍDEIIIYT |  |
| India | S2 | empty_addr | True & Sons Pvt Ltd | Andhra Pradesh, Gutur, Jai Krishna Bhavan, 1St Floor, 16/3, Arundelpet, D. No. 6-16-5, Guntur | True & Sbos Pvt Ltd |  |
| US | S2 | name_unrelated | Karon Harris Lease P.C. | 126 Vance Po Road, Unit 116, Statesville, NC | Umbracira | NC, VANCE PO ROAD, STATESVILLE |
| India | S3 | empty_addr | Memorial Bombay Private Limited | 687/13, (New No.556/13), Poonamallee High Road, Aminjikarai, Chennai, Tamil Nadu | Memorial Bombay Limited Center |  |
| US | S2 | empty_addr | Global Tankers Inc | 1517 Elizabeth Street, Albuquerque, NM | GLOBAL TIMHERS INC |  |
| US | S2 | empty_addr | Eye Partners | 51 Grace Lane, New Castle, NY | Eye Pátners |  |
| India | S2 | empty_addr | New Delhi Enterprise Pvt Ltd | 3/303, Ashish Complex New Rajdhani Enclave, Vikas Marg, New Delhi, East Delhi, Delhi | New Délhi Enterprise Pvt Ltd |  |
| India | S2 | name_unrelated | Oakwood Solution Private Limited | F-63, Riico Industrial Area Mandore, Rajasthan, Jodhpur | Synavi | F-A-63, JODHPUR, Rajasthan |
| India | S2 | native_script | Indian Hospitality Private Limited | Plot No 182, Mukundas Lohia Academic, Complex Behind Bmcc, Shivajinagar, Pune City, Pune, Maharashtra | इंडियन हॉस्पिटैलिटी प्राइवेट लिमिटेड | Maharashtra, PUNE REGION, PUNE, H.NO 182 |
| US | S2 | empty_addr | Behavioral Health Prairie Specialists Inc. | 717 Laurel Street, Corning, AR | BEHAVIORAL HEALTH HEALTH PRAIRIE SPECIALISTS [INCORPORATED] |  |
| India | S2 | native_script | Dream International Private Limited | 1, Moti Villa, Azad Road, Near Milan Sub Way, Vile Parle (East), Mumbai, Mumbai City, Maharashtra | ड्रीम इंटरनेशनल प्राइवेट लिमिटेड | MUMBAI, MUMBAI CITY, 1, महाराष्ट्र |
| US | S2 | empty_addr | Peridari King Inc | 11996 Richmond Road, Richmond County, VA | Peridari Inc Center |  |

## B. Model errors with the tuned rule `{"kind": "thresh", "t": 0.65, "floor": 0.0, "one_owner": true, "macro_f05": 0.9781272183982542}`

| country | macro_f05 | entities |
|---|---|---|
| US | 0.9795 | 1323633 |
| India | 0.9760 | 883188 |

### Recall by country × source, and why true pairs are lost

| country | src | pairs | recall |
|---|---|---|---|
| India | S2 | 1480545 | 0.9507 |
| India | S3 | 1579298 | 0.9454 |
| US | S2 | 2213074 | 0.9543 |
| US | S3 | 2365448 | 0.9598 |

| country | type | cause | len |
|---|---|---|---|
| India | clean | found | 1457794 |
| India | native_script | found | 525596 |
| India | house_no_diff | found | 460643 |
| India | name_typo | found | 324807 |
| India | name_unrelated | found | 85744 |
| India | empty_addr | found | 46042 |
| India | empty_addr | missed_by_blocking | 45891 |
| India | empty_addr | below_threshold | 25704 |
| India | native_script | missed_by_blocking | 22373 |
| India | house_no_diff | below_threshold | 22176 |
| India | name_unrelated | below_threshold | 9159 |
| India | name_unrelated | missed_by_blocking | 8492 |
| India | clean | below_threshold | 5321 |
| India | name_typo | below_threshold | 4727 |
| India | house_no_diff | missed_by_blocking | 4286 |
| India | native_script | below_threshold | 3167 |
| India | name_typo | missed_by_blocking | 2775 |
| India | clean | missed_by_blocking | 2702 |
| India | empty_addr | lost_to_one_owner_or_set_rule | 1652 |
| India | name_unrelated | lost_to_one_owner_or_set_rule | 399 |
| India | clean | lost_to_one_owner_or_set_rule | 113 |
| India | native_script | lost_to_one_owner_or_set_rule | 104 |
| India | name_typo | lost_to_one_owner_or_set_rule | 98 |
| India | house_no_diff | lost_to_one_owner_or_set_rule | 78 |
| US | clean | found | 3353729 |
| US | house_no_diff | found | 439403 |
| US | name_typo | found | 393085 |
| US | name_unrelated | found | 98626 |
| US | empty_addr | found | 97314 |
| US | empty_addr | below_threshold | 76341 |
| US | house_no_diff | below_threshold | 44129 |
| US | empty_addr | missed_by_blocking | 41819 |
| US | name_unrelated | missed_by_blocking | 13401 |
| US | name_unrelated | below_threshold | 11416 |
| US | clean | below_threshold | 3343 |
| US | name_typo | below_threshold | 2520 |
| US | empty_addr | lost_to_one_owner_or_set_rule | 1457 |
| US | name_unrelated | lost_to_one_owner_or_set_rule | 911 |
| US | name_typo | missed_by_blocking | 473 |
| US | house_no_diff | missed_by_blocking | 297 |
| US | clean | missed_by_blocking | 164 |
| US | clean | lost_to_one_owner_or_set_rule | 48 |
| US | name_typo | lost_to_one_owner_or_set_rule | 40 |
| US | house_no_diff | lost_to_one_owner_or_set_rule | 6 |

**Examples of true pairs predicted as non-match (candidates, below threshold)**

| country | src | type | name_a | addr_a | name_b | addr_b |
|---|---|---|---|---|---|---|
| US | S2 | empty_addr | Velez's Hypnosis | 45961 South Drive, Big Bear City, CA | VELEZ'S HYPNOSI |  |
| US | S2 | house_no_diff | Khan Grid Inc | 9600 Wightman Road, Gaithersburg, MD | Khan Groid Inc | 960 WIGHTMAN RD, GAITHERSBRG, MD |
| US | S2 | empty_addr | Regional Management LLC | 9303 Landave Cove, Unit Unit 2, Jacksonville, AR | Regional Management ((LLC)) |  |
| US | S2 | empty_addr | Business Biomedical Networks LLC | 222 Olive Street, Cashmere, WA | BUSINESS BIOMEDICAL NETWORKS |  |
| US | S3 | clean | Smith, Clarey R., P.A., DDS PC | 1308 Seventh Street, Frederick, MD | Smith, Clarey R., R., P.A., DDS | 1308a Seventh Saint, Frederick, Maryland |
| US | S2 | house_no_diff | Crystal Intelligence Ventures Care | 2880 Sunrise Boulevard, Unit STE 121, Rancho Cordova, CA | Ventures Intelligence Crystal Care-Inc | 288 SUNRISE BLVD, RANCHO CORDOVA CITY, CA |
| India | S3 | name_unrelated | CD (india) Private Limited | Apartment No.U-104, Maharashtra, Nagpur, Laxmi Vihar Complex, Plot No.B-2 Near Airport Centre Point, Wardha Road, So, Malwada, Nagpur | Korlumzeta | #660 Apartment No.u-104, Nagpur, महाराष्ट्र |
| India | S3 | house_no_diff | Silicon Foundation | A-2207, Hillcrest Apt, Jvlr, Opp Seepz Gate No. 3, Andheri East, Mumbai, Maharashtra | Smt Silicon Service | A, महाराष्ट्र, Hillcrest Apt, Jvlr, Opp Seepz Gate No. 3, Andheri East, Mumbai |
| US | S2 | name_unrelated | Pate Metro Paper LLC | 1726 Courage Creek, Crandall, TX | ONYXSYN | COURAGE CREEK, CRANDALL, TX |
| US | S2 | empty_addr | The Dent Hypnosis | MN, 18330 Krypton Street, Nowthen | The  Dent Hypnosis |  |
| US | S2 | empty_addr | Harrison Cdt | 9240 Cypress Waters Boulevard, Dallas, TX | Harrison Cdt |  |
| US | S2 | empty_addr | Summit Integrated Bancshares Group | 311 Edmonds Street, Edmonds, WA | Summit Integrated Bancshares |  |
| US | S2 | name_unrelated | Metropolitan Atlanticus | OH, 7524 Dayton-springfield Road, Fairborn | *** Círabrixaria | DAYTON-SPRINGFIELD RD, FAIRBORN, OH |
| India | S2 | house_no_diff | Suri (India) Infratel Private Limited | A-4702, Floor-47Th, Plot-Cs 77, 3, One Avighana Park Tower, Mahadeo Palav Marg, Curry Road, Mumbai, Maharashtra | Private Suri (India) Iftancntel Limited | 305 A-4702, BOMBAY, MUMBAI, Maharashtra |
| India | S2 | empty_addr | Vijay Ventures Private Limited | Kerala, Kunnumpurath House Thaikkattukara, Ernakulam | Vijay Ventures Prívate Limited |  |
| US | S2 | name_unrelated | Certified Energy Global | 105 Beechdale Road, Portsmouth City, VA | globalenergycertified.com | 93 BEECHDALE ROAD, PORTSMOUTH, VA |
| US | S3 | empty_addr | Prime Freight Partners LLC | NM, Questa, 55 Camino Refugio | Prime Freight |  |
| US | S2 | empty_addr | Pediatric Prime Care | 1930 Ragu Drive, Unit 6, Owensboro, KY | Pediatric Prime Care Inc |  |
| US | S2 | empty_addr | Teify Capital | 760 Roundhouse Court, Sisters, OR | Teify  Capital |  |
| US | S2 | empty_addr | Value Aerospace Ventures | 1615 Macarthur Street, Village Of Allouez, WI | VALUE AEROSPACE-VCENTUERS AEROSPACE-VCENTUERS |  |

### Precision by country × source

| country | src | predicted | precision |
|---|---|---|---|
| India | S2 | 1418074 | 0.9926 |
| India | S3 | 1506407 | 0.9911 |
| US | S2 | 2129070 | 0.9919 |
| US | S3 | 2287608 | 0.9924 |

### False-positive patterns

| country | pattern | len |
|---|---|---|
| India | other | 7804 |
| India | same_name_diff_addr | 5960 |
| India | near_twin_number | 5264 |
| India | same_addr_diff_name | 3309 |
| India | singleton_false_merge | 1518 |
| US | same_name_diff_addr | 13277 |
| US | near_twin_number | 8920 |
| US | other | 6016 |
| US | same_addr_diff_name | 3763 |
| US | singleton_false_merge | 2545 |

**Examples of false positives**

| country | src | type | name_a | addr_a | name_b | addr_b |
|---|---|---|---|---|---|---|
| US | S3 | name_unrelated | FR Health Inc | 153 Chestnut Street, Unit A2, Foxborough, MA | The Watkins Health Inc | 153 Chestntu St, # A2, Foxborough, Massachusetts |
| US | S2 | house_no_diff | Youth Services Group | 8810 Pioneer Street, Phoenix, AZ | Youth  Services Group | 8823 PIONEER ST, TOLLESON, AZ |
| US | S3 | name_unrelated | 5-Star | 229 Coastal Drive, Washington, NC | uptowndeli.com | 229 Coastal Dr, Washington, North Carolina |
| India | S3 | name_typo | AVX Trading Private Limited | Mumbai City, Maharashtra, Mumbai, Laxmi Plaza, Laxmi Industrial Estate, New Link Road, Andheri (West), 101 | Laxmi Trading Private Limited | MH, Mumbai, Mumbai City, 101 |
| US | S3 | empty_addr | Peak Trusted Tax Service | 555 Orleans Road, Orleans, MA | Peak Service Tax Trusted |  |
| India | S3 | house_no_diff | Universal Products Pvt Ltd | Shop - 604, Mega Center, Building F, Marapatta City, Hadapsar, Pune, Maharashtra | Universal Producer Pvt Ltd. | Shop - 605, Mega Center, Building F, Marapatta City, Hadapsar, Pune Region, Poona, MH |
| US | S3 | empty_addr | Cameron Trust | 3417 Williams Drive, Phoenix, AZ | Cameron  Trust |  |
| India | S3 | name_unrelated | Services Pallas Analytical LLP | Flat No. A- 704, Plot No 26, Jay Balaji Chs Ltd, Sector 6, Nerul West Navi Mumbai, Thane, Maharashtra | Xylovera | Flat No. A- 704, Plot No 26, Jay Balaji Chs Ltd, Sector 6, Nerul West Navi Mumbai, Thane, Maharashtra |
| US | S3 | house_no_diff | Premier Research Dynamics Inc. | 13405 Malloy Lane, Athens, OH | Premier Research Dynamics Inc | 13409-A Malloy Ln, Athens, Ohio |
| US | S3 | empty_addr | Kinney Premier Catalyst Inc | 20198 Cameo Road, Apple Valley, CA | Kinney Premier Catalyst Ínc |  |
| India | S2 | empty_addr | Tms Group of Companies | 8-B, Dover Road, Kolkata, Kolkata, Howrah, West Bengal | TMS GROUP-OF COMPANIES |  |
| India | S3 | name_unrelated | Ashvik Spacez | Door No4/461, 2Nd Floor, Valamkottil Towers, Kanayannur, Ernakulam, Kerala | Onyxhalo | Door No 4/461, 2Nd Floor, Valamkottil Towers, Kanayannur, Ernakulam, Kerala |
| India | S2 | clean | QTA Marketing Pvt. Ltd. | Ground Floor, Amba Bhuvan, Sion Circle 29, Sion, Mumbai, Maharashtra | QGA Marketing Pvt. Ltd. | GROUND FLOOR, AMBA BHUVAN, SION CIRCLE 29., SION, Maharashtra |
| India | S2 | name_typo | Silver Global Private Limited | 11-13-600/A/A2, Plot No.55, Road No. 3, 2Nd Floor, E Adhar Ambiance, Haripuri Colony, Saroor Nagar, Hyderabad, Telangana | Silver Global Bakery Private | 11-13-600/A/A13., PLOT NO.55, ROAD NO. 3, 2ND FLOOR, E ADHAR AMBIANCE, HARIPURI COLONY, SAROOR NAGAR, Telangana |
| US | S2 | house_no_diff | Supreme Printing Industries Company | Deer Park, WA, 1105 Crawford Street | Supreme Printing Industries Co | 01110 CRAWFORD STREET, DEER PARK, WA |
| US | S3 | name_unrelated | Bailey Net-Town Of Clyman | N4445 M, Town Of Clyman, WI | Lyrial [Opcde] | ##N4445 M, Town Of Clyman, Wisconsin |
| US | S2 | house_no_diff | Nanet Assets | 7514 Northlake Drive, Tyler, TX | Nanet Assets Corp | 751 NORTHLAKE DR, TYLER, TX |
| India | S3 | name_unrelated | SE Arogya | 238 B A J C Bose Road Ps Bhawanipur, Kolkata, Calcutta, West Bengal | Halofaye | Calcutta, WB, 238 B A J C Bose Road Ps Bhawanipur, Calcutta |
| India | S3 | name_typo | Krishna Food Private Limited | 129/1, Salkia School Road Opposite Hooghly Dock, Howrah, West Bengal | Sri Krishna Engineering Private Limited | WB, 129, Salkia School Road Opposite Hooghly Dock, Howrah |
| US | S2 | clean | Interstate Initiative IV | 333 Ethel Drive, Jessamine Co, KY | Interstate Initiative UV | 333 ETHEL DRIVE, NICHOLASVLLE, KY |