# C45-VAL

jobs: 144; records with an error: 0


| device | X1 C45 on failed | X2 C45 repeats | X3 ESP C45/QK3; below 0.98 x QK3 | X4 ESP C45/C44; below C44 | X5 C45n = C44 by value | X6 median time C45/C44 | X7 level 3 kept / tried (unverified) | C45/R4 | C45n/QK3 |
|---|---|---|---|---|---|---|---|---|---|
| FakeTorino | 0 | 24 / 24 | 1.229 (13); 1 of 13 | 1.021; 0 | 24 / 24 | 1.33 | 6 / 24 (6) | 1.021 | 1.204 |
| FakeKingston | 0 | 24 / 24 | 1.117 (17); 2 of 17 | 1.019; 0 | 24 / 24 | 1.35 | 14 / 24 (14) | 1.035 | 1.095 |
| FakeMarrakesh | 0 | 24 / 24 | 1.511 (16); 0 of 17 | 1.013; 0 | 24 / 24 | 1.38 | 9 / 24 (9) | 1.013 | 1.491 |
| FakeFez | 0 | 24 / 24 | 1.163 (14); 1 of 14 | 1.022; 0 | 24 / 24 | 1.36 | 10 / 24 (10) | 1.043 | 1.138 |
| FakeAachen | 0 | 24 / 24 | 1.100 (19); 0 of 19 | 1.034; 0 | 24 / 24 | 1.47 | 10 / 24 (10) | 1.061 | 1.064 |
| FakeBoston | 0 | 24 / 24 | 1.370 (20); 0 of 20 | 1.009; 0 | 24 / 24 | 1.43 | 14 / 24 (14) | 1.032 | 1.358 |
| all | 0 | 144 / 144 | 1.241 (99); 4 of 100 | 1.019; 0 | 144 / 144 | 1.36 | 63 / 144 (63) | 1.035 | 1.217 |

## By family (runnable; ESP C45/QK3, C45/C44, C45n/QK3)

- bv: 1.128, 1.024, 1.101 (12)
- ghz: 0.987, 1.039, 0.946 (11)
- qaoa: 1.653, 1.006, 1.644 (17)
- qft: 1.250, 1.008, 1.240 (18)
- qv: 1.015, 1.025, 0.991 (16)
- random: 1.322, 1.022, 1.293 (26)

## Per test (log10 ESP C45 - QK3, C45 - C44, C45n - QK3; level 3 kept; C45 repeats)

- gen_bv_n50 Aachen: +0.0134, +0.0000, +0.0134; 0; True
- gen_bv_n50 Boston: +0.0152, +0.0000, +0.0152; 0; True
- gen_bv_n50 Fez: +0.1034, +0.0000, +0.1034; 0; True
- gen_bv_n50 Kingston: +0.0859, +0.0000, +0.0859; 0; True
- gen_bv_n50 Marrakesh: +0.2447, +0.0000, +0.2447; 0; True
- gen_bv_n50 Torino: +0.0073, +0.0256, -0.0183; 1; True
- gen_bv_n90 Aachen: +0.0221, +0.0505, -0.0284; 1; True
- gen_bv_n90 Boston: +0.0103, +0.0019, +0.0084; 1; True
- gen_bv_n90 Fez: +0.0653, +0.0000, +0.0653; 0; True
- gen_bv_n90 Kingston: -0.0079, +0.0479, -0.0559; 1; True
- gen_bv_n90 Marrakesh: +0.0319, +0.0000, +0.0319; 0; True
- gen_bv_n90 Torino: +0.0367, +0.0000, +0.0367; 0; True
- gen_ghz_n100 Aachen: +0.1391, +0.0229, +0.1162; 1; True
- gen_ghz_n100 Boston: +0.0051, +0.0023, +0.0028; 1; True
- gen_ghz_n100 Fez: -0.0279, +0.0000, -0.0279; 0; True
- gen_ghz_n100 Kingston: -0.1397, +0.0773, -0.2170; 1; True
- gen_ghz_n100 Marrakesh: +30.0026, +0.0000, +30.0026; 0; True
- gen_ghz_n100 Torino: +120.2358, +0.0000, +120.2358; 0; True
- gen_ghz_n40 Aachen: -0.0019, +0.0055, -0.0073; 1; True
- gen_ghz_n40 Boston: -0.0012, +0.0009, -0.0021; 1; True
- gen_ghz_n40 Fez: -0.0008, +0.0149, -0.0157; 1; True
- gen_ghz_n40 Kingston: -0.0244, +0.0000, -0.0244; 0; True
- gen_ghz_n40 Marrakesh: +0.0000, +0.0001, -0.0001; 1; True
- gen_ghz_n40 Torino: -0.0032, +0.0603, -0.0634; 1; True
- gen_random_n14 Aachen: +0.0000, +0.0014, -0.0013; 1; True
- gen_random_n14 Boston: +0.0068, +0.0000, +0.0068; 0; True
- gen_random_n14 Fez: +0.0000, +0.0059, -0.0059; 1; True
- gen_random_n14 Kingston: +0.0090, +0.0000, +0.0090; 0; True
- gen_random_n14 Marrakesh: +0.0000, +0.0231, -0.0231; 1; True
- gen_random_n14 Torino: +0.0179, +0.0000, +0.0179; 0; True
- gen_random_n26 Aachen: +0.0282, +0.0000, +0.0282; 0; True
- gen_random_n26 Boston: -0.0003, +0.0000, -0.0003; 0; True
- gen_random_n26 Fez: +0.0000, +0.0420, -0.0420; 1; True
- gen_random_n26 Kingston: +0.0272, +0.0000, +0.0272; 0; True
- gen_random_n26 Marrakesh: +0.1147, +0.0000, +0.1147; 0; True
- gen_random_n26 Torino: +0.0721, +0.0000, +0.0721; 0; True
- gen_random_n36 Aachen: +0.0489, +0.0000, +0.0489; 0; True
- gen_random_n36 Boston: +0.0081, +0.0071, +0.0010; 1; True
- gen_random_n36 Fez: +0.0122, +0.0034, +0.0088; 1; True
- gen_random_n36 Kingston: +0.0563, +0.0000, +0.0563; 0; True
- gen_random_n36 Marrakesh: +0.3566, +0.0000, +0.3566; 0; True
- gen_random_n36 Torino: +0.0093, +0.0000, +0.0093; 0; True
- gen_random_n44 Aachen: +0.0160, +0.0706, -0.0546; 1; True
- gen_random_n44 Boston: -0.0009, +0.0448, -0.0458; 1; True
- gen_random_n44 Fez: +0.0408, +0.0000, +0.0408; 0; True
- gen_random_n44 Kingston: -0.0025, +0.0075, -0.0099; 1; True
- gen_random_n44 Marrakesh: +0.1789, +0.0432, +0.1357; 1; True
- gen_random_n44 Torino: -0.1809, +0.0000, -0.1809; 0; True
- gen_random_n52 Aachen: +0.3351, +0.0000, +0.3351; 0; True
- gen_random_n52 Boston: +1.9963, +0.0000, +1.9963; 0; True
- gen_random_n52 Fez: +360.2034, +0.0000, +360.2034; 0; True
- gen_random_n52 Kingston: +0.0009, +0.0339, -0.0330; 1; True
- gen_random_n52 Marrakesh: +360.7711, +0.0000, +360.7711; 0; True
- gen_random_n52 Torino: +120.8375, +0.0000, +120.8375; 0; True
- other_clifford_20_12345 Aachen: -0.0051, +0.0752, -0.0802; 1; True
- other_clifford_20_12345 Boston: +0.0016, +0.0828, -0.0812; 1; True
- other_clifford_20_12345 Fez: +0.3936, +0.0000, +0.3936; 0; True
- other_clifford_20_12345 Kingston: -0.0033, +0.0048, -0.0081; 1; True
- other_clifford_20_12345 Marrakesh: +2.3181, +0.0000, +2.3181; 0; True
- other_clifford_20_12345 Torino: +0.0049, +0.2696, -0.2647; 1; True
- qaoa_N16 Aachen: +0.0279, +0.0000, +0.0279; 0; True
- qaoa_N16 Boston: +0.0557, +0.0000, +0.0557; 1; True
- qaoa_N16 Fez: +0.0803, +0.0000, +0.0803; 1; True
- qaoa_N16 Kingston: +0.0928, +0.0000, +0.0928; 1; True
- qaoa_N16 Marrakesh: +0.1241, +0.0000, +0.1241; 1; True
- qaoa_N16 Torino: +0.2867, +0.0000, +0.2867; 0; True
- qaoa_N23 Aachen: +0.0241, +0.0000, +0.0241; 0; True
- qaoa_N23 Boston: +0.0786, +0.0000, +0.0786; 1; True
- qaoa_N23 Fez: +0.5017, +0.0000, +0.5017; 1; True
- qaoa_N23 Kingston: +0.1700, +0.0000, +0.1700; 1; True
- qaoa_N23 Marrakesh: +0.3049, +0.0000, +0.3049; 0; True
- qaoa_N23 Torino: +0.5749, +0.0000, +0.5749; 0; True
- qaoa_N29 Aachen: -0.0010, +0.0199, -0.0209; 1; True
- qaoa_N29 Boston: -0.0006, +0.0208, -0.0214; 1; True
- qaoa_N29 Fez: +271.0917, +0.0000, +271.0917; 0; True
- qaoa_N29 Kingston: +0.1833, +0.0000, +0.1833; 1; True
- qaoa_N29 Marrakesh: +0.9301, +0.0000, +0.9301; 0; True
- qaoa_N29 Torino: +0.3965, +0.0000, +0.3965; 0; True
- qaoa_N42 Aachen: +0.2290, +0.0000, +0.2290; 0; True
- qaoa_N42 Boston: +0.2761, +0.0000, +0.2761; 1; True
- qaoa_N42 Fez: +407.2515, +0.0000, +407.2515; 0; True
- qaoa_N42 Kingston: +0.7789, +0.0000, +0.7789; 1; True
- qaoa_N42 Marrakesh: +497.4204, +0.0000, +497.4204; 0; True
- qaoa_N42 Torino: +0.8605, +0.0000, +0.8605; 0; True
- qaoa_N65 Aachen: +1.4208, +0.0000, +1.4208; 0; True
- qaoa_N65 Boston: +11.2494, +0.0000, +11.2494; 0; True
- qaoa_N65 Fez: +1845.9544, +0.0000, +1845.9544; 0; True
- qaoa_N65 Kingston: +1.1890, +0.0000, +1.1890; 0; True
- qaoa_N65 Marrakesh: +1911.7905, +0.0000, +1911.7905; 0; True
- qaoa_N65 Torino: +1340.7234, +0.0000, +1340.7234; 0; True
- qft_N17 Aachen: +0.0392, +0.0000, +0.0392; 0; True
- qft_N17 Boston: +0.0410, +0.0000, +0.0410; 1; True
- qft_N17 Fez: +0.1127, +0.0000, +0.1127; 0; True
- qft_N17 Kingston: +0.2560, +0.0000, +0.2560; 0; True
- qft_N17 Marrakesh: +0.2340, +0.0000, +0.2340; 0; True
- qft_N17 Torino: +0.0708, +0.0000, +0.0708; 0; True
- qft_N21 Aachen: +0.0142, +0.0000, +0.0142; 0; True
- qft_N21 Boston: +0.0815, +0.0000, +0.0815; 1; True
- qft_N21 Fez: +0.0297, +0.0000, +0.0297; 0; True
- qft_N21 Kingston: +0.0315, +0.0000, +0.0315; 0; True
- qft_N21 Marrakesh: +0.0000, +0.0091, -0.0091; 1; True
- qft_N21 Torino: +0.2563, +0.0000, +0.2563; 0; True
- qft_N25 Aachen: +0.0793, +0.0000, +0.0793; 0; True
- qft_N25 Boston: +0.0691, +0.0000, +0.0691; 0; True
- qft_N25 Fez: +0.2239, +0.0000, +0.2239; 0; True
- qft_N25 Kingston: +0.0717, +0.0000, +0.0717; 0; True
- qft_N25 Marrakesh: +0.2844, +0.0000, +0.2844; 0; True
- qft_N25 Torino: +0.0000, +0.1566, -0.1566; 1; True
- qft_N33 Aachen: +0.0002, +0.0564, -0.0562; 1; True
- qft_N33 Boston: +0.0722, +0.0000, +0.0722; 0; True
- qft_N33 Fez: +1.3445, +0.0000, +1.3445; 0; True
- qft_N33 Kingston: -0.0015, +0.0499, -0.0514; 1; True
- qft_N33 Marrakesh: +0.0000, +0.1807, -0.1807; 1; True
- qft_N33 Torino: +0.1341, +0.0000, +0.1341; 0; True
- qft_N44 Aachen: +0.2845, +0.0000, +0.2845; 0; True
- qft_N44 Boston: +0.0000, +0.3320, -0.3320; 1; True
- qft_N44 Fez: +269.8083, +0.0000, +269.8083; 0; True
- qft_N44 Kingston: +0.0048, +0.1028, -0.0980; 1; True
- qft_N44 Marrakesh: +134.5642, +0.0000, +134.5642; 0; True
- qft_N44 Torino: +92.1991, +0.0000, +92.1991; 0; True
- qv_N13 Aachen: +0.0000, +0.0063, -0.0063; 1; True
- qv_N13 Boston: +0.0030, +0.0000, +0.0030; 0; True
- qv_N13 Fez: +0.0000, +0.0086, -0.0086; 1; True
- qv_N13 Kingston: +0.0000, +0.0078, -0.0078; 1; True
- qv_N13 Marrakesh: +0.0000, +0.0157, -0.0157; 1; True
- qv_N13 Torino: +0.0000, +0.0307, -0.0307; 1; True
- qv_N15 Aachen: +0.0002, +0.0000, +0.0002; 0; True
- qv_N15 Boston: +0.0044, +0.0000, +0.0044; 0; True
- qv_N15 Fez: +0.0000, +0.0561, -0.0561; 1; True
- qv_N15 Kingston: +0.0004, +0.0014, -0.0011; 1; True
- qv_N15 Marrakesh: +0.0000, +0.0027, -0.0027; 1; True
- qv_N15 Torino: +0.0148, +0.0000, +0.0148; 0; True
- qv_N19 Aachen: +0.0000, +0.0398, -0.0398; 1; True
- qv_N19 Boston: +0.0124, +0.0000, +0.0124; 0; True
- qv_N19 Fez: +0.0000, +0.1919, -0.1919; 1; True
- qv_N19 Kingston: +0.0059, +0.0000, +0.0059; 0; True
- qv_N19 Marrakesh: +0.0633, +0.0000, +0.0633; 0; True
- qv_N19 Torino: +0.0000, +0.0665, -0.0665; 1; True
- qv_N26 Aachen: +0.0651, +0.0000, +0.0651; 0; True
- qv_N26 Boston: +0.0000, +0.1341, -0.1341; 1; True
- qv_N26 Fez: +0.0000, +0.2214, -0.2214; 1; True
- qv_N26 Kingston: +0.0000, +0.1416, -0.1416; 1; True
- qv_N26 Marrakesh: -0.0001, +0.2096, -0.2097; 1; True
- qv_N26 Torino: +0.0882, +0.0000, +0.0882; 0; True
