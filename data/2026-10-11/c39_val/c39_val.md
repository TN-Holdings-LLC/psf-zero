# C39-VAL

jobs: 168; records with an error: 0


| device | commuting | X1 C39 on failed | X2 non-commuting C39 = R1 = R2 | X3 C39 >= R1 every commuting test | X4 C39 >= R2 every commuting test | runnable commuting | X5 ESP C39/R1 | ESP R2/R1 | ESP C39/QK3 | X6 tsp-type: C39/R1, R2/R1 | X7 median time C39 / R1 s (commuting) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| FakeTorino | 25 | 0 | 3 / 3 | 25 / 25 | 25 / 25 | 23 | 1.487 (23) | 1.486 | 1.551 | 1.592, 1.592 (9) | 0.57 / 0.09 |
| FakeKingston | 25 | 0 | 3 / 3 | 25 / 25 | 25 / 25 | 25 | 1.295 (25) | 1.295 | 1.279 | 1.282, 1.282 (9) | 0.47 / 0.11 |
| FakeMarrakesh | 25 | 0 | 3 / 3 | 25 / 25 | 25 / 25 | 25 | 1.908 (25) | 1.905 | 1.677 | 1.948, 1.948 (9) | 0.50 / 0.10 |
| FakeFez | 25 | 0 | 3 / 3 | 25 / 25 | 25 / 25 | 25 | 1.647 (25) | 1.644 | 1.687 | 1.438, 1.438 (9) | 0.58 / 0.12 |
| FakeAachen | 25 | 0 | 3 / 3 | 25 / 25 | 25 / 25 | 25 | 1.238 (25) | 1.221 | 1.220 | 1.250, 1.249 (9) | 0.56 / 0.11 |
| FakeBoston | 25 | 0 | 3 / 3 | 25 / 25 | 25 / 25 | 25 | 1.215 (25) | 1.211 | 1.171 | 1.259, 1.259 (9) | 0.60 / 0.12 |
| all | 150 | 0 | 18 / 18 | 150 / 150 | 150 / 150 | 148 | 1.445 (148) | 1.439 | 1.406 | 1.442, 1.442 (54) | 0.56 / 0.11 |

## Runnable commuting tests (log10 ESP C39 - R1, R2 - R1, C39 - QK3; two-qubit C39 / R1 / R2; chosen)

- Aachen gen_densegnp_n10_s1: +0.0118, +0.0118, +0.0075; 74 / 83 / 74; steiner
- Aachen gen_densegnp_n10_s2: +0.0036, +0.0036, +0.0050; 58 / 58 / 58; steiner
- Aachen gen_densegnp_n16_s1: +0.0109, +0.0109, +0.0190; 240 / 258 / 240; steiner
- Aachen gen_densegnp_n16_s2: +0.0371, +0.0371, +0.0512; 220 / 252 / 220; steiner
- Aachen gen_densegnp_n22_s1: +0.1610, +0.1610, +0.1849; 432 / 598 / 432; steiner
- Aachen gen_densegnp_n22_s2: +0.0533, +0.0533, +0.0518; 492 / 622 / 492; steiner
- Aachen gen_maxcut3reg_n12_s1: +0.0073, +0.0073, -0.0076; 80 / 78 / 80; steiner
- Aachen gen_maxcut3reg_n12_s2: +0.0130, +0.0130, +0.0154; 66 / 83 / 66; steiner
- Aachen gen_maxcut3reg_n24_s1: +0.0076, +0.0076, -0.0001; 202 / 192 / 202; steiner
- Aachen gen_maxcut3reg_n24_s2: +0.0450, +0.0450, +0.0294; 149 / 188 / 149; steiner
- Aachen gen_maxcut3reg_n36_s1: +0.0013, +0.0013, +0.0910; 364 / 322 / 364; steiner
- Aachen gen_maxcut3reg_n36_s2: +0.0000, -0.1487, +0.0415; 332 / 332 / 581; usual
- Aachen gen_sat3_n14_s1: +0.2266, +0.2266, +0.2119; 415 / 638 / 415; steiner
- Aachen gen_sat3_n14_s2: +0.1417, +0.1417, +0.1091; 448 / 583 / 448; steiner
- Aachen gen_sat3_n24_s1: +0.3721, +0.3721, +0.4869; 1238 / 1639 / 1238; steiner
- Aachen gen_sat3_n24_s2: +0.3585, +0.3585, +0.2737; 1188 / 1625 / 1188; steiner
- Aachen gen_tspunary_k3_s1: +0.0000, -0.0006, +0.0217; 112 / 112 / 109; usual
- Aachen gen_tspunary_k3_s2: +0.0000, -0.0008, +0.0214; 112 / 112 / 109; usual
- Aachen gen_tspunary_k3_s3: +0.0000, -0.0003, +0.0217; 112 / 112 / 109; usual
- Aachen gen_tspunary_k4_s1: +0.0465, +0.0465, +0.0073; 349 / 431 / 349; steiner
- Aachen gen_tspunary_k4_s2: +0.0465, +0.0465, +0.0068; 349 / 431 / 349; steiner
- Aachen gen_tspunary_k4_s3: +0.0468, +0.0468, +0.0073; 349 / 431 / 349; steiner
- Aachen gen_tspunary_k5_s1: +0.2429, +0.2429, +0.1660; 887 / 1141 / 887; steiner
- Aachen gen_tspunary_k5_s2: +0.2439, +0.2439, +0.1666; 887 / 1141 / 887; steiner
- Aachen gen_tspunary_k5_s3: +0.2448, +0.2448, +0.1654; 887 / 1141 / 887; steiner
- Boston gen_densegnp_n10_s1: +0.0098, +0.0098, +0.0098; 62 / 79 / 62; steiner
- Boston gen_densegnp_n10_s2: +0.0038, +0.0038, +0.0034; 52 / 58 / 52; steiner
- Boston gen_densegnp_n16_s1: +0.0375, +0.0375, +0.0242; 226 / 280 / 226; steiner
- Boston gen_densegnp_n16_s2: +0.0095, +0.0095, +0.0116; 242 / 271 / 242; steiner
- Boston gen_densegnp_n22_s1: +0.1312, +0.1312, +0.1156; 435 / 619 / 435; steiner
- Boston gen_densegnp_n22_s2: +0.0932, +0.0932, +0.0566; 487 / 628 / 487; steiner
- Boston gen_maxcut3reg_n12_s1: +0.0002, +0.0002, -0.0002; 72 / 70 / 72; steiner
- Boston gen_maxcut3reg_n12_s2: +0.0052, +0.0052, +0.0096; 66 / 74 / 66; steiner
- Boston gen_maxcut3reg_n24_s1: +0.0090, +0.0090, +0.0040; 196 / 190 / 196; steiner
- Boston gen_maxcut3reg_n24_s2: +0.0000, -0.0047, -0.0022; 161 / 161 / 172; usual
- Boston gen_maxcut3reg_n36_s1: +0.0000, -0.0361, +0.0367; 339 / 339 / 393; usual
- Boston gen_maxcut3reg_n36_s2: +0.0346, +0.0346, +0.0329; 315 / 348 / 315; steiner
- Boston gen_sat3_n14_s1: +0.1630, +0.1630, +0.1251; 440 / 684 / 440; steiner
- Boston gen_sat3_n14_s2: +0.0607, +0.0607, +0.0559; 452 / 564 / 452; steiner
- Boston gen_sat3_n24_s1: +0.3168, +0.3168, +0.3203; 1229 / 1731 / 1229; steiner
- Boston gen_sat3_n24_s2: +0.3422, +0.3422, +0.3049; 1162 / 1553 / 1162; steiner
- Boston gen_tspunary_k3_s1: +0.0096, +0.0096, +0.0085; 112 / 130 / 112; steiner
- Boston gen_tspunary_k3_s2: +0.0092, +0.0092, +0.0082; 112 / 130 / 112; steiner
- Boston gen_tspunary_k3_s3: +0.0093, +0.0093, +0.0085; 112 / 130 / 112; steiner
- Boston gen_tspunary_k4_s1: +0.0060, +0.0060, +0.0176; 385 / 423 / 385; steiner
- Boston gen_tspunary_k4_s2: +0.0050, +0.0050, +0.0163; 385 / 423 / 385; steiner
- Boston gen_tspunary_k4_s3: +0.0061, +0.0061, +0.0164; 385 / 423 / 385; steiner
- Boston gen_tspunary_k5_s1: +0.2850, +0.2850, +0.1770; 799 / 1184 / 799; steiner
- Boston gen_tspunary_k5_s2: +0.2855, +0.2855, +0.1769; 799 / 1184 / 799; steiner
- Boston gen_tspunary_k5_s3: +0.2846, +0.2846, +0.1767; 799 / 1184 / 799; steiner
- Fez gen_densegnp_n10_s1: +0.0277, +0.0277, +0.0188; 64 / 83 / 64; steiner
- Fez gen_densegnp_n10_s2: +0.0013, +0.0013, +0.0095; 52 / 53 / 52; steiner
- Fez gen_densegnp_n16_s1: +0.0462, +0.0462, +0.0344; 238 / 276 / 238; steiner
- Fez gen_densegnp_n16_s2: +0.0527, +0.0527, +0.0485; 236 / 272 / 236; steiner
- Fez gen_densegnp_n22_s1: +0.3987, +0.3987, +0.3444; 460 / 659 / 460; steiner
- Fez gen_densegnp_n22_s2: +0.4608, +0.4608, +0.3159; 453 / 648 / 453; steiner
- Fez gen_maxcut3reg_n12_s1: +0.0095, +0.0095, -0.0033; 73 / 77 / 73; steiner
- Fez gen_maxcut3reg_n12_s2: +0.0273, +0.0273, +0.0228; 71 / 83 / 71; steiner
- Fez gen_maxcut3reg_n24_s1: +0.0159, +0.0159, +0.0591; 191 / 191 / 191; steiner
- Fez gen_maxcut3reg_n24_s2: +0.0415, +0.0415, +0.0532; 137 / 175 / 137; steiner
- Fez gen_maxcut3reg_n36_s1: +0.0000, -0.0180, +75.3281; 392 / 392 / 420; usual
- Fez gen_maxcut3reg_n36_s2: +0.1459, +0.1459, +45.1080; 349 / 383 / 349; steiner
- Fez gen_sat3_n14_s1: +0.3267, +0.3267, +0.2552; 478 / 693 / 478; steiner
- Fez gen_sat3_n14_s2: +0.1901, +0.1901, +0.1811; 433 / 564 / 433; steiner
- Fez gen_sat3_n24_s1: +1.0843, +1.0843, +1.2354; 1169 / 1762 / 1169; steiner
- Fez gen_sat3_n24_s2: +1.1717, +1.1717, +1.2967; 1092 / 1637 / 1092; steiner
- Fez gen_tspunary_k3_s1: +0.0272, +0.0272, +0.0286; 110 / 138 / 110; steiner
- Fez gen_tspunary_k3_s2: +0.0271, +0.0271, +0.0285; 110 / 138 / 110; steiner
- Fez gen_tspunary_k3_s3: +0.0279, +0.0279, +0.0289; 110 / 138 / 110; steiner
- Fez gen_tspunary_k4_s1: +0.1837, +0.1837, +0.0984; 321 / 456 / 321; steiner
- Fez gen_tspunary_k4_s2: +0.1844, +0.1844, +0.0988; 321 / 456 / 321; steiner
- Fez gen_tspunary_k4_s3: +0.1847, +0.1847, +0.0982; 321 / 456 / 321; steiner
- Fez gen_tspunary_k5_s1: +0.2596, +0.2596, +0.3228; 883 / 1104 / 883; steiner
- Fez gen_tspunary_k5_s2: +0.2612, +0.2612, +0.3227; 883 / 1104 / 883; steiner
- Fez gen_tspunary_k5_s3: +0.2627, +0.2627, +0.3227; 883 / 1104 / 883; steiner
- Kingston gen_densegnp_n10_s1: +0.0138, +0.0138, +0.0095; 62 / 75 / 62; steiner
- Kingston gen_densegnp_n10_s2: +0.0019, +0.0019, +0.0041; 52 / 57 / 52; steiner
- Kingston gen_densegnp_n16_s1: +0.0368, +0.0368, +0.0175; 241 / 266 / 241; steiner
- Kingston gen_densegnp_n16_s2: +0.0461, +0.0461, +0.0456; 230 / 271 / 230; steiner
- Kingston gen_densegnp_n22_s1: +0.1680, +0.1680, +0.1316; 479 / 618 / 479; steiner
- Kingston gen_densegnp_n22_s2: +0.0917, +0.0917, +0.1208; 470 / 575 / 470; steiner
- Kingston gen_maxcut3reg_n12_s1: +0.0031, +0.0031, +0.0031; 74 / 69 / 74; steiner
- Kingston gen_maxcut3reg_n12_s2: +0.0115, +0.0115, +0.0116; 80 / 86 / 80; steiner
- Kingston gen_maxcut3reg_n24_s1: +0.0085, +0.0085, +0.0095; 190 / 182 / 190; steiner
- Kingston gen_maxcut3reg_n24_s2: +0.0401, +0.0401, +0.0183; 135 / 172 / 135; steiner
- Kingston gen_maxcut3reg_n36_s1: +0.0186, +0.0186, +0.0317; 405 / 379 / 405; steiner
- Kingston gen_maxcut3reg_n36_s2: +0.0101, +0.0101, +0.0415; 343 / 325 / 343; steiner
- Kingston gen_sat3_n14_s1: +0.1896, +0.1896, +0.1408; 478 / 651 / 478; steiner
- Kingston gen_sat3_n14_s2: +0.1136, +0.1136, +0.1065; 441 / 564 / 441; steiner
- Kingston gen_sat3_n24_s1: +0.5087, +0.5087, +0.4696; 1222 / 1682 / 1222; steiner
- Kingston gen_sat3_n24_s2: +0.5776, +0.5776, +0.5443; 1129 / 1601 / 1129; steiner
- Kingston gen_tspunary_k3_s1: +0.0088, +0.0088, +0.0172; 106 / 123 / 106; steiner
- Kingston gen_tspunary_k3_s2: +0.0090, +0.0090, +0.0165; 106 / 123 / 106; steiner
- Kingston gen_tspunary_k3_s3: +0.0090, +0.0090, +0.0166; 106 / 123 / 106; steiner
- Kingston gen_tspunary_k4_s1: +0.0737, +0.0737, +0.0378; 354 / 438 / 354; steiner
- Kingston gen_tspunary_k4_s2: +0.0729, +0.0729, +0.0355; 354 / 438 / 354; steiner
- Kingston gen_tspunary_k4_s3: +0.0732, +0.0732, +0.0378; 354 / 438 / 354; steiner
- Kingston gen_tspunary_k5_s1: +0.2410, +0.2410, +0.2696; 818 / 1117 / 818; steiner
- Kingston gen_tspunary_k5_s2: +0.2413, +0.2413, +0.2682; 818 / 1117 / 818; steiner
- Kingston gen_tspunary_k5_s3: +0.2413, +0.2413, +0.2695; 818 / 1117 / 818; steiner
- Marrakesh gen_densegnp_n10_s1: +0.0170, +0.0170, +0.0392; 81 / 85 / 81; steiner
- Marrakesh gen_densegnp_n10_s2: +0.0012, +0.0012, +0.0100; 64 / 59 / 64; steiner
- Marrakesh gen_densegnp_n16_s1: +0.1899, +0.1899, +0.1423; 222 / 270 / 222; steiner
- Marrakesh gen_densegnp_n16_s2: +0.0773, +0.0773, +0.0563; 242 / 285 / 242; steiner
- Marrakesh gen_densegnp_n22_s1: +0.4151, +0.4151, +0.2251; 443 / 641 / 443; steiner
- Marrakesh gen_densegnp_n22_s2: +0.3091, +0.3091, +0.3434; 492 / 602 / 492; steiner
- Marrakesh gen_maxcut3reg_n12_s1: +0.0000, -0.0100, +0.0002; 68 / 68 / 80; usual
- Marrakesh gen_maxcut3reg_n12_s2: +0.0182, +0.0182, +0.0174; 75 / 78 / 75; steiner
- Marrakesh gen_maxcut3reg_n24_s1: +0.0270, +0.0270, +0.0281; 181 / 199 / 181; steiner
- Marrakesh gen_maxcut3reg_n24_s2: +0.0000, -0.0093, +0.0268; 164 / 164 / 160; usual
- Marrakesh gen_maxcut3reg_n36_s1: +0.1297, +0.1297, +90.2887; 348 / 375 / 348; steiner
- Marrakesh gen_maxcut3reg_n36_s2: +0.2881, +0.2881, +45.2167; 311 / 399 / 311; steiner
- Marrakesh gen_sat3_n14_s1: +0.2501, +0.2501, +0.3363; 440 / 643 / 440; steiner
- Marrakesh gen_sat3_n14_s2: +0.2716, +0.2716, +0.2460; 446 / 578 / 446; steiner
- Marrakesh gen_sat3_n24_s1: +1.3109, +1.3109, +0.5825; 1211 / 1761 / 1211; steiner
- Marrakesh gen_sat3_n24_s2: +1.1046, +1.1046, +0.9909; 1118 / 1645 / 1118; steiner
- Marrakesh gen_tspunary_k3_s1: +0.0415, +0.0415, +0.0839; 122 / 130 / 122; steiner
- Marrakesh gen_tspunary_k3_s2: +0.0409, +0.0409, +0.0837; 122 / 130 / 122; steiner
- Marrakesh gen_tspunary_k3_s3: +0.0407, +0.0407, +0.0838; 122 / 130 / 122; steiner
- Marrakesh gen_tspunary_k4_s1: +0.1610, +0.1610, -0.0198; 371 / 431 / 371; steiner
- Marrakesh gen_tspunary_k4_s2: +0.1616, +0.1616, -0.0198; 371 / 431 / 371; steiner
- Marrakesh gen_tspunary_k4_s3: +0.1619, +0.1619, -0.0199; 371 / 431 / 371; steiner
- Marrakesh gen_tspunary_k5_s1: +0.6618, +0.6618, +0.6407; 809 / 1109 / 809; steiner
- Marrakesh gen_tspunary_k5_s2: +0.6684, +0.6684, +0.6433; 809 / 1109 / 809; steiner
- Marrakesh gen_tspunary_k5_s3: +0.6678, +0.6678, +0.6417; 809 / 1109 / 809; steiner
- Torino gen_densegnp_n10_s1: +0.0317, +0.0317, +0.0400; 62 / 80 / 62; steiner
- Torino gen_densegnp_n10_s2: +0.0164, +0.0164, +0.0372; 52 / 63 / 52; steiner
- Torino gen_densegnp_n16_s1: +0.1681, +0.1681, +0.1754; 226 / 267 / 226; steiner
- Torino gen_densegnp_n16_s2: +0.0526, +0.0526, +0.1261; 239 / 271 / 239; steiner
- Torino gen_densegnp_n22_s1: +0.3528, +0.3528, +0.3728; 446 / 594 / 446; steiner
- Torino gen_densegnp_n22_s2: +0.4310, +0.4310, +0.3176; 546 / 623 / 546; steiner
- Torino gen_maxcut3reg_n12_s1: +0.0133, +0.0133, -0.0080; 72 / 83 / 72; steiner
- Torino gen_maxcut3reg_n12_s2: +0.0207, +0.0207, +0.0135; 66 / 76 / 66; steiner
- Torino gen_maxcut3reg_n24_s1: +0.0256, +0.0256, -0.0205; 170 / 180 / 170; steiner
- Torino gen_maxcut3reg_n24_s2: +0.0436, +0.0436, +0.0991; 143 / 183 / 143; steiner
- Torino gen_maxcut3reg_n36_s1: +0.0000, -0.0071, +0.0176; 342 / 342 / 318; usual
- Torino gen_maxcut3reg_n36_s2: +0.1606, +0.1606, +0.0831; 330 / 370 / 330; steiner
- Torino gen_sat3_n14_s1: +0.4577, +0.4577, +0.3743; 484 / 653 / 484; steiner
- Torino gen_sat3_n14_s2: +0.3711, +0.3711, +0.3022; 449 / 616 / 449; steiner
- Torino gen_tspunary_k3_s1: +0.0350, +0.0350, +0.0505; 111 / 123 / 111; steiner
- Torino gen_tspunary_k3_s2: +0.0337, +0.0337, +0.0502; 111 / 123 / 111; steiner
- Torino gen_tspunary_k3_s3: +0.0343, +0.0343, +0.0507; 111 / 123 / 111; steiner
- Torino gen_tspunary_k4_s1: +0.0723, +0.0723, +0.1752; 324 / 406 / 324; steiner
- Torino gen_tspunary_k4_s2: +0.0713, +0.0713, +0.1755; 324 / 406 / 324; steiner
- Torino gen_tspunary_k4_s3: +0.0722, +0.0722, +0.1756; 324 / 406 / 324; steiner
- Torino gen_tspunary_k5_s1: +0.4992, +0.4992, +0.5918; 852 / 1145 / 852; steiner
- Torino gen_tspunary_k5_s2: +0.5011, +0.5011, +0.5904; 852 / 1145 / 852; steiner
- Torino gen_tspunary_k5_s3: +0.4989, +0.4989, +0.5913; 852 / 1145 / 852; steiner
