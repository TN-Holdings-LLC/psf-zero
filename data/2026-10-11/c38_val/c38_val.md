# C38-VAL

jobs: 232; records with an error: 12

- FakeFez REL -nonpbc-qubitnodes_Lx-11_Ly-11_U-100_enc-gray_d-4]: FailedElementsError: "no placement avoids the failed couplers and qubits ('A connected component of the DAGCircuit is too large for any of the connect
- FakeFez C38 -nonpbc-qubitnodes_Lx-11_Ly-11_U-100_enc-gray_d-4]: FailedElementsError: "no placement avoids the failed couplers and qubits ('A connected component of the DAGCircuit is too large for any of the connect
- FakeKingston REL -nonpbc-qubitnodes_Lx-11_Ly-11_U-100_enc-gray_d-4]: FailedElementsError: "no placement avoids the failed couplers and qubits ('A connected component of the DAGCircuit is too large for any of the connect
- FakeKingston C38 -nonpbc-qubitnodes_Lx-11_Ly-11_U-100_enc-gray_d-4]: FailedElementsError: "no placement avoids the failed couplers and qubits ('A connected component of the DAGCircuit is too large for any of the connect
- FakeAachen REL -nonpbc-qubitnodes_Lx-11_Ly-11_U-100_enc-gray_d-4]: FailedElementsError: "no placement avoids the failed couplers and qubits ('A connected component of the DAGCircuit is too large for any of the connect
- FakeAachen C38 -nonpbc-qubitnodes_Lx-11_Ly-11_U-100_enc-gray_d-4]: FailedElementsError: "no placement avoids the failed couplers and qubits ('A connected component of the DAGCircuit is too large for any of the connect
- FakeMarrakesh REL -nonpbc-qubitnodes_Lx-11_Ly-11_U-100_enc-gray_d-4]: FailedElementsError: "no placement avoids the failed couplers and qubits ('A connected component of the DAGCircuit is too large for any of the connect
- FakeMarrakesh C38 -nonpbc-qubitnodes_Lx-11_Ly-11_U-100_enc-gray_d-4]: FailedElementsError: "no placement avoids the failed couplers and qubits ('A connected component of the DAGCircuit is too large for any of the connect
- FakeTorino REL riag-pbc-qubitnodes_Lx-3_Ly-22_U-70_enc-unary_d-4]: FailedElementsError: "no placement avoids the failed couplers and qubits ('A connected component of the DAGCircuit is too large for any of the connect
- FakeTorino C38 riag-pbc-qubitnodes_Lx-3_Ly-22_U-70_enc-unary_d-4]: FailedElementsError: "no placement avoids the failed couplers and qubits ('A connected component of the DAGCircuit is too large for any of the connect

| device | commuting tests (STEINER used) | V1 C38 on failed | V2 non-commuting: C38 = REL | runnable commuting | V3 ESP C38/REL | V4 ESP C38/QK3 | V5 2q C38/REL (all commuting) | V6 time C38/REL (commuting) | QK3 lost to failed (runnable) |
|---|---|---|---|---|---|---|---|---|---|
| FakeTorino | 19 | 0 | 17 / 17 | 17 | 1.412 (17) | 1.393 (15) | 0.877 | 3.57 | 2 |
| FakeKingston | 20 | 0 | 18 / 18 | 19 | 1.468 (19) | 1.407 (19) | 0.890 | 8.16 | 0 |
| FakeMarrakesh | 20 | 0 | 18 / 18 | 17 | 1.563 (17) | 1.573 (14) | 0.857 | 9.04 | 3 |
| FakeFez | 20 | 0 | 18 / 18 | 17 | 1.367 (17) | 1.422 (15) | 0.859 | 9.52 | 2 |
| FakeAachen | 20 | 0 | 18 / 18 | 19 | 1.340 (19) | 1.287 (19) | 0.967 | 9.87 | 0 |
| FakeBoston | 20 | 0 | 18 / 18 | 19 | 1.216 (19) | 1.186 (19) | 0.955 | 8.50 | 0 |
| all | 119 | 0 | 107 / 107 | 108 | 1.387 (108) | 1.361 (101) | 0.900 | 8.69 | 7 |

## Commuting runnable tests, per test (log10 ESP C38 - REL, C38 - QK3; two-qubit C38 / REL / QK3)

- Aachen gen_maxcut3reg_n10_s1: -0.0020, +0.0039; 60 / 55 / 54
- Aachen gen_maxcut3reg_n10_s2: -0.0012, +0.0017; 63 / 59 / 57
- Aachen gen_maxcut3reg_n16_s1: -0.0197, -0.0244; 136 / 104 / 104
- Aachen gen_maxcut3reg_n16_s2: -0.0055, +0.0055; 111 / 102 / 108
- Aachen gen_maxcut3reg_n22_s1: +0.0124, +0.0105; 154 / 164 / 162
- Aachen gen_maxcut3reg_n22_s2: +0.0200, +0.0145; 168 / 168 / 162
- Aachen gen_maxcut3reg_n28_s1: -0.0238, -0.0062; 264 / 209 / 217
- Aachen gen_maxcut3reg_n28_s2: -0.0395, -0.0266; 301 / 232 / 224
- Aachen gen_maxcut3reg_n34_s1: +0.0062, +0.0128; 354 / 290 / 305
- Aachen gen_maxcut3reg_n34_s2: +0.0627, +0.0824; 241 / 274 / 306
- Aachen gen_maxcut3reg_n40_s1: +0.0039, +0.0369; 410 / 371 / 397
- Aachen gen_maxcut3reg_n40_s2: +0.0705, +0.1495; 349 / 370 / 370
- Aachen gen_sat3_n12_s1: +0.0906, +0.0655; 344 / 426 / 441
- Aachen gen_sat3_n12_s2: +0.1134, +0.1056; 323 / 439 / 417
- Aachen gen_sat3_n20_s1: +0.3874, +0.2876; 882 / 1280 / 1268
- Aachen gen_sat3_n20_s2: +0.3557, +0.3240; 818 / 1231 / 1178
- Aachen gen_sat3_n30_s1: +0.6663, +0.5230; 1929 / 2613 / 2500
- Aachen gen_sat3_n30_s2: +0.7645, +0.5646; 1725 / 2469 / 2357
- Aachen anspile[ham_tsp_prob-ts225_Ncity-5_enc-unary]: -0.0468, -0.0478; 544 / 496 / 508
- Boston gen_maxcut3reg_n10_s1: +0.0027, +0.0041; 47 / 56 / 54
- Boston gen_maxcut3reg_n10_s2: +0.0039, -0.0006; 57 / 59 / 57
- Boston gen_maxcut3reg_n16_s1: +0.0001, +0.0015; 103 / 99 / 104
- Boston gen_maxcut3reg_n16_s2: -0.0131, -0.0035; 126 / 94 / 108
- Boston gen_maxcut3reg_n22_s1: +0.0142, +0.0140; 161 / 162 / 162
- Boston gen_maxcut3reg_n22_s2: +0.0266, +0.0166; 147 / 173 / 162
- Boston gen_maxcut3reg_n28_s1: -0.0019, -0.0183; 252 / 236 / 217
- Boston gen_maxcut3reg_n28_s2: +0.0234, +0.0065; 234 / 244 / 224
- Boston gen_maxcut3reg_n34_s1: -0.0037, -0.0086; 322 / 311 / 305
- Boston gen_maxcut3reg_n34_s2: -0.0125, -0.0075; 342 / 299 / 306
- Boston gen_maxcut3reg_n40_s1: -0.0419, -0.0037; 433 / 360 / 397
- Boston gen_maxcut3reg_n40_s2: -0.0870, -0.0390; 462 / 322 / 370
- Boston gen_sat3_n12_s1: +0.0742, +0.0739; 316 / 436 / 441
- Boston gen_sat3_n12_s2: +0.0618, +0.0726; 314 / 416 / 417
- Boston gen_sat3_n20_s1: +0.2977, +0.2951; 871 / 1305 / 1268
- Boston gen_sat3_n20_s2: +0.2505, +0.2048; 849 / 1197 / 1178
- Boston gen_sat3_n30_s1: +0.5302, +0.4500; 1839 / 2526 / 2500
- Boston gen_sat3_n30_s2: +0.5316, +0.3779; 1778 / 2454 / 2357
- Boston anspile[ham_tsp_prob-ts225_Ncity-5_enc-unary]: -0.0422, -0.0292; 553 / 467 / 508
- Fez gen_maxcut3reg_n10_s1: +0.0119, -0.0028; 56 / 67 / 54
- Fez gen_maxcut3reg_n10_s2: -0.0032, -0.0047; 60 / 59 / 57
- Fez gen_maxcut3reg_n16_s1: -0.0137, -0.0275; 124 / 112 / 104
- Fez gen_maxcut3reg_n16_s2: +0.0136, +0.0393; 94 / 109 / 108
- Fez gen_maxcut3reg_n22_s1: -0.0019, -0.0017; 168 / 164 / 162
- Fez gen_maxcut3reg_n22_s2: +0.0750, +0.0490; 140 / 177 / 162
- Fez gen_maxcut3reg_n28_s1: +0.0600, +0.0499; 217 / 241 / 217
- Fez gen_maxcut3reg_n28_s2: -0.0367, +0.0347; 248 / 237 / 224
- Fez gen_maxcut3reg_n34_s1: -0.0161, +0.0558; 366 / 339 / 305
- Fez gen_maxcut3reg_n34_s2: +0.1049, +0.2744; 285 / 331 / 306
- Fez gen_maxcut3reg_n40_s1: +0.1206, +75.4454; 391 / 409 / 397
- Fez gen_maxcut3reg_n40_s2: +0.1713, +44.9322; 362 / 402 / 370
- Fez gen_sat3_n12_s1: +0.1293, +0.1600; 352 / 417 / 441
- Fez gen_sat3_n12_s2: +0.2404, +0.1925; 297 / 461 / 417
- Fez gen_sat3_n20_s1: +0.7895, +0.7039; 856 / 1296 / 1268
- Fez gen_sat3_n20_s2: +0.7677, +0.7472; 815 / 1276 / 1178
- Fez anspile[ham_tsp_prob-ts225_Ncity-5_enc-unary]: -0.1042, +0.0223; 527 / 484 / 508
- Kingston gen_maxcut3reg_n10_s1: +0.0077, +0.0064; 55 / 55 / 54
- Kingston gen_maxcut3reg_n10_s2: +0.0029, -0.0006; 60 / 61 / 57
- Kingston gen_maxcut3reg_n16_s1: -0.0002, -0.0018; 118 / 104 / 104
- Kingston gen_maxcut3reg_n16_s2: -0.0065, +0.0008; 118 / 100 / 108
- Kingston gen_maxcut3reg_n22_s1: +0.0179, +0.0170; 144 / 162 / 162
- Kingston gen_maxcut3reg_n22_s2: +0.0243, +0.0156; 144 / 174 / 162
- Kingston gen_maxcut3reg_n28_s1: +0.0373, +0.0134; 214 / 210 / 217
- Kingston gen_maxcut3reg_n28_s2: +0.0427, +0.0456; 215 / 232 / 224
- Kingston gen_maxcut3reg_n34_s1: +0.0125, -0.0036; 331 / 313 / 305
- Kingston gen_maxcut3reg_n34_s2: +0.0724, +0.0526; 231 / 311 / 306
- Kingston gen_maxcut3reg_n40_s1: +0.0929, +0.0875; 378 / 396 / 397
- Kingston gen_maxcut3reg_n40_s2: +0.0465, +0.0171; 411 / 376 / 370
- Kingston gen_sat3_n12_s1: +0.0950, +0.0767; 326 / 444 / 441
- Kingston gen_sat3_n12_s2: +0.1057, +0.0811; 324 / 448 / 417
- Kingston gen_sat3_n20_s1: +0.4220, +0.4815; 766 / 1247 / 1268
- Kingston gen_sat3_n20_s2: +0.3997, +0.2844; 834 / 1231 / 1178
- Kingston gen_sat3_n30_s1: +0.9749, +0.7799; 1766 / 2580 / 2500
- Kingston gen_sat3_n30_s2: +0.8709, +0.8567; 1686 / 2462 / 2357
- Kingston anspile[ham_tsp_prob-ts225_Ncity-5_enc-unary]: -0.0514, +0.0074; 511 / 423 / 508
- Marrakesh gen_maxcut3reg_n10_s1: +0.0106, +0.0331; 47 / 61 / 54
- Marrakesh gen_maxcut3reg_n10_s2: -0.0043, +0.0072; 64 / 59 / 57
- Marrakesh gen_maxcut3reg_n16_s1: +0.0051, +0.0032; 114 / 104 / 104
- Marrakesh gen_maxcut3reg_n16_s2: -0.0178, +0.0191; 94 / 90 / 108
- Marrakesh gen_maxcut3reg_n22_s1: +0.0442, +0.0537; 171 / 193 / 162
- Marrakesh gen_maxcut3reg_n22_s2: +0.0227, +0.0663; 145 / 162 / 162
- Marrakesh gen_maxcut3reg_n28_s1: -0.0207, +0.0829; 218 / 204 / 217
- Marrakesh gen_maxcut3reg_n28_s2: +0.0885, +0.1072; 221 / 252 / 224
- Marrakesh gen_maxcut3reg_n34_s1: +0.0443, +30.2401; 293 / 313 / 305
- Marrakesh gen_maxcut3reg_n34_s2: +0.2184, +0.2088; 243 / 320 / 306
- Marrakesh gen_maxcut3reg_n40_s1: +0.3523, +75.1967; 429 / 418 / 397
- Marrakesh gen_maxcut3reg_n40_s2: +0.2289, +30.5854; 348 / 370 / 370
- Marrakesh gen_sat3_n12_s1: +0.1257, +0.2403; 358 / 426 / 441
- Marrakesh gen_sat3_n12_s2: +0.3085, +0.1970; 318 / 477 / 417
- Marrakesh gen_sat3_n20_s1: +0.9853, +0.9463; 846 / 1284 / 1268
- Marrakesh gen_sat3_n20_s2: +0.9004, +0.7582; 795 / 1216 / 1178
- Marrakesh anspile[ham_tsp_prob-ts225_Ncity-5_enc-unary]: +0.0058, +0.0298; 566 / 508 / 508
- Torino gen_maxcut3reg_n10_s1: +0.0209, +0.0142; 55 / 55 / 52
- Torino gen_maxcut3reg_n10_s2: +0.0229, +0.0273; 57 / 64 / 63
- Torino gen_maxcut3reg_n16_s1: -0.0053, -0.0009; 107 / 97 / 101
- Torino gen_maxcut3reg_n16_s2: +0.0371, -0.0012; 108 / 105 / 91
- Torino gen_maxcut3reg_n22_s1: +0.0672, +0.0376; 144 / 186 / 165
- Torino gen_maxcut3reg_n22_s2: +0.0809, +0.0424; 145 / 173 / 166
- Torino gen_maxcut3reg_n28_s1: +0.1108, +0.0586; 237 / 250 / 233
- Torino gen_maxcut3reg_n28_s2: +0.1006, +0.0773; 210 / 246 / 239
- Torino gen_maxcut3reg_n34_s1: -0.0588, +45.2353; 300 / 264 / 307
- Torino gen_maxcut3reg_n34_s2: +0.1894, +0.0645; 238 / 306 / 280
- Torino gen_maxcut3reg_n40_s1: +0.0205, +75.0142; 437 / 363 / 366
- Torino gen_maxcut3reg_n40_s2: -0.1080, +0.0718; 352 / 329 / 365
- Torino gen_sat3_n12_s1: +0.1915, +0.1872; 343 / 434 / 432
- Torino gen_sat3_n12_s2: +0.3924, +0.2533; 342 / 479 / 453
- Torino gen_sat3_n20_s1: +0.7731, +0.8572; 989 / 1253 / 1277
- Torino gen_sat3_n20_s2: +0.8322, +0.8118; 856 / 1192 / 1187
- Torino anspile[ham_tsp_prob-ts225_Ncity-5_enc-unary]: -0.1213, -0.3408; 569 / 521 / 456
