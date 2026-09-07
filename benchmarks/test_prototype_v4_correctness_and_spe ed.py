パスワードを覚えていますか？ 今すぐ検証
このバナーを閉じる
Proton Mail
新しいメッセージ

ナビゲーション
受信トレイ
2
下書き
送信済み
スター付き
隠す
隠す
迷惑メール
6
アーカイブ
ごみ箱
すべてのメール
2
表示
表示
フォルダ
フォルダ
新しいフォルダを作成
フォルダを管理
ラベル
ラベル
新しいラベルを作成
ラベルを管理
現在のストレージ: 20.60 GB / 500.00 GB
20.60 GB / 500.00 GB
Proton Mail1.13.4 (5.0.130.4)
戻る
未読にする
ごみ箱に移動
アーカイブに移動
迷惑メールに移動


スヌーズ
前の会話
次の会話

特別オファー
 
設定を切り替え

love.os.architect
love.os.architect@pm.me
L
[6]会話内に 6 件のメッセージ1
Tom N
メッセージにスターをつける
受信トレイ
3 件の添付ファイル (96.39 KB)
8月 262026年8月26日水曜日 9:20
Tom N
メッセージにスターをつける
受信トレイ
2 件の添付ファイル (85.02 KB)
8月 312026年8月31日月曜日 15:21
Tom N
メッセージにスターをつける
受信トレイ
5 件の添付ファイル (206.17 KB)
9月 22026年9月2日水曜日 16:17
Tom N
メッセージにスターをつける
受信トレイ
11 件の添付ファイル (776.04 KB)
9月 42026年9月4日金曜日 12:41
トラッカーは見つかりませんでした,クリアされたリンクはありません
差出人
ゼロアクセス暗号化で保存されています
Tom N<rote2480@gmail.com>
メッセージにスターをつける
受信トレイ
16 件の添付ファイル (519.44 KB)
8:502026年9月7日月曜日 8:50
宛先
love.os.architect@proton.me
詳細を表示
未読にする
ゴミ箱に移動
移動先
ラベル付け
絞り込み
詳細設定を表示
返信
全員に返信
転送

519.35 KB
16 個のファイル添付
すべての添付ファイルをダウンロード


compile_optional_veri
fy.patch
5.96 KB


compile_for_hardware_verify_passthrough
_1.patch
2.36 KB


make_chart_compile_time_scali
ng.py
4.84 KB

4.84 KB


phase3_v5_seed
ed.py
12.25 KB

12.25 KB


phase1_verify_fal
se.patch
3.55 KB


phase3_v4_dense_pair_bloc
ks.py
11.86 KB

11.86 KB


phase3_v3_physical_topolo
gy.py
10.49 KB

10.49 KB


profile_qiskit_multiprocess_vs_mainproce
ss.py
5.36 KB

5.36 KB


profile_compile_for_hardware_breakdo
wn.py
5.50 KB

5.50 KB


profile_synthesize_fast_vs_verifi
ed.py
5.66 KB

5.66 KB


profile_warmup_dep
th.py
4.93 KB

4.93 KB


profile_synthesize_breakdo
wn.py
2.93 KB

2.93 KB


test_prototype_v4_correctness_and_spe
ed.py
3.00 KB

3.00 KB


psf_compile_prototype_
v4.py
8.57 KB

8.57 KB


compile_time_scaling
_3.png
215.81 KB

215.81 KB


compile_time_scaling
_2.png
216.26 KB

216.26 KB
loveos
メッセージにスターをつける
送信済み
8:502026年9月7日月曜日 8:50
サイドパネルを表示
連絡先アプリを切り替える

7
カレンダーアプリを切り替え
セキュリティセンターアプリを切り替える
test_prototype_v4_correctness_and_spe
ed.py
前へ
13 / 16
次へ
ダウンロード
閉じる
"""
Sanity check for psf_compile_prototype_v4.py, meant to run on the real
machine with the real psf_zero_core Rust extension (not the stub used
elsewhere in this project's own sandbox).

Two things are checked:
  1. Correctness: verify=True and verify=False must produce circuits that
     are Operator-equivalent to the original input (and to each other) --
     verify=False turns off the per-block self-check, not the math.
  2. Speed: verify=False should be meaningfully faster than verify=True on
     the same circuit, consistent with (though not necessarily identical in
     magnitude to) the stub-core measurement of 8.11x on synthesize() alone
     (benchmarks/profile_synthesize_fast_vs_verified.py) and 6.8x on the
     full compile() pipeline in this project's own sandbox re-check.

Run this before trusting the projected section 4 numbers in the README, or
before considering verify=False for any real use.
"""
import time
import numpy as np
from scipy.stats import unitary_group
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from qiskit.circuit.library import UnitaryGate

import psf_compile_prototype_v4 as pcp


def build_chain_circuit(num_pairs, gates_per_pair, seed):
    """Same generator shape as phase1.py/phase2.py's dense same-pair chains."""
    r = np.random.default_rng(seed)
    qc = QuantumCircuit(num_pairs * 2)
    for p in range(num_pairs):
        for _ in range(gates_per_pair):
            U = unitary_group.rvs(4, random_state=r)
            qc.append(UnitaryGate(U), [2 * p, 2 * p + 1])
    return qc


def main():
    print("=== 1. Correctness (small circuit, Operator-equivalence check) ===")
    qc_small = build_chain_circuit(num_pairs=5, gates_per_pair=20, seed=1)
    outs = {}
    for verify in (True, False):
        outs[verify] = pcp.compile(qc_small, verify=verify)
        eq = Operator(qc_small).equiv(Operator(outs[verify]))
        print(f"verify={verify}: equivalent to original input = {eq}")
    eq_cross = Operator(outs[True]).equiv(Operator(outs[False]))
    print(f"verify=True output equivalent to verify=False output = {eq_cross}")
    if not (Operator(qc_small).equiv(Operator(outs[True]))
            and Operator(qc_small).equiv(Operator(outs[False]))):
        print("\n*** FAIL: verify=False changed correctness. Do not use it. ***")
        return

    print("\n=== 2. Speed (larger circuit, matching section 4's 156-qubit/78-block scale) ===")
    qc_big = build_chain_circuit(num_pairs=78, gates_per_pair=20, seed=2)
    times = {}
    for verify in (True, False):
        t0 = time.perf_counter()
        pcp.compile(qc_big, verify=verify)
        times[verify] = time.perf_counter() - t0
        print(f"verify={verify}: {times[verify]:.4f}s")
    print(f"\nSpeedup from verify=False: {times[True] / times[False]:.2f}x")
    print("(compare against this project's stub-core sandbox result of ~6.8x on")
    print(" the full compile() pipeline, and ~8.11x on synthesize() alone)")


if __name__ == "__main__":
    main()
