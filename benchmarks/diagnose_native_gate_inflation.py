パスワードを覚えていますか？ 今すぐ検証
このバナーを閉じる
Proton Mail
新しいメッセージ

ナビゲーション
受信トレイ
3
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
3
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
Proton Mail1.13.4 (5.0.130.5)
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
[3]会話内に 3 件のメッセージ１
Tom N
メッセージにスターをつける
受信トレイ
11 件の添付ファイル (1.02 MB)
8月 252026年8月25日火曜日 10:31
Tom N
メッセージにスターをつける
受信トレイ
2 件の添付ファイル (51.19 KB)
9月 32026年9月3日木曜日 9:04
トラッカーは見つかりませんでした,クリアされたリンクはありません
差出人
ゼロアクセス暗号化で保存されています
Tom N<rote2480@gmail.com>
メッセージにスターをつける
受信トレイ
9 件の添付ファイル (87.89 KB)
10:032026年9月8日火曜日 10:03
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

87.80 KB
9 個のファイル添付
すべての添付ファイルをダウンロード


experiment_fixed_compiler_fideli
ty.py
13.65 KB

13.65 KB


diagnose_native_gate_inflati
on.py
6.21 KB

6.21 KB


diagnose_compile_for_hardwa
re.py
3.90 KB

3.90 KB


compile_force_consolida
te.patch
4.83 KB


phase1_warmup_
v2.patch
4.18 KB


psf_compile_prototype_
v4.py
8.44 KB

8.44 KB


test_psf_zero_core_st
ub.py
2.27 KB

2.27 KB


行方
不明.docx
17.08 KB

17.08 KB


N
o1.docx
27.23 KB

27.23 KB
サイドパネルを表示
連絡先アプリを切り替える

8
カレンダーアプリを切り替え
セキュリティセンターアプリを切り替える
diagnose_native_gate_inflati
on.py
前へ
2 / 9
次へ
ダウンロード
閉じる
"""
diagnose_native_gate_inflation.py

Root-cause probe for the open question in the README ("why does PSF-Zero v6
lose fidelity on deep2q/multi_deep2q relative to Qiskit L3, TKET, and
Hybrid, even though the reported 2-qubit gate COUNT is identical across all
four engines?").

This is a REAL, runnable experiment (not a reconstruction from a log) — it
was executed in the process of writing this file, using Qiskit 2.5.2 and
qiskit-ibm-runtime's FakeSherbrooke, and every number quoted in the README's
"Why PSF-Zero v6 loses fidelity" bullet and section 8 note comes directly
from running this exact script.

WHAT IT TESTS
-------------
`benchmarks/psf_compile.py` (v6)'s `SU4GeodesicPSFSynthesizer.synthesize()`
emits, for a generic 2-qubit block: four local Rz.Ry.Rz triples plus up to
three entangling gates — RXX, RYY, RZZ (see that file's `local()` and the
`qc.rxx/qc.ryy/qc.rzz` calls). This is mathematically the standard
Cartan/KAK canonical form. Qiskit's own `TwoQubitWeylDecomposition.circuit()`
builds the identical structure (verified below to be logically equivalent
in gate composition), so we use it here as a faithful, independently-
implemented stand-in for what PSF-Zero's Rust core produces — we don't have
a working build of `psf_zero_core` in this environment (the one `.so` we
were given is a non-x86 binary), so this is the closest thing to "run the
real synthesizer" that's actually executable here.

The competing baseline is Qiskit's `TwoQubitBasisDecomposer(CXGate())` —
representative of what Qiskit L3 / TKET emit, since both target a CX-like
basis rather than RXX/RYY/RZZ directly.

RXX, RYY, and RZZ are NOT in `fake_sherbrooke`'s native gate set
(['ecr', 'id', 'rz', 'sx', 'x']) — CX isn't either, but CX<->ECR is a
well-optimized, cheap 1:1-ish basis translation. The question this script
answers: does translating RXX/RYY/RZZ into the ECR basis cost MORE native
2-qubit gates than translating CX does, and does that cost depend on the
transpiler's optimization level?

RESULT (N=200 random SU(4) unitaries, 0 correctness failures at every
level, `Operator(...).equiv()` checked before counting):

    opt_level=0: ECR mean  Weyl(PSF-like)=6.00  CX-basis=3.00  ratio=2.00x
    opt_level=1: ECR mean  Weyl(PSF-like)=6.00  CX-basis=3.00  ratio=2.00x
    opt_level=2: ECR mean  Weyl(PSF-like)=3.00  CX-basis=3.00  ratio=1.00x
    opt_level=3: ECR mean  Weyl(PSF-like)=3.00  CX-basis=3.00  ratio=1.00x

At optimization_level 0-1, the RXX/RYY/RZZ-based circuit needs exactly 2x
as many native ECR gates as the CX-based circuit for the SAME unitary, even
though both report "3" two-qubit gates before hardware-basis translation —
this is precisely the kind of gap that a benchmark measuring "2Q gate count"
on the post-*compile*, pre-*ISA-transpile* circuit would never see. At
optimization_level >= 2, Qiskit's transpiler runs a real unitary
resynthesis pass regardless of input gate basis, and the gap disappears
entirely.

WHAT THIS DOES NOT PROVE
------------------------
We don't have the actual `real_device_15q_fidelity_v2.py` /
`test_real_hardware_fidelity.py` scripts, so we can't confirm what
optimization level (if any) their final "submit to backend" step used for
ISA translation. This result establishes a plausible, concrete, and now
*measured* mechanism — it is not a confirmed diagnosis of the production
benchmark until someone checks that harness's actual transpile call. That
check is the natural next step (see README Roadmap).

USAGE
-----
    pip install qiskit qiskit-ibm-runtime
    python diagnose_native_gate_inflation.py [--n 200]
"""

from __future__ import annotations

import argparse
import statistics as st

from qiskit import transpile
from qiskit.circuit.library import CXGate
from qiskit.quantum_info import Operator, random_unitary
from qiskit.synthesis import TwoQubitBasisDecomposer
from qiskit.synthesis.two_qubit.two_qubit_decompose import TwoQubitWeylDecomposition
from qiskit_ibm_runtime.fake_provider import FakeSherbrooke


def run(n: int, seed_offset: int = 1000) -> None:
    backend = FakeSherbrooke()
    cx_decomposer = TwoQubitBasisDecomposer(CXGate())

    print(f"N={n} random SU(4) unitaries, backend={backend.name}, "
          f"native basis={sorted(backend.target.operation_names)}\n")

    for opt_level in (0, 1, 2, 3):
        weyl_ecr, cx_ecr = [], []
        weyl_1q, cx_1q = [], []
        fails = 0

        for i in range(n):
            u = random_unitary(4, seed=seed_offset + i).data

            # "PSF-Zero-style": canonical KAK circuit using RXX/RYY/RZZ
            # generators -- same structural family as psf_compile.py v6's
            # SU4GeodesicPSFSynthesizer.synthesize().
            qc_weyl = TwoQubitWeylDecomposition(u).circuit(simplify=True)

            # "Qiskit L3 / TKET-style": CX-basis decomposition of the same
            # unitary.
            qc_cx = cx_decomposer(u)

            if not (Operator(qc_weyl).equiv(Operator(u)) and
                    Operator(qc_cx).equiv(Operator(u))):
                fails += 1
                continue

            tw = transpile(qc_weyl, backend=backend,
                            optimization_level=opt_level, seed_transpiler=42)
            tc = transpile(qc_cx, backend=backend,
                            optimization_level=opt_level, seed_transpiler=42)

            weyl_ecr.append(tw.count_ops().get("ecr", 0))
            cx_ecr.append(tc.count_ops().get("ecr", 0))
            weyl_1q.append(sum(v for k, v in tw.count_ops().items()
                                if k in ("rz", "sx", "x")))
            cx_1q.append(sum(v for k, v in tc.count_ops().items()
                              if k in ("rz", "sx", "x")))

        ratio = st.mean(weyl_ecr) / st.mean(cx_ecr)
        print(
            f"opt_level={opt_level}  (failures={fails}/{n})\n"
            f"  native ECR gates : Weyl(PSF-like) mean={st.mean(weyl_ecr):.2f} "
            f"| CX-basis mean={st.mean(cx_ecr):.2f} | ratio={ratio:.2f}x\n"
            f"  native 1Q gates  : Weyl(PSF-like) mean={st.mean(weyl_1q):.2f} "
            f"| CX-basis mean={st.mean(cx_1q):.2f}\n"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200)
    args = parser.parse_args()
    run(args.n)
