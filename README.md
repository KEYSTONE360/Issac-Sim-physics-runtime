# NCRC Physics Runtime / Reward Lab

목표 기준: Isaac Sim 5.1, PhysX SDK 5.6.1, Omniverse PhysX 107.3, CPU headless.

## 가장 쉬운 실행

배포 폴더의 `NCRC-Physics-Runtime.exe`를 실행하고 번호를 선택합니다.

1. PhysX native self-test: 실제 PxFoundation/PxPhysics/PxScene, rigid body, contact, articulation/joint state를 검사합니다.
2. H1 로컬 정책 평가: `env.yaml` 경로와 `policy.onnx` 경로, 반복 횟수를 입력합니다.
3. env 잠금 검증: 최초 제공 H1 YAML과 reward 이외의 모든 값을 정확 비교합니다.

전체 평가는 YAML의 `scene.num_envs=4096`, `sim.dt=0.005`, `decimation=4`를 임의로 줄이지 않습니다. 렌더러와 CUDA는 사용하지 않습니다.

주의: native PhysX H1 전체 articulation 이식은 개발 중입니다. 현재 메뉴 1은 PhysX native 검증 경로이고 메뉴 2의 전체 H1 정책 loop는 MuJoCo CPU 보조 백엔드입니다. 결과 JSON에 이 구분과 미지원 항목이 기록되며 공식 NCRC 점수로 표시하지 않습니다.

Experiment Intelligence & Reward Optimization Workbench for NCRC H1/Go2 reward experiments.

This repository treats official NCRC rules/source and official server artifacts as the highest-priority evidence. It does not label locally trained policies as submission-eligible and does not invent missing competition values.

## Quick start (Windows PowerShell)

```powershell
py -3.14 -m pip install -e .
ncrc-lab doctor
ncrc-lab analyze-source
ncrc-lab sync-isaaclab-reference
ncrc-lab environment set-default C:\path\to\env.yaml --robot H1
ncrc-lab rewards functions
ncrc-lab import-run C:\path\to\run --robot H1 --official-server
ncrc-lab experiments list
ncrc-lab best
ncrc-lab recommend --mode isolation
```

Place official starter code under `ncrc_source/` and current official rule files under `rules/`, then run `ncrc-lab analyze-source`. Until those files are supplied, their status remains `UNKNOWN`/`NOT_AVAILABLE`.

All generated data stays under this project directory: SQLite in `database/`, immutable run copies in `runs/`, generated manifests in `generated/`, and backups in `backups/`.

## Safety boundaries

- PT/PTH files are not deserialized by the core inspector.
- Artifact hashes use chunked I/O.
- Local training is never submission-eligible.
- `[WARN] skip` rewards are excluded from applied reward vectors.
- Full source patches are server-ready only when byte-aware validation confirms that existing `REWARD_WEIGHTS` numeric literals were the only changed bytes.
- Missing evidence returns `UNKNOWN`, `INCONCLUSIVE`, or `INSUFFICIENT_EVIDENCE`.

The NVIDIA reference catalog lists every public callable found in the pinned official `source/**/rewards.py` files. A callable becomes weight-editable only when an NCRC source or server `env.yaml` defines a corresponding reward term. This preserves the boundary between changing an existing weight and creating a new reward term/function.
