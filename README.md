# NCRC Physics Runtime / Reward Lab

Target baseline: Isaac Sim 5.1, PhysX SDK 5.6.1, Omniverse PhysX 107.3, CPU headless.

## Simplest way to run

Run `NCRC-Physics-Runtime.exe` in the distribution folder and select a number.

1. PhysX native self-test: verifies the actual PxFoundation/PxPhysics/PxScene path, rigid bodies, contacts, articulation, and joint state.
2. Engine cross-validation: compares repeated PhysX runs, a semi-implicit analytic solution, and a matched MuJoCo scene.
3. H1 local policy evaluation: accepts an `env.yaml` path, a `policy.onnx` path, and a repetition count.
4. Environment-lock validation: compares every non-reward value exactly against the initially supplied H1 YAML.

Full evaluation does not arbitrarily reduce `scene.num_envs=4096`, `sim.dt=0.005`, or `decimation=4` from the YAML. Rendering and CUDA are not used.

Important: the complete H1 articulation has not yet been ported to the native PhysX runtime. Menu item 1 exercises the native PhysX validation path, while menu item 3 runs the full H1 policy loop through the auxiliary MuJoCo CPU backend. Result JSON explicitly records this distinction and unsupported items, and never labels the result as an official NCRC score.

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

User-supplied YAML/ONNX/PT files and experiment data are not committed to GitHub. In a fresh clone, first register the baseline YAML with the `environment set-default` command above, then run `scripts\build_portable.ps1`. The local packaged build already contains the registered baseline YAML.

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
