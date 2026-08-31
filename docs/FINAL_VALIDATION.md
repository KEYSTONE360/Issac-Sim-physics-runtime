# Final validation — 2026-08-31

## Build identity

- Target: Isaac Sim 5.1 observable CPU/headless physics path
- Native solver: NVIDIA PhysX 5.6.1, Omniverse PhysX 107.3
- Auxiliary H1 evaluator: MuJoCo 3.12.0 + ONNX Runtime
- Packaged executable SHA-256: `39A065DBF73609781179768854E7D544138A39E5F17DE047073A7CC84704D94A`
- Native runtime SHA-256: `1E7F0B353DB36891A7D865F18C94D71494D45E31D01B690626BDF3D587DE1180`
- Locked H1 baseline SHA-256: `B24471D6FCBB4B61D5F50CD4AC01D35F1A767B99630118DCD1A9F531B70DAB4A`

## Passed checks

- Python unit suite: 25/25 passed twice.
- Native PhysX CTest: 1/1 passed.
- Packaged native self-test: passed.
- Packaged numbered-menu startup/exit and Korean UTF-8 output: passed.
- Packaged H1 evaluator: provided YAML + ONNX, 4,096 logical environments, one policy step: passed.
- Strict background lock: provided baseline matched exactly; only existing reward numeric values are editable.

## Engine cross-validation

- PhysX versus semi-implicit free fall at 0.5 s: position error `3.0e-7 m`; velocity error `2.0e-7 m/s`.
- PhysX versus matched MuJoCo free fall: position error `3.0e-7 m`; velocity error `2.0e-7 m/s`.
- PhysX versus MuJoCo resting box center height: absolute difference `1.078813e-4 m`.
- Repeated PhysX rigid/joint position/joint velocity/joint force values: difference `0` at emitted precision.
- Contacts observed in both solvers.

## Explicit limits

This build does not claim numerical identity with Isaac Sim or an official NCRC score. Full H1 import into the native PhysX articulation, native NCRC terrain parity, and Isaac Sim trace-based contact-force/observation/reward RMSE remain unverified. These states are emitted as `NOT_IMPLEMENTED` or `REFERENCE_TRACE_REQUIRED`; no fabricated result is produced.
