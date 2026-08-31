# Implementation status

Target: Isaac Sim 5.1 observable physics path, PhysX SDK 5.6.1, Omniverse PhysX 107.3, CPU, headless.

## Verified native PhysX path

- `PxFoundation`, `PxPhysics`, `PxScene`, CPU dispatcher
- 0.005 s timestep and TGS solver selection
- rigid-body pose, quaternion, world/body velocity extraction
- ground collision and contact point/normal/impulse/separation extraction
- reduced-coordinate articulation foundation
- link pose/velocity/mass/COM/inertia extraction
- joint position, velocity, finite-difference acceleration, incoming joint-axis torque
- renderer disabled and no CUDA dependency

Native CTest and packaged executable self-tests pass.

## Engine cross-validation

- PhysX versus semi-implicit analytic free-fall: position error about `3e-7 m`, velocity error about `2e-7 m/s` at 0.5 s.
- PhysX versus MuJoCo matched free-fall: the same error scale.
- Resting box center-height difference: about `1.08e-4 m`.
- Repeated PhysX rigid/articulation probe values are deterministic at emitted precision.

Run `NCRC-Physics-Runtime.exe cross-validate` to regenerate the JSON report.

## Implemented auxiliary H1 evaluation path

The auxiliary CPU backend supports the provided H1 19-joint model, 256-element observation, ONNX policy inference, decimation, ordered float32 reward accumulation, 4096 logical environments, repetitions, ETA, and complete term statistics. It is explicitly labeled MuJoCo and is not an official NCRC score.

## Not yet verified

- Full H1 model imported into the native PhysX articulation
- NCRC terrain mesh parity in native PhysX
- Isaac Sim trace-based contact-force, observation, and reward RMSE
- Exact NCRC training/PPO pipeline and official scoring

These are reported as `NOT_IMPLEMENTED` or `REFERENCE_TRACE_REQUIRED`; no placeholder numeric result is emitted.
