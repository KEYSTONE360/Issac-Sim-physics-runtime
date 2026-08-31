$ErrorActionPreference = 'Stop'
$project = Split-Path -Parent $PSScriptRoot
$pythonScripts = Join-Path (python -c "import sysconfig; print(sysconfig.get_path('scripts'))") ''
$env:Path = "$pythonScripts;$env:Path"

cmake --build "$project\native_runtime\build" --config Release --parallel 12
python -m unittest discover -s "$project\tests" -v
python -m PyInstaller --noconfirm --clean "$project\build\NCRC-Physics-Runtime.spec"

$dist = "$project\dist\NCRC-Physics-Runtime"
New-Item -ItemType Directory -Force -Path "$dist\native", "$dist\presets\user\H1", "$dist\generated", "$dist\engine\vendor\mujoco_menagerie" | Out-Null
Copy-Item -LiteralPath "$project\native_runtime\build\Release\ncrc_physics.exe", "$project\native_runtime\build\Release\PhysX_64.dll", "$project\native_runtime\build\Release\PhysXCommon_64.dll", "$project\native_runtime\build\Release\PhysXFoundation_64.dll", "$project\native_runtime\build\Release\PhysXCooking_64.dll" -Destination "$dist\native" -Force
Copy-Item -LiteralPath "$project\presets\user\H1\server_env_background_default.yaml" -Destination "$dist\presets\user\H1" -Force
Copy-Item -LiteralPath "$project\generated\background_defaults_h1.json", "$project\generated\ncrc_runtime_profile.json" -Destination "$dist\generated" -Force
Copy-Item -LiteralPath "$project\engine\vendor\mujoco_menagerie\unitree_h1" -Destination "$dist\engine\vendor\mujoco_menagerie" -Recurse -Force
Copy-Item -LiteralPath "$project\README.md" -Destination $dist -Force

& "$dist\NCRC-Physics-Runtime.exe" native-test
& "$dist\NCRC-Physics-Runtime.exe" cross-validate
