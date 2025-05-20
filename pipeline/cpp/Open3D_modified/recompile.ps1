$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition

Set-Location "$ScriptDir\build"
cmake --build . --config Release --parallel --target INSTALL
cmake --build . --config Release --target python-package
cmake --install .
