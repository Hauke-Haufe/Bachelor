$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition

Set-Location "$ScriptDir\build"
cmake --build . --config Release --parallel --target INSTALL
cmake --install .