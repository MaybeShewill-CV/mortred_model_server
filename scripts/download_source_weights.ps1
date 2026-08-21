#!/usr/bin/env pwsh
# download_source_weights.ps1 - Download source weights for models that need
# onnx self-export into weights_src/.
#
# Background: under the ONNX-first strategy, models without a prebuilt onnx on
# HF need their official source weights (.pt/.pth/pdparams/TF checkpoint)
# downloaded first, then exported to onnx locally (see docs/onnx-export-guide.md).
# This script downloads them into $ROOT/weights_src/<model>/, isolated from the
# runtime weights/ tree.
#
# Usage (run from repo root, Windows PowerShell 5.1+ / pwsh 7+):
#   ./scripts/download_source_weights.ps1                # download all
#   ./scripts/download_source_weights.ps1 -Only yolov6    # only entries matching yolov6
#   ./scripts/download_source_weights.ps1 -List           # print manifest only
#
# Deps: curl.exe (ships with Windows 10 1803+). To use a proxy:
#   $env:HTTPS_PROXY="http://127.0.0.1:7897"; ./scripts/download_source_weights.ps1

param(
    [string]$Only = "",
    [switch]$List
)

$ErrorActionPreference = "Stop"
$ROOT = Split-Path -Parent $PSScriptRoot
$DST_ROOT = Join-Path $ROOT "weights_src"

# Manifest: name -> @{ url = direct link; note = source description }
# Entries with url = "MANUAL" require manual download (see note).
# lightglue is intentionally ABSENT: weights/feature_point/lightglue/ already has
# extractor.onnx / matcher.onnx / superpoint_lightglue_end2end.onnx.
$ITEMS = [ordered]@{
    # ---- verified direct links ----
    "yolov6/yolov6n.pt" = @{ url = "https://huggingface.co/kadirnar/yolov6n-v3.0/resolve/main/yolov6n.pt"; note = "HF mirror kadirnar (Meituan YOLOv6 v3.0 official); export via tools/export_onnx.py" }
    "yolov6/yolov6s.pt" = @{ url = "https://huggingface.co/kadirnar/yolov6s-v3.0/resolve/main/yolov6s.pt"; note = "HF mirror kadirnar" }
    "yolov6/yolov6m.pt" = @{ url = "https://huggingface.co/kadirnar/yolov6m-v3.0/resolve/main/yolov6m.pt"; note = "HF mirror kadirnar" }
    "yolov6/yolov6l.pt" = @{ url = "https://huggingface.co/kadirnar/yolov6l-v3.0/resolve/main/yolov6l.pt"; note = "HF mirror kadirnar" }
    "superpoint/superpoint_v1.pth" = @{ url = "https://raw.githubusercontent.com/magicleap/SuperPointPretrainedNetwork/master/superpoint_v1.pth"; note = "official MagicLeap weights; one checkpoint exported per 5 resolutions" }
    # ---- HF lijiacai/pp-matting (modnet + ppmatting paddlesource), filenames verified pattern ----
    "modnet/modnet-hrnet_w18.pdparams" = @{ url = "https://huggingface.co/lijiacai/pp-matting/resolve/main/models/modnet-hrnet_w18.pdparams"; note = "PaddleSeg PP-ModNet hrnet_w18 (HF mirror); if 404 list files under models/ and adjust name" }
    "ppmatting/ppmatting-hrnet_w18.pdparams" = @{ url = "https://huggingface.co/lijiacai/pp-matting/resolve/main/models/ppmatting-hrnet_w18.pdparams"; note = "PaddleSeg PP-Matting hrnet_w18 (HF mirror); if 404 list files under models/ and adjust name" }
    "ppmatting/ppmatting-resnet34_vd.pdparams" = @{ url = "https://huggingface.co/lijiacai/pp-matting/resolve/main/models/ppmatting-resnet34_vd.pdparams"; note = "PaddleSeg PP-Matting resnet34_vd (HF mirror); if 404 list files under models/ and adjust name" }
    # ---- manual / candidate URLs (not fully verified) ----
    "nanodet/nanodet-plus-m_1.5x_coco.pth" = @{ url = "MANUAL"; note = "RangiLyu/nanodet GitHub Releases asset; check https://github.com/RangiLyu/nanodet/releases (or official Baidu drive link in repo README)" }
    "bisenetv2/bisenetv2_cityscapes.tar" = @{ url = "MANUAL"; note = "PaddleSeg pretrained: try https://paddleseg.bj.bcebos.com/models/bisenetv2_cityscapes_1024x1024_160k.tar (candidate, verify); or use mmsegmentation .pth" }
    "pphumanseg/pp_humansegv2_lite.pdmodel" = @{ url = "MANUAL"; note = "PaddleSeg contrib/PP-HumanSeg: run src/download_pretrained_models.py from the contrib dir (official script)" }
    "pphumanseg/pp_humansegv2_server.pdmodel" = @{ url = "MANUAL"; note = "PaddleSeg contrib/PP-HumanSeg: run src/download_pretrained_models.py (official script)" }
    "enlightengan/Epoch_latest.pth" = @{ url = "MANUAL"; note = "soralire/EnlightenGAN official checkpoint is Google-Drive-hosted (link in repo README); no HF mirror with weights" }
    "attentive_gan_derain/derain.pb" = @{ url = "MANUAL"; note = "MaybeShewill-CV/attentive-gan-derainnet TF1 checkpoint in-repo; freeze_graph then tf2onnx (see docs/onnx-export-guide.md)" }
}

function Fail($msg) { Write-Host "[ERROR] $msg" -ForegroundColor Red; exit 1 }

if ($List) {
    foreach ($k in $ITEMS.Keys) {
        $it = $ITEMS[$k]
        $disp = if ($it.url -eq "MANUAL") { "MANUAL: $($it.note)" } else { "<- $($it.url)" }
        Write-Host ("{0,-48} {1}" -f $k, $disp)
    }
    exit 0
}

$curl = Get-Command curl.exe -ErrorAction SilentlyContinue
if (-not $curl) { Fail "curl.exe not found (Windows 10 1803+ ships it)" }

$proxyArgs = @()
if ($env:HTTPS_PROXY) { $proxyArgs = @("-x", $env:HTTPS_PROXY) }

$n_ok = 0; $n_fail = 0; $n_manual = 0
foreach ($k in $ITEMS.Keys) {
    if ($Only -and $k -notlike "*$Only*") { continue }
    $it = $ITEMS[$k]
    if ($it.url -eq "MANUAL") {
        Write-Host "[skip] $k (MANUAL download required)" -ForegroundColor Yellow
        Write-Host "       $($it.note)"
        $n_manual++
        continue
    }
    $dst = Join-Path $DST_ROOT ($k -replace "/", [IO.Path]::DirectorySeparatorChar)
    $dir = Split-Path -Parent $dst
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    if (Test-Path $dst -PathType Leaf) {
        $mb = [math]::Round((Get-Item $dst).Length / 1MB, 1)
        Write-Host "[skip] $k ($mb MB already present)"
        $n_ok++
        continue
    }
    Write-Host "[get ] $k"
    Write-Host "       <- $($it.url)  ($($it.note))"
    & $curl.Source -sSL --fail --retry 3 --connect-timeout 15 @proxyArgs -o $dst $it.url
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[FAIL] $k (curl exit $LASTEXITCODE)" -ForegroundColor Red
        $n_fail++
        continue
    }
    $mb = [math]::Round((Get-Item $dst).Length / 1MB, 1)
    Write-Host "[ok  ] $k ($mb MB)"
    $n_ok++
}

Write-Host ""
Write-Host "== done: $n_ok ok, $n_fail failed, $n_manual manual"
Write-Host "target dir: $DST_ROOT"
if ($n_fail -gt 0) { exit 1 }
