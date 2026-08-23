#!/usr/bin/env bash
# tarball_install.sh - installer INSIDE the release tarball (run as root from
# the unpacked tarball root). Idempotent.
#
#   tar -xzf mortred_model_server-<version>-<profile>-linux-x64.tar.gz
#   cd mortred_model_server-<version>-<profile>-linux-x64 && sudo ./install.sh
#
# What it does:
#   1. apt runtime deps (ubuntu 20.04/22.04; no build toolchain needed)
#   2. /opt/mortred tree + mortred system user
#   3. systemd unit (reads /etc/mortred/supervisor.env for tokens; fail-closed:
#      without tokens the supervisor binds loopback only)
#   4. weights hint: fetch_weights.py --profile <profile>
set -euo pipefail

[ "$(id -u)" -eq 0 ] || { echo "[ERROR] run as root: sudo ./install.sh" >&2; exit 1; }

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREFIX="/opt/mortred"
PROFILE="$(cat "$SRC/opt/mortred/PROFILE" 2>/dev/null || echo gpu)"

echo "== [1/4] runtime deps (profile: $PROFILE) =="
export DEBIAN_FRONTEND=noninteractive
apt-get update
PKGS="ca-certificates curl libssl3 libgoogle-glog0v6 \
      libopencv-core4.5d libopencv-imgproc4.5d libopencv-imgcodecs4.5d"
# ubuntu 20.04 ships opencv 4.2 / libssl1.1; 22.04 ships 4.5d / libssl3
if ! apt-get install -y --no-install-recommends $PKGS >/dev/null 2>&1; then
    PKGS="ca-certificates curl libssl1.1 libgoogle-glog0v5 \
          libopencv-core4.2 libopencv-imgproc4.2 libopencv-imgcodecs4.2"
    apt-get install -y --no-install-recommends $PKGS
fi
if [ "$PROFILE" = "gpu" ]; then
    # NVIDIA runtime stack (TensorRT/cuDNN): install via the NVIDIA apt repo
    # when absent; the gpu tarball's bundled 3rd_party libs need them at dlopen
    if ! ldconfig -p 2>/dev/null | grep -q libnvinfer.so; then
        curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb -o /tmp/cuda-keyring.deb
        dpkg -i /tmp/cuda-keyring.deb && rm -f /tmp/cuda-keyring.deb
        apt-get update
        apt-get install -y --no-install-recommends libnvinfer8 libnvinfer-plugin8 libnvonnxparser8 libcudnn8 ocl-icd-libopencl1
    fi
fi

echo "== [2/4] install tree -> $PREFIX =="
id -u mortred >/dev/null 2>&1 || useradd -r -s /usr/sbin/nologin mortred
mkdir -p "$PREFIX"
cp -a "$SRC/opt/mortred/." "$PREFIX/"
mkdir -p /etc/mortred
chown -R mortred:mortred "$PREFIX"

echo "== [3/4] systemd unit =="
if [ "$PROFILE" = "cpu" ]; then
    # cpu deployments must not depend on a GPU being present
    sed -i '\|^Environment=MORTRED_PROFILE=|d; s|^\[Service\]|[Service]\nEnvironment=MORTRED_PROFILE=cpu|' \
        "$SRC/deploy/mortred-supervisor.service"
fi
cp "$SRC/deploy/mortred-supervisor.service" /etc/systemd/system/
if [ ! -f /etc/mortred/supervisor.env ]; then
    cat > /etc/mortred/supervisor.env <<'EOF'
# REQUIRED for non-loopback serving (fail-closed without them):
# MORTRED_API_TOKEN=<management token>
# MORTRED_GATEWAY_AUTH_TOKEN=<inference token>
EOF
    chmod 600 /etc/mortred/supervisor.env
fi
systemctl daemon-reload
systemctl enable mortred-supervisor

echo "== [4/4] next steps =="
cat <<EOF
  1. edit /etc/mortred/supervisor.env (set both tokens; chmod 600 kept)
  2. download weights:   cd $PREFIX && python3 scripts/fetch_weights.py --profile $PROFILE
  3. start:              sudo systemctl start mortred-supervisor
  4. verify:             curl -fs http://127.0.0.1:8787/api/v1/health
EOF
echo "== install done (profile: $PROFILE) =="
