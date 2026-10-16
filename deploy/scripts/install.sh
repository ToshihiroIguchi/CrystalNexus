#!/usr/bin/env bash
# CrystalNexus installer for Ubuntu Server 24.04 (run as root on the VM).
#
# Usage:
#   sudo bash install.sh [-u BASIC_AUTH_USER] [-p BASIC_AUTH_PASSWORD] [-r REPO_URL] [-b BRANCH] [-s SERVER_NAME_OR_IP]
#
# Set CRYSTALNEXUS_TORCH_CPU=0 in the environment (sudo env CRYSTALNEXUS_TORCH_CPU=0 bash ...) to skip CPU-only torch.
# Without -u/-p you are prompted. Safe to re-run: existing user, venv, cert and
# htpasswd entries are reused or updated.
set -euo pipefail

REPO_URL="https://github.com/ToshihiroIguchi/CrystalNexus"
BRANCH="main"
APP_USER="crystalnexus"
BASE_DIR="/opt/crystalnexus"
APP_DIR="${BASE_DIR}/app"
VENV_DIR="${BASE_DIR}/venv"
ENV_FILE="/etc/crystalnexus/crystalnexus.env"
HTPASSWD_FILE="/etc/nginx/.htpasswd-crystalnexus"
SSL_DIR="/etc/nginx/ssl"
AUTH_USER=""
AUTH_PASS=""
SERVER_NAME=""
# Set CRYSTALNEXUS_TORCH_CPU=0 to skip the CPU-only PyTorch step (e.g. on a GPU VM).
TORCH_CPU="${CRYSTALNEXUS_TORCH_CPU:-1}"

while getopts "u:p:r:b:s:h" opt; do
  case "$opt" in
    u) AUTH_USER="$OPTARG" ;;
    p) AUTH_PASS="$OPTARG" ;;
    r) REPO_URL="$OPTARG" ;;
    b) BRANCH="$OPTARG" ;;
    s) SERVER_NAME="$OPTARG" ;;
    h|*) sed -n '2,8p' "$0"; exit 1 ;;
  esac
done

if [[ $EUID -ne 0 ]]; then
  echo "ERROR: run this script as root (use: sudo bash $0)" >&2
  exit 1
fi


echo "==> [1/9] Installing OS packages"
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y python3.12 python3.12-venv python3-pip git nginx apache2-utils openssl ufw build-essential curl

echo "==> [2/9] Creating system user '${APP_USER}'"
if ! id "$APP_USER" >/dev/null 2>&1; then
  useradd --system --create-home --home-dir "$BASE_DIR" --shell /usr/sbin/nologin "$APP_USER"
else
  echo "    user already exists, skipping"
fi

echo "==> [3/9] Fetching the application into ${APP_DIR}"
if [[ -d "${APP_DIR}/.git" ]]; then
  sudo -u "$APP_USER" git -C "$APP_DIR" fetch origin
  sudo -u "$APP_USER" git -C "$APP_DIR" checkout "$BRANCH"
  sudo -u "$APP_USER" git -C "$APP_DIR" pull --ff-only origin "$BRANCH"
else
  sudo -u "$APP_USER" git clone --branch "$BRANCH" "$REPO_URL" "$APP_DIR"
fi
mkdir -p "${APP_DIR}/uploads"
chown -R "$APP_USER:$APP_USER" "$BASE_DIR"

echo "==> [4/9] Creating the Python virtual environment and installing dependencies (this can take 10+ minutes)"
if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  sudo -u "$APP_USER" python3.12 -m venv "$VENV_DIR"
fi
sudo -u "$APP_USER" "${VENV_DIR}/bin/pip" install --upgrade pip
if [[ "$TORCH_CPU" != "0" ]]; then
  # Install the CPU-only torch wheel first (much smaller than the default CUDA build on
  # Linux, no GPU needed). The version is read from requirements.txt; the later
  # "pip install -r" then sees torch==X as already satisfied (X+cpu matches X).
  TORCH_VER="$(sed -n 's/^torch==\([^ ;#]*\).*/\1/p' "${APP_DIR}/requirements.txt" | head -n1)"
  if [[ -n "$TORCH_VER" ]]; then
    echo "    installing CPU-only torch==${TORCH_VER}"
    sudo -u "$APP_USER" "${VENV_DIR}/bin/pip" install "torch==${TORCH_VER}" --index-url https://download.pytorch.org/whl/cpu
  else
    echo "    WARNING: no torch pin found in requirements.txt; skipping CPU-only step"
  fi
else
  echo "    CRYSTALNEXUS_TORCH_CPU=0: skipping CPU-only torch (default torch build will be used)"
fi
sudo -u "$APP_USER" "${VENV_DIR}/bin/pip" install -r "${APP_DIR}/requirements.txt"
echo "    installed torch build:"
sudo -u "$APP_USER" "${VENV_DIR}/bin/python" -c "import torch; print('   ', torch.__version__, 'cuda_available=' + str(torch.cuda.is_available()))"
TORCH_INSTALLED="$("${VENV_DIR}/bin/python" -c 'import torch; print(torch.__version__)')"
if [[ "$TORCH_CPU" != "0" && "$TORCH_INSTALLED" == *"+cu"* ]]; then
  echo "    WARNING: torch ${TORCH_INSTALLED} is a CUDA build (large download, not needed without a GPU)."
  echo "             Fix: ${VENV_DIR}/bin/pip install --force-reinstall --no-deps torch==${TORCH_INSTALLED%%+*} --index-url https://download.pytorch.org/whl/cpu"
fi

echo "==> [5/9] Installing the environment file and systemd unit"
mkdir -p "$(dirname "$ENV_FILE")"
if [[ ! -f "$ENV_FILE" ]]; then
  ANALYTICS_TOKEN="$(openssl rand -base64 32 | tr -d '/+=' | cut -c1-40)"
  cat > "$ENV_FILE" <<ENVEOF
# CrystalNexus environment overrides (read by systemd). Restart after editing:
#   sudo systemctl restart crystalnexus
# Token the app requires for /analytics. Nginx adds it automatically for Basic-auth users
# (rendered into /etc/nginx/sites-available/crystalnexus.conf). To rotate, see docs/deployment-nutanix.md.
CRYSTALNEXUS_ANALYTICS_TOKEN=${ANALYTICS_TOKEN}
# Other optional settings (defaults shown in main.py):
#MAX_FILE_SIZE=52428800
#RELAX_TIMEOUT_SECONDS=600
#MAX_SESSIONS=100
ENVEOF
  chmod 640 "$ENV_FILE"
  chown root:"$APP_USER" "$ENV_FILE"
else
  echo "    ${ENV_FILE} exists, keeping it"
fi
install -m 644 "${APP_DIR}/deploy/systemd/crystalnexus.service" /etc/systemd/system/crystalnexus.service
systemctl daemon-reload

echo "==> [6/9] Generating a self-signed TLS certificate"
mkdir -p "$SSL_DIR"
if [[ -z "$SERVER_NAME" ]]; then
  SERVER_NAME="$(hostname -I | awk '{print $1}')"
fi
if [[ ! -f "${SSL_DIR}/crystalnexus.crt" ]]; then
  SAN="IP:${SERVER_NAME}"
  [[ "$SERVER_NAME" =~ ^[0-9.]+$ ]] || SAN="DNS:${SERVER_NAME}"
  openssl req -x509 -nodes -newkey rsa:2048 -days 825 \
    -keyout "${SSL_DIR}/crystalnexus.key" -out "${SSL_DIR}/crystalnexus.crt" \
    -subj "/CN=${SERVER_NAME}" -addext "subjectAltName=${SAN}"
  chmod 600 "${SSL_DIR}/crystalnexus.key"
else
  echo "    certificate exists, keeping it"
fi

echo "==> [7/9] Configuring HTTP Basic authentication"
if [[ -z "$AUTH_USER" ]]; then
  read -r -p "    Basic auth username: " AUTH_USER
fi
if [[ -z "$AUTH_PASS" ]]; then
  read -r -s -p "    Basic auth password: " AUTH_PASS
  echo
fi
if [[ -z "$AUTH_USER" || -z "$AUTH_PASS" ]]; then
  echo "ERROR: username and password must not be empty" >&2
  exit 1
fi
# -i reads the password from stdin (keeps it out of the process list); -c only for a new file.
if [[ -f "$HTPASSWD_FILE" ]]; then
  printf '%s' "$AUTH_PASS" | htpasswd -i -B "$HTPASSWD_FILE" "$AUTH_USER"
else
  printf '%s' "$AUTH_PASS" | htpasswd -i -B -c "$HTPASSWD_FILE" "$AUTH_USER"
fi
chown root:www-data "$HTPASSWD_FILE"
chmod 640 "$HTPASSWD_FILE"

echo "==> [8/9] Installing the Nginx site"
mkdir -p /etc/nginx/snippets
install -m 644 "${APP_DIR}/deploy/nginx/crystalnexus-proxy.conf" /etc/nginx/snippets/crystalnexus-proxy.conf
# Reuse the analytics token from the env file (generated in step 5) and render it into
# the Nginx site so Basic-auth users can open /analytics without a token.
ANALYTICS_TOKEN="$(sed -n 's/^CRYSTALNEXUS_ANALYTICS_TOKEN=//p' "$ENV_FILE" | head -n1)"
if [[ -z "$ANALYTICS_TOKEN" ]]; then
  echo "ERROR: CRYSTALNEXUS_ANALYTICS_TOKEN is empty or missing in ${ENV_FILE}; set it and re-run" >&2
  exit 1
fi
NGINX_SITE=/etc/nginx/sites-available/crystalnexus.conf
install -m 640 -o root -g www-data "${APP_DIR}/deploy/nginx/crystalnexus.conf" "$NGINX_SITE"
sed -i "s|__ANALYTICS_TOKEN__|${ANALYTICS_TOKEN}|g" "$NGINX_SITE"
ln -sf /etc/nginx/sites-available/crystalnexus.conf /etc/nginx/sites-enabled/crystalnexus.conf
rm -f /etc/nginx/sites-enabled/default
nginx -t

echo "==> [9/9] Enabling firewall and starting services"
ufw allow 22/tcp
ufw allow 443/tcp
ufw --force enable
systemctl enable --now crystalnexus
systemctl enable nginx
systemctl restart nginx

echo
echo "Done."
echo "  URL:            https://${SERVER_NAME}/   (browser warns about the self-signed certificate)"
echo "  Login:          ${AUTH_USER} / (the password you entered)"
echo "  Analytics page: https://${SERVER_NAME}/analytics   (same Basic auth login, no token needed)"
echo "  Verify:         systemctl status crystalnexus --no-pager"
echo "                  curl -k -u ${AUTH_USER} https://127.0.0.1/health"
echo "  Logs:           journalctl -u crystalnexus -f"
