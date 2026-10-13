# Deploying CrystalNexus on a Linux VM on Nutanix AHV

This guide is for complete beginners. It walks you from "I have a Nutanix cluster" to "my colleagues can open CrystalNexus in a browser", using one Ubuntu Linux VM. The Prism Central menu names below are described generically and may differ slightly by version.

## 1. Overview

```
 Browser (HTTPS, user + password)
        |
        v   port 443
+-----------------------------------------------+
|  Ubuntu 24.04 VM on Nutanix AHV               |
|                                               |
|  ufw firewall: allow 22 (SSH) and 443 only    |
|        |                                      |
|  Nginx  (HTTPS, Basic auth)                   |
|        |  proxy to 127.0.0.1:8090             |
|  CrystalNexus (uvicorn, venv, systemd)        |
|        |                                      |
|  /opt/crystalnexus/app/analytics.db (SQLite)  |
+-----------------------------------------------+
        ^
        |  daily snapshot, kept 7 days (Nutanix protection policy)
```

Why this layout: CrystalNexus has **no login of its own**, so Nginx adds HTTPS and a username/password (HTTP Basic auth), and the firewall exposes only what is needed. The app itself listens only on `127.0.0.1:8090` (the default) and is never reachable directly.

Files used (all in this repository):

| File | Purpose |
|------|---------|
| `deploy/nutanix/cloud-init/user-data.yaml` | First-boot setup of the VM (packages, firewall, SSH key) |
| `deploy/scripts/install.sh` | Installs the app, systemd service, Nginx, certificate, Basic auth |
| `deploy/systemd/crystalnexus.service` | systemd unit (auto-restart on failure) |
| `deploy/nginx/crystalnexus.conf` | Nginx reverse proxy site (the analytics token placeholder is filled in by `install.sh`) |
| `deploy/nginx/crystalnexus-proxy.conf` | Shared proxy settings (timeouts aside, headers and SSE buffering) included by the site |

## 2. Prerequisites checklist

- [ ] Access to Prism Central with permission to upload images and create VMs
- [ ] A network (VLAN/subnet) for the VM, with DHCP or a free static IP, reachable by your users
- [ ] The VM can reach the internet (GitHub, PyPI, Ubuntu mirrors) for installation. If not, ask your network admin for a proxy or mirror.
- [ ] An SSH key pair on your PC. To create one: `ssh-keygen -t ed25519` (then `cat ~/.ssh/id_ed25519.pub` to see the public key)
- [ ] The Ubuntu 24.04 cloud image: `https://cloud-images.ubuntu.com/noble/current/noble-server-cloudimg-amd64.img`
- [ ] The edited `user-data.yaml` (hostname, your SSH public key, time zone)

## 3. Sizing

| Item | Recommended | Notes |
|------|-------------|-------|
| OS | Ubuntu Server 24.04 LTS (cloud image) | Ships Python 3.12, which the lock file needs |
| vCPU | 8 | ML relaxations are CPU-bound (CPU-only PyTorch, no GPU) |
| RAM | 16 GB (32 GB for large structures) | Each loaded structure is held in memory |
| Disk | 80 GB | With CPU-only PyTorch the Python environment is a few GB (the default CUDA build would add many more GB); MACE weights download on first use |
| GPU | none | |
| Availability | 1 VM, no HA | About 3 users per day; short downtime is acceptable |

## 4. Step-by-step

### A. Nutanix Prism Central: create the VM

1. **Upload the image.** In Prism Central, open the image management area (usually *Compute & Storage > Images*, or *Infrastructure > Images*). Choose *Add Image* / *Upload Image*. Select the cluster, and either upload the `.img` file or give it the download URL above. Set the image type to **Disk** (not ISO). Name it e.g. `ubuntu-24.04-cloudimg`. Wait until the status is *Active*.
2. **Create the VM.** Go to the VM list and choose *Create VM*.
   - Name: `crystalnexus`
   - vCPUs / memory: 8 vCPU, 16-32 GB (see the sizing table)
   - Disk: add a disk **cloned from the image** you uploaded, and set its size to **80 GB**. (Cloud images are small, about 3.5 GB, and grow to the size you set at first boot.)
   - Boot: UEFI is fine; make sure this disk is the boot device.
   - Network: attach a NIC on your target subnet.
3. **Attach cloud-init.** In the VM's *Guest Customization* section (the label may read "Custom Script" or "Cloud-init"), choose cloud-init and paste the contents of `deploy/nutanix/cloud-init/user-data.yaml` after replacing every `REPLACE_ME` (hostname, SSH public key, time zone). Do not leave the placeholder key in.
4. **Power on** the VM and wait a few minutes (cloud-init installs packages on first boot; allow about 5 minutes).
5. **Find the IP address.** Open the VM's details (NICs / Network tab) in Prism Central, or check your DHCP server. Write it down; below it is shown as `VM_IP`.
6. **Nutanix Guest Tools (NGT).** Mount NGT from Prism Central (VM *Manage Guest Tools* or similar) and install it inside the VM following the Nutanix documentation for Linux. NGT is needed for application-consistent snapshots and some management features. Cloud-init already installs `qemu-guest-agent`, which is harmless. Skipping NGT still works, but snapshots are then only crash-consistent.

### B. Connect with SSH

From your PC (replace `VM_IP`):

```bash
ssh ubuntu@VM_IP
```

Check that cloud-init has finished:

```bash
cloud-init status --wait
```

It should print `status: done`. If SSH asks about a fingerprint, type `yes`.

### C. Run the installer

On the VM:

```bash
git clone https://github.com/ToshihiroIguchi/CrystalNexus.git
cd CrystalNexus
sudo bash deploy/scripts/install.sh
```

The script asks for the Basic-auth username and password that users will type in the browser. To pass them on the command line instead (note: they are saved in your shell history):

```bash
sudo bash deploy/scripts/install.sh -u myuser -p 'a-long-password' -s VM_IP
```

Optional flags: `-r REPO_URL` (use a fork/mirror), `-b BRANCH`, `-s SERVER_NAME_OR_IP` (name placed in the certificate; defaults to the VM's first IP).

The installation step for Python packages takes a while (10+ minutes) because PyTorch is large. When it finishes it prints the URL. The script is safe to run again; it reuses the existing user, venv and certificate, and updates the password.

What it does, in order: installs OS packages, creates the `crystalnexus` system user, clones the app to `/opt/crystalnexus/app`, creates the venv in `/opt/crystalnexus/venv`, installs the CPU-only build of PyTorch (see below) and runs `pip install -r requirements.txt`, writes `/etc/crystalnexus/crystalnexus.env` (including a random `CRYSTALNEXUS_ANALYTICS_TOKEN`, reused on re-runs), installs the systemd unit, generates a self-signed certificate, creates the Basic-auth user, installs the Nginx site, opens ufw for 22 and 443, and starts everything.

#### About CPU-only PyTorch

On Linux, `pip install torch` downloads the default build, which bundles the NVIDIA CUDA libraries (several GB of extra download and disk, and a much longer install). This VM has no GPU, so `install.sh` first installs the CPU-only build of the exact torch version pinned in `requirements.txt` (currently `2.14.1`, available as `2.14.1+cpu` for Linux / Python 3.12), then runs `pip install -r requirements.txt`, which keeps it because the pin is already satisfied. If you have a GPU VM, skip this with `sudo env CRYSTALNEXUS_TORCH_CPU=0 bash deploy/scripts/install.sh`.

If you install by hand instead of using `install.sh` (inside the venv, in this order):

```bash
TORCH_VER=$(sed -n 's/^torch==\([^ ;#]*\).*/\1/p' requirements.txt)
pip install "torch==$TORCH_VER" --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Verify:

```bash
/opt/crystalnexus/venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Expected: `2.14.1+cpu False`. A `+cu...` suffix means the CUDA build was installed; fix with `pip install --force-reinstall --no-deps torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu` (use the version from `requirements.txt`).

### D. Verify

On the VM:

```bash
systemctl status crystalnexus --no-pager
curl -s http://127.0.0.1:8090/health
curl -ks -o /dev/null -w '%{http_code}\n' https://127.0.0.1/
curl -ks -u myuser:'a-long-password' https://127.0.0.1/health
```

Expected results, in order: `active (running)`; `{"status":"healthy","service":"CrystalNexus"}`; `401` (login required); the same healthy JSON again.

From your PC's browser: open `https://VM_IP/`. The browser warns that the certificate is not trusted (it is self-signed): choose *Advanced > Proceed*. Enter the username and password. You should see the CrystalNexus page; try loading a sample CIF.

The analytics dashboard needs no extra step: open `https://VM_IP/analytics` after the same Basic-auth login. (The app itself requires a token for analytics; Nginx adds it automatically for logged-in users and strips any token a browser sends on other URLs.)

About the certificate: a self-signed certificate is acceptable for internal use, but every user sees a warning. A certificate issued by your organization's internal CA (installed as trusted on client PCs) is better: put its files at `/etc/nginx/ssl/crystalnexus.crt` and `/etc/nginx/ssl/crystalnexus.key`, then run `sudo nginx -t && sudo systemctl reload nginx`.

### E. Backup and protection policy

Application data is small: `analytics.db` (usage history only; the app does not keep uploaded CIF files) and the env file. The simplest and most complete backup is a VM snapshot.

1. In Prism Central open the data protection area (often *Data Protection > Protection Policies*; names vary by version).
2. Create a policy: local snapshot **daily**, retention **7 days**, and add the `crystalnexus` VM (directly or via a category).
3. Optionally take a manual snapshot before every update.

To restore: use *Restore* on a snapshot in Prism Central (as a new VM or replacing the VM, depending on your version).

### F. Updating the app

```bash
ssh ubuntu@VM_IP
sudo -u crystalnexus git -C /opt/crystalnexus/app pull --ff-only
sudo -u crystalnexus /opt/crystalnexus/venv/bin/pip install -r /opt/crystalnexus/app/requirements.txt
sudo systemctl restart crystalnexus
systemctl status crystalnexus --no-pager
```

Torch must stay the CPU build. Because `requirements.txt` pins `torch==X`, the `pip install -r` line normally keeps the installed `X+cpu`. If `requirements.txt` changed the torch version in your update, or you see a `+cu` suffix, re-pin before restarting:

```bash
sudo -u crystalnexus /opt/crystalnexus/venv/bin/pip install "torch==NEW_VERSION" --index-url https://download.pytorch.org/whl/cpu
sudo -u crystalnexus /opt/crystalnexus/venv/bin/pip install -r /opt/crystalnexus/app/requirements.txt
/opt/crystalnexus/venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Take a snapshot first if you want an easy rollback. Dependency versions are locked in `requirements.txt` (see "Updating dependencies" in the README for how maintainers change them); after a `git pull`, the `pip install -r` line above is all that is needed. OS security updates: `sudo apt update && sudo apt upgrade -y` (reboot if asked).

### G. Troubleshooting

| Symptom | What to check |
|---------|---------------|
| Service not starting | `sudo systemctl status crystalnexus --no-pager` and `sudo journalctl -u crystalnexus -n 100 --no-pager`. Follow live with `sudo journalctl -u crystalnexus -f`. Common causes: pip install did not finish (re-run the pip line in section F); wrong permissions (`sudo chown -R crystalnexus:crystalnexus /opt/crystalnexus`). |
| 502 Bad Gateway from Nginx | The app is not running or still starting (loading takes some seconds). Check `curl -s http://127.0.0.1:8090/health`, then `sudo systemctl restart crystalnexus`. Also `sudo tail -n 50 /var/log/nginx/error.log`. |
| Nginx config problem | `sudo nginx -t`, then `sudo systemctl reload nginx` |
| Cannot reach from browser | On the VM, `curl -ks -o /dev/null -w '%{http_code}\n' https://127.0.0.1/` should print 401. If it does, the problem is the network: check `sudo ufw status` (443 allowed, and your PC inside any CIDR you restricted to), the VM's subnet/VLAN and any network firewall. Use `https://`, not `http://`. |
| Browser keeps asking for a password | Wrong credentials. Reset with `sudo htpasswd -B /etc/nginx/.htpasswd-crystalnexus myuser` |
| Analytics page says 401 | The token in `/etc/crystalnexus/crystalnexus.env` and in `/etc/nginx/sites-available/crystalnexus.conf` no longer match (or the site file was replaced without rendering the token). Re-run `install.sh` (it reuses the token from the env file), or follow the token rotation steps in section H. |
| Relaxation stops with a timeout | Default limit is 600 s (`RELAX_TIMEOUT_SECONDS`). Set a larger value in `/etc/crystalnexus/crystalnexus.env` and restart. Nginx allows 900 s per request. |
| Upload rejected as too large | App limit `MAX_FILE_SIZE` (default 50 MB); Nginx `client_max_body_size` is 55 MB. Raise both together. |
| Out of memory (process killed) | `free -h`; `sudo journalctl -k \| grep -i "out of memory"`. Add RAM to the VM (power off, change memory in Prism Central), or lower `MAX_SESSIONS` / `MAX_TOTAL_SITES` in the env file. |
| pip install very slow or "No space left on device" | `df -h`. Check that torch is the CPU build (`+cpu`, see "About CPU-only PyTorch"); the CUDA build is several GB larger and the main cause of slow or full-disk installs. Confirm the disk was grown to 80 GB (`lsblk`; if not, `sudo growpart /dev/sda 1 && sudo resize2fs /dev/sda1`, device names may differ). Free space with `sudo -u crystalnexus /opt/crystalnexus/venv/bin/pip cache purge`. Slow downloads usually mean a slow proxy or internet link; re-run the installer or pip line. |
| First MACE use is slow | MACE weights download to `/opt/crystalnexus/.cache/mace` on first use; the VM needs internet access for that. |

After editing the env file: `sudo systemctl restart crystalnexus`.

### H. Security notes

- **CrystalNexus has no built-in authentication.** Anyone who can reach the app can use it. This deployment therefore adds Nginx HTTP Basic auth over HTTPS and a firewall. Basic auth is a shared-password style gate, not per-user access control; use strong passwords.
- **Do not expose this VM to the internet.** Keep it on an internal network and, where possible, restrict the source network at the network firewall in front of the VM and optionally with ufw, for example:
  ```bash
  sudo ufw delete allow 443/tcp
  sudo ufw allow from 192.168.0.0/16 to any port 443 proto tcp   # replace with your internal CIDR
  ```
- The app listens only on `127.0.0.1:8090`; do not set `CRYSTALNEXUS_HOST` to `0.0.0.0` here (that is the "LAN mode" from the README, which would bypass Nginx).
- Behind Nginx the app sees the real client address through `X-Forwarded-For` (uvicorn trusts it from 127.0.0.1 only). Because requests then come from a non-loopback address, the app requires `CRYSTALNEXUS_ANALYTICS_TOKEN` for `/analytics` and `/api/analytics/*`. `install.sh` generates it once, stores it in `/etc/crystalnexus/crystalnexus.env`, and renders it into the Nginx site (`/etc/nginx/sites-available/crystalnexus.conf`, mode 640 root:www-data, so it is a secret file). Nginx sends the token only on those two locations and clears any client-supplied `X-Analytics-Token` elsewhere.
- **Anyone who has a Basic-auth account can open the analytics dashboard**, which shows visitor IP addresses, User-Agents, and uploaded file names and formulas. Share the Basic-auth login accordingly.
- **Rotating the token** (for example after someone leaves):
  ```bash
  OLD=$(sudo sed -n 's/^CRYSTALNEXUS_ANALYTICS_TOKEN=//p' /etc/crystalnexus/crystalnexus.env)
  NEW=$(openssl rand -base64 32 | tr -d '/+=' | cut -c1-40)
  sudo sed -i "s|$OLD|$NEW|" /etc/crystalnexus/crystalnexus.env /etc/nginx/sites-available/crystalnexus.conf
  sudo nginx -t && sudo systemctl reload nginx
  sudo systemctl restart crystalnexus
  ```
  To change Basic-auth passwords instead, use `sudo htpasswd -B /etc/nginx/.htpasswd-crystalnexus USER`.
- The Windows Firewall instructions in the README do **not** apply on Linux; `ufw` is configured by cloud-init and `install.sh` instead.
- Keep the OS patched (`apt upgrade`) and limit who has SSH access (key only; password login is disabled by the cloud-init template).
