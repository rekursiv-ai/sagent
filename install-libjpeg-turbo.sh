#!/usr/bin/env bash
# Ensure a libturbojpeg new enough for pyturbojpeg 2.x is on the loader path.
#
# pyturbojpeg 2.x binds the TurboJPEG 3 API and raises "PyTurboJPEG 2.0
# requires libjpeg-turbo 3.0 or later" against anything older.
#
# Prefers apt: when the distro's candidate is >= 3.0 this installs it and
# stops, so a newer release needs no change here and the box carries no
# unmanaged payload. Noble's newest is 2.1.5, which is why the fallback
# exists at all -- it fetches the upstream .deb.
#
# That .deb unpacks to /opt/libjpeg-turbo because the prefix is compiled in
# (every binary carries RPATH=[/opt/libjpeg-turbo/lib64]), so relocating the
# payload breaks it. Instead symlinks publish the library on the local linker
# paths, ahead of the distro copy under /usr/lib.
#
# Run AFTER apt. Idempotent. Undo:
#   multiarch="$(uname -m)-linux-gnu"
#   sudo rm -f /usr/local/lib/libturbojpeg.so.0 \
#     "/usr/local/lib/${multiarch}/libturbojpeg.so.0"
#   sudo dpkg -r libjpeg-turbo-official && sudo ldconfig

set -euo pipefail

# Pinned: an unpinned release would change what CI resolves with no diff here.
VERSION="${LIBJPEG_TURBO_VERSION:-3.2.0}"
MINIMUM=3.0.0

SUDO=""
[ "$(id -u)" -eq 0 ] || SUDO="sudo"
multiarch_local="/usr/local/lib/$(uname -m)-linux-gnu"
target=/opt/libjpeg-turbo/lib64/libturbojpeg.so.0

remove_managed_link() {
  local link="$1"
  if [ -L "$link" ] && [ "$(readlink "$link")" = "$target" ]; then
    $SUDO rm -f -- "$link"
  fi
}

candidate="$(apt-cache policy libturbojpeg 2>/dev/null |
  awk '/Candidate:/{if (!f++) c=$2} END{print c}')"
# Strip the epoch before comparing. Ubuntu ships `1:2.1.5-2ubuntu2`, and an
# epoch outranks everything after it -- `dpkg --compare-versions 1:2.1.5 ge
# 3.0.0` is TRUE, which silently accepted noble's 2.1.5 and left tj3Init
# missing. Putting the epoch in the threshold instead would wrongly reject an
# unepoched upstream 3.0.4, so the comparison is on upstream version alone.
if [ -n "${candidate}" ] && [ "${candidate}" != "(none)" ] &&
  dpkg --compare-versions "${candidate#*:}" ge "${MINIMUM}"; then
  echo "libturbojpeg ${candidate} from apt is >= ${MINIMUM}; using it."
  $SUDO apt-get install -y --no-install-recommends libturbojpeg
  remove_managed_link /usr/local/lib/libturbojpeg.so.0
  remove_managed_link "$multiarch_local/libturbojpeg.so.0"
  if dpkg-query -W -f='${db:Status-Abbrev}' libjpeg-turbo-official \
    2>/dev/null | grep -q '^ii '; then
    $SUDO dpkg -r libjpeg-turbo-official
  fi
  $SUDO ldconfig
  exit 0
fi

echo "apt libturbojpeg (${candidate:-none}) < ${MINIMUM}; installing ${VERSION} .deb."

deb="libjpeg-turbo-official_${VERSION}_$(dpkg --print-architecture).deb"
tmp="$(mktemp -d)"
trap 'rm -rf "${tmp}"' EXIT

# -f: without it a 404 writes an HTML error page and dpkg reports a confusing
# archive-format error instead of a download failure.
curl -fsSL -o "${tmp}/${deb}" \
  "https://github.com/libjpeg-turbo/libjpeg-turbo/releases/download/${VERSION}/${deb}"
$SUDO dpkg -i "${tmp}/${deb}"

# Target the deb's own unversioned symlink: 3.0.4 ships .so.0.3.0 and 3.2.0
# ships .so.0.5.0, so a versioned target dangles on the next bump. Debian's
# multiarch linker path puts /usr/local/lib/<triplet> before /lib/<triplet>,
# while generic /usr/local/lib may be processed later (observed on AArch64).
# Publish both: the multiarch link wins there and the generic link preserves
# the installer contract on hosts without the architecture-specific path.
$SUDO install -d -m 0755 "$multiarch_local"
$SUDO ln -sf "$target" /usr/local/lib/libturbojpeg.so.0
$SUDO ln -sf "$target" "$multiarch_local/libturbojpeg.so.0"
$SUDO ldconfig
