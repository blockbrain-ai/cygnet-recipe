#!/usr/bin/env bash
# All downloaded source, objects and executables stay in this repo's .runtime/.
set -euo pipefail
if [[ "$(uname -s)" != Darwin || "$(uname -m)" != arm64 ]]; then
    printf 'The Metal backend requires macOS on Apple Silicon.\n' >&2
    exit 1
fi
project_root="$(cd "$(dirname "$0")/.." && pwd)"
revision="8086439a4cea94c71a5dfb8fe4ad1546aebd640f"
archive_sha256="1984103666eb25bd45110a40cba22b9d4286116f26e51bbc76f6f41dc86bc7b5"
runtime_dir="$project_root/.runtime"
source_dir="$runtime_dir/llama.cpp-$revision"
build_dir="$runtime_dir/metal-build"
jobs="${CYGNET_BUILD_JOBS:-2}"
if [[ ! "$jobs" =~ ^[1-9][0-9]*$ ]]; then
    printf 'CYGNET_BUILD_JOBS must be a positive integer.\n' >&2
    exit 1
fi
for program in cmake curl tar shasum; do
    command -v "$program" >/dev/null || { printf 'Missing build tool: %s\n' "$program" >&2; exit 1; }
done
mkdir -p "$runtime_dir"
unpack_dir=""
install_tmp=""
trap '[[ -z "$unpack_dir" ]] || rm -rf "$unpack_dir"; [[ -z "$install_tmp" ]] || rm -f "$install_tmp"' EXIT
if [[ ! -f "$source_dir/include/llama.h" ]]; then
    unpack_dir="$(mktemp -d "$runtime_dir/unpack.XXXXXX")"
    archive="$runtime_dir/llama.cpp-$revision.tar.gz"
    override="${CYGNET_LLAMA_SOURCE:-}"
    if [[ -n "$override" && -d "$override" && -e "$override/.git" ]]; then
        # git archive ignores uncommitted changes; never patch the caller's source.
        actual_revision="$(git -C "$override" rev-parse HEAD)"
        if [[ "$actual_revision" != "$revision" ]]; then
            printf 'Expected source revision %s, found %s\n' "$revision" "$actual_revision" >&2
            exit 1
        fi
        git -C "$override" archive "$revision" | tar -x -C "$unpack_dir"
    else
        if [[ -n "$override" ]]; then
            if [[ -f "$override" ]]; then
                archive="$override"
            else
                printf 'CYGNET_LLAMA_SOURCE needs a pinned Git checkout or archive.\n' >&2
                exit 1
            fi
        elif [[ ! -f "$archive" ]]; then
            curl --fail --location --retry 3 --output "$archive.part" \
                "https://github.com/ggml-org/llama.cpp/archive/$revision.tar.gz"
            mv "$archive.part" "$archive"
        fi
        actual_hash="$(shasum -a 256 "$archive" | awk '{print $1}')"
        if [[ "$actual_hash" != "$archive_sha256" ]]; then
            printf 'Pinned llama.cpp archive SHA-256 mismatch: %s\n' "$archive" >&2
            exit 1
        fi
        tar -xzf "$archive" -C "$unpack_dir" --strip-components=1
    fi
    printf '%s\n' "$revision" > "$unpack_dir/.cygnet-revision"
    if [[ -e "$source_dir" ]]; then
        printf 'Incomplete source directory exists; inspect or remove %s before retrying.\n' "$source_dir" >&2
        exit 1
    fi
    mv "$unpack_dir" "$source_dir"
    unpack_dir=""
fi
cmake -S "$project_root/metal" -B "$build_dir" \
    -DCMAKE_BUILD_TYPE=Release -DCYGNET_LLAMA_SOURCE="$source_dir"
cmake --build "$build_dir" --target cygnet-metal-worker --parallel "$jobs"
mkdir -p "$runtime_dir/build/bin"
install_tmp="$(mktemp "$runtime_dir/build/bin/.cygnet-metal-worker.XXXXXX")"
cp "$build_dir/staging/cygnet-metal-worker" "$install_tmp"
chmod 755 "$install_tmp"
mv -f "$install_tmp" "$runtime_dir/build/bin/cygnet-metal-worker"
install_tmp=""
printf '\nRuntime built: %s\n' "$runtime_dir/build/bin/cygnet-metal-worker"
