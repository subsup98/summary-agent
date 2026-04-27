from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from src.shared.io import ensure_directory

LIBREOFFICE_CANDIDATES = [
    Path("C:/Program Files/LibreOffice/program/soffice.com"),
    Path("C:/Program Files/LibreOffice/program/soffice.exe"),
    Path("C:/Program Files (x86)/LibreOffice/program/soffice.com"),
    Path("C:/Program Files (x86)/LibreOffice/program/soffice.exe"),
]
LIBREOFFICE_CONVERTIBLE_EXTENSIONS = {".doc", ".docx", ".hwp", ".hwpx"}


def find_libreoffice() -> Path | None:
    for candidate in LIBREOFFICE_CANDIDATES:
        if candidate.exists():
            return candidate
    found = shutil.which("soffice")
    return Path(found) if found else None


def iter_libreoffice_candidates(preferred: Path | None = None) -> list[Path]:
    candidates: list[Path] = []
    seen: set[str] = set()

    def add(candidate: Path | None) -> None:
        if candidate is None:
            return
        key = str(candidate).lower()
        if key in seen or not candidate.exists():
            return
        seen.add(key)
        candidates.append(candidate)

    add(preferred)
    if preferred is not None:
        sibling_com = preferred.with_suffix(".com")
        sibling_exe = preferred.with_suffix(".exe")
        if preferred.suffix.lower() == ".exe":
            add(sibling_com)
            add(sibling_exe)
        elif preferred.suffix.lower() == ".com":
            add(sibling_com)
            add(sibling_exe)
    for candidate in LIBREOFFICE_CANDIDATES:
        add(candidate)
    found = shutil.which("soffice")
    add(Path(found) if found else None)
    return candidates


def get_libreoffice_user_profile() -> Path | None:
    candidates = [
        Path.home() / "AppData" / "Roaming" / "LibreOffice" / "4" / "user",
        Path("C:/Users/yongseop.im/AppData/Roaming/LibreOffice/4/user"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def libreoffice_has_h2orestart() -> bool:
    profile = get_libreoffice_user_profile()
    if profile is None:
        return False
    pmap_path = profile / "uno_packages" / "cache" / "uno_packages.pmap"
    if not pmap_path.exists():
        return False
    try:
        return "ebandal.libreoffice.H2Orestart" in pmap_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False


def build_libreoffice_env(*, source_path: Path, profile_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    if source_path.suffix.lower() not in {".hwp", ".hwpx"} or not libreoffice_has_h2orestart():
        return env
    home_dir = profile_root / "home"
    temp_dir = profile_root / "tmp"
    ensure_directory(home_dir)
    ensure_directory(temp_dir)
    java_tool_options = str(env.get("JAVA_TOOL_OPTIONS") or "").strip()
    user_home_option = f"-Duser.home={home_dir}"
    env["JAVA_TOOL_OPTIONS"] = f"{java_tool_options} {user_home_option}".strip() if java_tool_options else user_home_option
    env["USERPROFILE"] = str(home_dir)
    env["HOME"] = str(home_dir)
    env["TMP"] = str(temp_dir)
    env["TEMP"] = str(temp_dir)
    return env


def seed_libreoffice_profile(*, source_path: Path, profile_root: Path) -> None:
    if source_path.suffix.lower() not in {".hwp", ".hwpx"} or not libreoffice_has_h2orestart():
        return
    user_profile = get_libreoffice_user_profile()
    if user_profile is None:
        return
    src_uno_packages = user_profile / "uno_packages"
    if not src_uno_packages.exists():
        return
    dest_uno_packages = profile_root / "user" / "uno_packages"
    ensure_directory(dest_uno_packages.parent)
    shutil.copytree(src_uno_packages, dest_uno_packages, dirs_exist_ok=True)


def _truncate_debug_text(text: str, *, limit: int = 4000) -> str:
    normalized = str(text or "").strip()
    if len(normalized) <= limit:
        return normalized
    return normalized[:limit] + "...<truncated>"


def _copy_converted_pdf(*, staging_output_dir: Path, source_path: Path, dest_path: Path) -> bool:
    converted = staging_output_dir / f"{source_path.stem}.pdf"
    if not converted.exists():
        # NOTE: HWP -> PDF via LibreOffice/H2Orestart has been observed to emit a
        # differently named PDF in the staging directory even when conversion
        # succeeds. Keep this fallback search in place unless we can prove the
        # output filename is stable for Korean filenames across environments.
        pdf_candidates = sorted(
            staging_output_dir.glob("*.pdf"),
            key=lambda item: item.stat().st_mtime if item.exists() else 0.0,
            reverse=True,
        )
        if len(pdf_candidates) == 1:
            converted = pdf_candidates[0]
        else:
            normalized_stem = source_path.stem.strip().lower()
            matching_candidates = [
                candidate
                for candidate in pdf_candidates
                if candidate.stem.strip().lower() == normalized_stem
            ]
            if len(matching_candidates) == 1:
                converted = matching_candidates[0]
            else:
                return False
    ensure_directory(dest_path.parent)
    if dest_path.exists():
        try:
            dest_path.unlink()
        except OSError:
            pass
    shutil.move(str(converted), str(dest_path))
    return dest_path.exists()


def convert_office_source_to_pdf_with_diagnostics(source_path: Path, dest_path: Path) -> dict[str, object]:
    preferred = find_libreoffice()
    soffice_candidates = iter_libreoffice_candidates(preferred)
    diagnostics: dict[str, object] = {
        "source_path": source_path.as_posix(),
        "dest_path": dest_path.as_posix(),
        "preferred_soffice": str(preferred) if preferred else None,
        "candidate_soffice_paths": [str(candidate) for candidate in soffice_candidates],
        "used_h2orestart_profile": source_path.suffix.lower() in {".hwp", ".hwpx"} and libreoffice_has_h2orestart(),
        "attempts": [],
        "succeeded": False,
        "failure_reason": None,
    }
    if not soffice_candidates:
        diagnostics["failure_reason"] = "no_soffice_candidate"
        return diagnostics
    stage_root = Path(tempfile.mkdtemp(prefix="lo-cvt-"))
    staging_output_dir = stage_root / "o"
    ensure_directory(staging_output_dir)
    diagnostics["stage_root"] = str(stage_root)
    diagnostics["staging_output_dir"] = str(staging_output_dir)
    try:
        ensure_directory(dest_path.parent)
        if dest_path.exists():
            try:
                dest_path.unlink()
            except OSError:
                pass
        for soffice in soffice_candidates:
            profile_dir = stage_root / f"p-{len(diagnostics['attempts']) + 1}"
            ensure_directory(profile_dir)
            attempt: dict[str, object] = {
                "soffice_path": str(soffice),
                "profile_dir": str(profile_dir),
                "profile_uri": None,
                "returncode": None,
                "stdout": "",
                "stderr": "",
                "converted_exists": False,
                "dest_exists": False,
            }
            try:
                seed_libreoffice_profile(source_path=source_path, profile_root=profile_dir)
                profile_uri = profile_dir.resolve().as_uri()
                attempt["profile_uri"] = profile_uri
                env = build_libreoffice_env(source_path=source_path, profile_root=profile_dir)
                result = subprocess.run(
                    [
                        str(soffice),
                        "--headless",
                        "--nologo",
                        "--nodefault",
                        "--nolockcheck",
                        "--norestore",
                        f"-env:UserInstallation={profile_uri}",
                        "--convert-to",
                        "pdf",
                        "--outdir",
                        str(staging_output_dir),
                        str(source_path),
                    ],
                    capture_output=True,
                    timeout=120,
                    cwd=str(soffice.parent),
                    env=env,
                )
                attempt["returncode"] = result.returncode
                attempt["stdout"] = _truncate_debug_text(result.stdout.decode("utf-8", errors="ignore"))
                attempt["stderr"] = _truncate_debug_text(result.stderr.decode("utf-8", errors="ignore"))
            finally:
                shutil.rmtree(profile_dir, ignore_errors=True)
            converted = staging_output_dir / f"{source_path.stem}.pdf"
            attempt["converted_exists"] = converted.exists()
            attempt["dest_exists"] = _copy_converted_pdf(
                staging_output_dir=staging_output_dir,
                source_path=source_path,
                dest_path=dest_path,
            )
            diagnostics["attempts"].append(attempt)
            if dest_path.exists():
                # NOTE: Treat an actual PDF artifact as success even if soffice
                # returns a non-zero exit code. With H2Orestart-enabled HWP
                # conversion we have seen PDFs generated successfully while the
                # process still exits with 3221226356.
                diagnostics["succeeded"] = True
                if result.returncode != 0:
                    diagnostics["failure_reason"] = None
                return diagnostics
            if result.returncode != 0:
                continue
        diagnostics["failure_reason"] = "conversion_command_failed_or_no_output"
        return diagnostics
    except Exception as error:
        diagnostics["failure_reason"] = "exception"
        diagnostics["exception"] = repr(error)
        return diagnostics
    finally:
        shutil.rmtree(stage_root, ignore_errors=True)


def convert_office_source_to_pdf(source_path: Path, dest_path: Path) -> bool:
    return bool(convert_office_source_to_pdf_with_diagnostics(source_path, dest_path).get("succeeded"))
