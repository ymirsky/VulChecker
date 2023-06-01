import collections
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Generator, Iterable, List

import click

from hector_ml.configure import HectorConfig, detect_build_system, get_build_system

ALL_CWES = [
    121,
    # 122,  # same pipeline as 121
    190,
    # 191,  # same pipeline as 190
    415,
    416,
]

logger = logging.getLogger(__name__)


LABELS_NAME = "hector_labels.json"


def find_labels(base_dir: Path) -> Path:
    work = collections.deque([base_dir])
    while work:
        current_dir = work.pop()
        labels_file = current_dir / LABELS_NAME
        if labels_file.exists():
            return labels_file
        for entry in current_dir.iterdir():
            if entry.is_dir():
                work.append(entry)
    raise FileNotFoundError(base_dir / LABELS_NAME)


def gather_builds(
    examples_root: Path, llap_lib_dir: Path
) -> Generator[HectorConfig, None, None]:
    for example_dir in examples_root.iterdir():
        if not example_dir.is_dir():
            continue
        try:
            labels_file = find_labels(example_dir)
        except FileNotFoundError:
            logger.warning("%s has no labels, skipping...", example_dir)
            continue

        source_dir = labels_file.parent
        build_system = detect_build_system(source_dir)
        build_system_module = get_build_system(build_system)
        yield HectorConfig(
            source_dir=source_dir,
            build_dir=source_dir,
            hector_dir=source_dir,
            llap_lib_dir=llap_lib_dir,
            build_system=build_system,
            target=build_system_module.infer_target(source_dir, source_dir),
            cwes=ALL_CWES,
            labels=labels_file,
        )


def do_builds(builds: Iterable[HectorConfig]) -> List[HectorConfig]:
    """Build and extract graphs from all sources.

    This is better than

    ::

        for build in builds:
            build.make()

    because it runs the compilers concurrently.

    """
    processes = []
    for build in builds:
        build.configure_if_needed()
        processes.append(
            (
                build,
                subprocess.Popen(["ninja", "-f", "hector.ninja"], cwd=build.hector_dir),
            )
        )
    successes = []
    failures = []
    for build, process in processes:
        result = process.wait()
        if result:
            failures.append(build)
        else:
            successes.append(build)
    # Extra loop to ensure failure messages appear after any other output from Ninja.
    for build in failures:
        logger.warning("Failed to build %s.", build.source_dir)
    return successes


def copy_if_newer(src: Path, dst: Path):
    src_mtime = src.stat().st_mtime
    try:
        dst_mtime = dst.stat().st_mtime
    except FileNotFoundError:
        dst_mtime = 0.0
    if src_mtime > dst_mtime:
        shutil.copy2(src, dst)


@click.command()
@click.option(
    "--llap-lib-dir",
    type=click.Path(exists=True, file_okay=False),
    default="/usr/local/lib/llap",
    help="Path where LLAP shared libraries live.",
)
@click.argument("examples_root", type=click.Path(exists=True, file_okay=False))
@click.argument("output_dir", type=click.Path(file_okay=False, writable=True))
def main(llap_lib_dir: str, examples_root: str, output_dir: str):
    llap_lib_dir = Path(llap_lib_dir).resolve()
    examples_root = Path(examples_root).resolve()
    output_dir = Path(output_dir)

    builds = gather_builds(examples_root, llap_lib_dir)
    succeeded_builds = do_builds(builds)
    for build in succeeded_builds:
        for cwe in ALL_CWES:
            (output_dir / f"CWE{cwe}").mkdir(parents=True, exist_ok=True)
            copy_if_newer(
                build.hector_dir / f"hector-{cwe}.json",
                output_dir / f"CWE{cwe}" / f"{build.source_dir.name}.json",
            )
