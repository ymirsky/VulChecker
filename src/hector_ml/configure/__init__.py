import abc
import importlib
import json
import pathlib
import shlex
import subprocess
import sys
import urllib.parse
from typing import ClassVar, Iterable, Optional, Sequence, TextIO

import attr
import click

from hector_ml.configure.base import SourceFile
from hector_ml.configure.ninja_syntax import Writer
from hector_ml.paths import relative_to_with_parents

KNOWN_SOURCE_TYPES = [
    "cmake",
    "ninja",
]

try:
    shlex_join = shlex.join
except AttributeError:
    # Python <3.8
    def shlex_join(args):
        return " ".join(shlex.quote(arg) for arg in args)


def detect_build_system(path: pathlib.Path) -> Optional[str]:
    """Guess the build system used by a directory."""
    if (path / "CMakeLists.txt").exists():
        return "cmake"
    elif (path / "build.ninja").exists():
        return "ninja"
    else:
        return None


# Really, this should be a Protocol, not an ABC, but that's not until Python 3.8.
class BuildSystem(abc.ABC):
    """API implemented by a build system.

    The build systems are actually modules, not classes.
    This class represents a protocol implemented by the build system module.
    Even though these are represented as abstract methods,
    they're actually just ordinary functions.

    """

    @abc.abstractmethod
    def get_targets(
        self, build_dir: pathlib.Path, hector_dir: pathlib.Path
    ) -> Iterable[str]:
        """Get all available build targets."""

    @abc.abstractmethod
    def get_sources(
        self, build_dir: pathlib.Path, hector_dir: pathlib.Path, target: str
    ) -> Iterable[SourceFile]:
        """Get source information to build a specific target."""

    @abc.abstractmethod
    def get_reconfigure_inputs(
        self, build_dir: pathlib.Path, hector_dir: pathlib.Path, target: str
    ) -> Iterable[pathlib.Path]:
        """Get paths to files that should cause a re-configure."""

    @abc.abstractmethod
    def infer_target(
        self, build_dir: pathlib.Path, hector_dir: pathlib.Path
    ) -> Optional[str]:
        """Guess what target should be built, based on labels."""


def get_build_system(name: str) -> BuildSystem:
    """Import the named build system."""
    return importlib.import_module(f"hector_ml.configure.{name}")


@click.command()
@click.option(
    "--build-dir",
    metavar="BUILD_DIR",
    default=".",
    show_default=True,
    type=click.Path(file_okay=False, exists=True),
    help="Directory where build system files are located.",
)
@click.option(
    "--source-dir",
    metavar="SOURCE_DIR",
    type=click.Path(file_okay=False, exists=True),
    help="Directory where project source is located. [default: BUILD_DIR]",
)
@click.option(
    "--hector-dir",
    metavar="HECTOR_DIR",
    type=click.Path(file_okay=False, writable=True),
    help=(
        "Directory where hector.ninja will be written. "
        "[default: BUILD_DIR/hector_build]"
    ),
)
@click.option(
    "--labels",
    metavar="LABELS",
    type=click.Path(dir_okay=False, exists=True),
    help="JSON labels file for running LLAP.",
)
@click.option(
    "--llap-lib-dir",
    metavar="LLAP_LIB_DIR",
    type=click.Path(file_okay=False, exists=True),
    default="/usr/local/lib",
    help="Path where LLAP module shared libraries are located.",
)
@click.argument("build_system", type=click.Choice(KNOWN_SOURCE_TYPES), required=True)
@click.argument("target")
@click.argument("cwes", metavar="CWE", nargs=-1, type=int)
def main(
    build_dir: str,
    source_dir: Optional[str],
    hector_dir: Optional[str],
    labels: Optional[str],
    llap_lib_dir: str,
    build_system: str,
    target: Optional[str],
    cwes: Sequence[int],
):
    """Configure a codebase to be analyzed by HECTOR.

    If you pass the empty string (i.e. "") as TARGET,
    all known targets will be listed
    and the command will exit without doing any work.

    """
    build_dir = pathlib.Path(build_dir).resolve()
    if source_dir is None:
        source_dir = build_dir
    else:
        source_dir = pathlib.Path(source_dir).resolve()
    if hector_dir is None:
        hector_dir = build_dir / "hector_build"
    else:
        hector_dir = pathlib.Path(hector_dir).resolve()
    hector_dir.mkdir(parents=True, exist_ok=True)
    if labels is not None:
        labels = pathlib.Path(labels).resolve()

    llap_lib_dir = pathlib.Path(llap_lib_dir).resolve()

    if not target:
        _list_targets(build_dir, hector_dir, build_system)
    else:
        HectorConfig(
            source_dir,
            build_dir,
            hector_dir,
            llap_lib_dir,
            build_system,
            target,
            cwes,
            labels,
        ).build_ninja_file()


def _list_targets(build_dir, hector_dir, build_system):
    build_system_module = get_build_system(build_system)
    for target in build_system_module.get_targets(build_dir, hector_dir):
        print(target)


ALLOW_OPTIMIZATION_SED = "/^attributes #[0-9]+ =/ { s/ noinline / /; s/ optnone / / }"


@attr.s(auto_attribs=True, slots=True, frozen=True)
class HectorConfig:
    """Options for setting up the LLAP pipeline.

    For now,
    you should always set ``source_dir``, ``build_dir``, and ``hector_dir``
    to the top level directory containing the code.
    This ensures that the outputs have the appropriate relative paths until
    `issue #35 <https://github.gatech.edu/HECTOR/hector_ml/issues/35>`_
    is addressed.

    :param source_dir: The top-level directory of the source code of the program.
    :param build_dir:
        The top-level directory of the build system of the program.
        In most cases,
        this will be the same as ``source_dir``.
    :param hector_dir:
        The directory where the ``hector.ninja`` file will be written.
        The LLAP intermediate and output files
        (i.e. ``*.ll`` and ``hector-$cwe.json``)
        will also be created under this directory.
    :param llap_lib_dir:
        The directory containing the LLAP plugin libraries
        (i.e. ``LLVM_HECTOR_$cwe.so``).
    :param build_system:
        The build system of the code under analysis.
        If you don't know the build system from out-of-band means,
        :func:`detect_build_system` can be used to infer the correct value.
    :param target:
        The build system target to build.
        This is the thing that Make refers to as a "target".
        Right now,
        the target is required to refer to a single executable,
        so ``all``, for example, would be inappropriate.
        If you have labels,
        you can use :func:`BuildSystem.infer_target` to infer an appropriate value.
    :param cwes:
        CWE pipelines to configure.
        Note that LLAP has several "combined" pipelines that are used for several CWEs.
        In this parameter,
        you must provide only the first one for each combined pipeline.
        For example, provide ``[190]`` to build information for CWEs 190 and 191.
    :param labels: Path to an LLAP labels file for this code.

    """

    source_dir: pathlib.Path = attr.ib(converter=pathlib.Path)
    build_dir: pathlib.Path = attr.ib(converter=pathlib.Path)
    hector_dir: pathlib.Path = attr.ib(converter=pathlib.Path)
    llap_lib_dir: pathlib.Path = attr.ib(converter=pathlib.Path)
    build_system: str
    target: str
    cwes: Sequence[int] = attr.ib(converter=tuple)
    labels: Optional[pathlib.Path] = attr.ib(
        default=None, converter=attr.converters.optional(pathlib.Path)
    )

    _NINJA_PREFIX: ClassVar[str] = "# CONFIG="

    def _to_ninja_file(self, writer: TextIO):
        """Serialize this config to the hector.ninja file."""
        hector_configuration = json.dumps(attr.asdict(self), default=str)
        writer.write(f"{self._NINJA_PREFIX}{hector_configuration}\n")

    @classmethod
    def _from_ninja_file(cls, reader: TextIO) -> Optional["HectorConfig"]:
        """Load this config from a hector.ninja file."""
        for line in reader:
            if line.startswith(cls._NINJA_PREFIX):
                return cls(**json.loads(line[len(cls._NINJA_PREFIX) :]))
        return None

    def _reconfigure_args(self) -> Iterable[str]:
        """Args to re-invoke this configure through hector configure."""
        yield from [
            str(pathlib.Path(sys.executable).parent / "hector"),
            "configure",
            "--source-dir",
            str(self.source_dir),
            "--build-dir",
            str(self.build_dir),
            "--hector-dir",
            str(self.hector_dir),
            "--llap-lib-dir",
            str(self.llap_lib_dir),
        ]
        if self.labels:
            yield from ["--labels", str(self.labels)]
        yield from [self.build_system, self.target]
        yield from (str(cwe) for cwe in self.cwes)

    def configure_if_needed(self):
        """Build the hector.ninja file if needed.

        A rebuild is needed if hector.ninja does not exist,
        or if the configuration stored therein does not match.

        """
        try:
            with open(self.hector_dir / "hector.ninja") as f:
                existing_config = HectorConfig._from_ninja_file(f)
            need_configure = self != existing_config
        except FileNotFoundError:
            need_configure = True

        if need_configure:
            self.build_ninja_file()

    def make(self):
        """Run the pipeline.

        This calls :meth:`configure_if_needed`,
        so you do not need to call it first.

        """
        self.configure_if_needed()
        subprocess.run(["ninja", "-f", "hector.ninja"], check=True, cwd=self.hector_dir)

    def _relative(self, path: pathlib.Path) -> pathlib.Path:
        return relative_to_with_parents(path, self.hector_dir)

    def build_ninja_file(self):
        """Build the hector.ninja file (unconditionally)."""
        build_system_module = get_build_system(self.build_system)
        if self.labels:
            relative_labels = self._relative(self.labels)
        else:
            relative_labels = None

        with click.open_file(
            self.hector_dir / "hector.ninja", "wt", atomic=True
        ) as ninja_file:
            writer = Writer(ninja_file)
            writer.comment("Generated by HECTOR")
            self._to_ninja_file(ninja_file)
            writer.variable("llap_path", str(self.llap_lib_dir))
            writer.variable(
                "labels", f"-labelFilename={relative_labels}" if self.labels else ""
            )
            writer.rule(
                "analyze_source",
                "clang -O0 -g -S -MD -MF $out.d $extra_flags -emit-llvm -o $out $in",
                depfile="$out.d",
            )
            writer.rule("combine_ll", "llvm-link -S -o $out $in")
            writer.rule(
                "allow_optimization",
                f"sed -E '{ALLOW_OPTIMIZATION_SED}' $in >$out",
            )
            writer.rule(
                "opt_indirectbr",
                (
                    "opt --indirectbr-expand --inline-threshold=10000 "
                    "--inline -S -o $out $in"
                ),
            )
            writer.rule(
                "opt_globaldce",
                (
                    "opt --internalize-public-api-list='main' "
                    "--internalize --globaldce -S -o $out $in"
                ),
            )
            writer.rule(
                "opt_llap",
                (
                    "opt -load $llap_path/LLVM_HECTOR_$llap_plugin.so "
                    "-HECTOR_$llap_plugin $labels -outputFilename $out $in >/dev/null"
                ),
            )
            writer.rule(
                "reconfigure_hector",
                shlex_join(self._reconfigure_args()),
                generator=True,
            )

            source_files: Iterable[SourceFile] = build_system_module.get_sources(
                self.build_dir, self.hector_dir, self.target
            )

            all_ll_files = []
            for source_file in source_files:
                target_file = str(
                    source_file.path.relative_to(self.source_dir).with_suffix(".ll")
                )
                writer.build(
                    [target_file],
                    "analyze_source",
                    [str(self._relative(source_file.path))],
                    variables={
                        "extra_flags": [shlex.quote(x) for x in source_file.extra_flags]
                    },
                )
                all_ll_files.append(target_file)

            if not all_ll_files:
                raise ValueError("No source files were identified for that target.")

            previous_target = all_ll_files
            safe_target = urllib.parse.quote(self.target, safe="")
            for rule in [
                "combine_ll",
                "allow_optimization",
                "opt_indirectbr",
                "opt_globaldce",
            ]:
                this_target = [f"{safe_target}-{rule}.ll"]
                writer.build(this_target, rule, previous_target)
                previous_target = this_target

            for cwe in self.cwes:
                llap_implicits = [f"$llap_path/LLVM_HECTOR_{cwe}.so"]
                if self.labels:
                    llap_implicits.append(str(relative_labels))
                writer.build(
                    [f"hector-{cwe}.json"],
                    "opt_llap",
                    previous_target,
                    implicit=llap_implicits,
                    variables={"llap_plugin": cwe},
                )

            writer.build(
                ["hector.ninja"],
                "reconfigure_hector",
                [
                    str(self._relative(p))
                    for p in build_system_module.get_reconfigure_inputs(
                        self.build_dir, self.hector_dir, self.target
                    )
                ],
            )
