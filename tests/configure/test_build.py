import io
from pathlib import Path
from unittest import mock

from hector_ml.configure import HectorConfig
from hector_ml.configure.base import SourceFile


def test_build_ninja_file(tmp_path):
    with mock.patch(
        "importlib.import_module",
        autospec=True,
        return_value=mock.NonCallableMock(
            get_sources=mock.Mock(
                return_value=[
                    SourceFile(tmp_path / "foo.c"),
                    SourceFile(tmp_path / "bar.c"),
                ]
            ),
            get_reconfigure_inputs=mock.Mock(return_value=[]),
        ),
    ), mock.patch("hector_ml.configure.Writer", autospec=True) as writer_cls_mock:
        (tmp_path / "hector").mkdir()
        HectorConfig(
            tmp_path,
            tmp_path,
            tmp_path / "hector",
            Path("/usr/lib"),
            "dummy",
            "foo",
            [0],
            None,
        ).build_ninja_file()

        writer_mock = writer_cls_mock(None)
        assert writer_mock.build.mock_calls == [
            mock.call(
                ["foo.ll"],
                "analyze_source",
                ["../foo.c"],
                variables={"extra_flags": []},
            ),
            mock.call(
                ["bar.ll"],
                "analyze_source",
                ["../bar.c"],
                variables={"extra_flags": []},
            ),
            mock.call(["foo-combine_ll.ll"], "combine_ll", ["foo.ll", "bar.ll"]),
            mock.call(
                ["foo-allow_optimization.ll"],
                "allow_optimization",
                ["foo-combine_ll.ll"],
            ),
            mock.call(
                ["foo-opt_indirectbr.ll"],
                "opt_indirectbr",
                ["foo-allow_optimization.ll"],
            ),
            mock.call(
                ["foo-opt_globaldce.ll"], "opt_globaldce", ["foo-opt_indirectbr.ll"]
            ),
            mock.call(
                ["hector-0.json"],
                "opt_llap",
                ["foo-opt_globaldce.ll"],
                implicit=["$llap_path/LLVM_HECTOR_0.so"],
                variables={"llap_plugin": 0},
            ),
            mock.call(["hector.ninja"], "reconfigure_hector", []),
        ]


def test_hector_config_round_trip():
    hector_config = HectorConfig(
        source_dir="/foo",
        build_dir="/bar",
        hector_dir="/baz",
        llap_lib_dir="/qux",
        build_system="dummy",
        target="default",
        cwes=[0],
    )
    f = io.StringIO()
    hector_config._to_ninja_file(f)
    f.seek(0)
    new_config = HectorConfig._from_ninja_file(f)
    assert new_config == hector_config
