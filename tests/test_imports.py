from rag.common import ROOT


def test_package_root_exists():
    assert ROOT.exists()
