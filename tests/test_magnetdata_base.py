"""Tests for magnetdata_base's shared pint UnitRegistry singleton."""

import threading

from python_magnetrun import magnetdata_base


class TestMakeUreg:
    def test_concurrent_first_calls_share_one_registry(self, monkeypatch):
        # Cold start: several threads (e.g. parallel Dash callbacks) ask for
        # the registry at the same time — they must all get the same one, or
        # comparing their Units raises "different registries".
        monkeypatch.setattr(magnetdata_base, "_ureg", None)
        n_threads = 4
        barrier = threading.Barrier(n_threads)
        registries = []

        def worker():
            barrier.wait()
            registries.append(magnetdata_base._make_ureg())

        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(registries) == n_threads
        assert len({id(r) for r in registries}) == 1
        assert magnetdata_base._make_ureg() is registries[0]
