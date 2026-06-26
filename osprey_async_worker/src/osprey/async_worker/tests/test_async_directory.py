"""Regression tests for the threaded continuous-consumption etcd watcher.

These tests cover the production incident where a worker's smite_shortlist instance
set got wiped to empty and never repopulated (select() raised "No service" forever
until the worker was restarted). The root cause was that the watch loop consumed
``continue_watching()`` one event at a time, recreating the generator per event,
which reset its resume index / dedup mux and defeated its built-in recovery.

The fix iterates ``continue_watching()`` continuously on a dedicated daemon thread,
forwarding each event to the asyncio loop where it is applied single-threaded.

Covered here:
  * recovery-after-empty-full-sync — an empty FullSyncRecursive wipes the set (no
    guard), and the SAME continuous generator repopulates it via incremental upserts;
  * stop() teardown — stop() sets the stop event and cancels the consumer task, and
    the daemon watch threads unwind on their next poll (they are not joined);
  * the same fix applied to _AsyncHashRing (SCALAR routing).
"""

import asyncio
import json
import queue
import threading
from types import SimpleNamespace

import pytest
from osprey.worker.lib.discovery.exceptions import ServiceUnavailable
from osprey.worker.lib.discovery.service import Service
from osprey.worker.lib.etcd import (
    FullSyncOne,
    FullSyncOneNoKey,
    FullSyncRecursive,
    IncrementalSyncDelete,
    IncrementalSyncUpsert,
)

from osprey.async_worker.lib.discovery.async_directory import AsyncServiceWatcher, _AsyncHashRing

# Sentinel pushed into a fake watcher's inbox to make continue_watching() return,
# simulating the watcher's next poll/timeout so the daemon thread can unwind.
_STOP = object()


class FakeWatcher:
    """A controllable stand-in for an etcd watcher.

    ``begin_watching()`` returns a preset initial sync event. ``continue_watching()``
    is an infinite generator that yields whatever the test pushes into its inbox and
    only returns when ``_STOP`` is pushed (blocking on the inbox in between, like the
    real watcher blocking on etcd).
    """

    def __init__(self, initial_event):
        self.initial_event = initial_event
        self.inbox: "queue.Queue" = queue.Queue()
        self.begin_count = 0
        self.continue_count = 0

    def begin_watching(self):
        self.begin_count += 1
        return self.initial_event

    def push(self, event):
        self.inbox.put(event)

    def close(self):
        """Release a blocked continue_watching() so the daemon thread can unwind."""
        self.inbox.put(_STOP)

    def continue_watching(self):
        self.continue_count += 1
        while True:
            item = self.inbox.get()
            if item is _STOP:
                return
            yield item


class FakeEtcdClient:
    """Serves preconfigured FakeWatchers, routed by the ``recursive`` flag.

    AsyncServiceWatcher requests a recursive watcher (instances); _AsyncHashRing
    requests a non-recursive (scalar) watcher (ring). Extra requests beyond the
    provided pools get a benign idle watcher.
    """

    def __init__(self, recursive_watchers=None, scalar_watchers=None):
        self._recursive = list(recursive_watchers or [])
        self._scalar = list(scalar_watchers or [])
        self.calls = []  # list of (key, recursive)
        self.all_watchers = []

    def get_watcher(self, key, recursive=False, _use_mux=True):
        self.calls.append((key, recursive))
        pool = self._recursive if recursive else self._scalar
        if pool:
            watcher = pool.pop(0)
        else:
            default = FullSyncRecursive(items=[]) if recursive else FullSyncOneNoKey(key=key)
            watcher = FakeWatcher(initial_event=default)
        self.all_watchers.append(watcher)
        return watcher


# --- helpers -----------------------------------------------------------------


def _svc(host: str, port: int) -> Service:
    return Service(name='smite_shortlist', address=host, port=port)


def _node(value: str) -> SimpleNamespace:
    return SimpleNamespace(key=f'/k/{value}', value=value)


def _full_sync(services) -> FullSyncRecursive:
    return FullSyncRecursive(items=[_node(s.serialize()) for s in services])


def _upsert(service: Service) -> IncrementalSyncUpsert:
    return IncrementalSyncUpsert(key=f'/k/{service.id}', value=service.serialize())


def _delete(service: Service) -> IncrementalSyncDelete:
    return IncrementalSyncDelete(key=f'/k/{service.id}', prev_value=service.serialize())


def _ring_full_sync(members) -> FullSyncOne:
    return FullSyncOne(key='/discovery/svc/ring', value=json.dumps(list(members)))


async def _wait_until(predicate, timeout: float = 2.0, interval: float = 0.01) -> bool:
    """Poll ``predicate`` (yielding to the loop so the consumer task can run)."""
    elapsed = 0.0
    while elapsed < timeout:
        if predicate():
            return True
        await asyncio.sleep(interval)
        elapsed += interval
    return predicate()


def _live_thread_names() -> set:
    """Names of currently-running daemon watch threads (no handle is stored on the
    watcher, so liveness is observed via the threading registry instead)."""
    return {t.name for t in threading.enumerate() if t.is_alive()}


def _watch_thread_names(watcher: AsyncServiceWatcher) -> tuple:
    return (
        f'async-service-watch:{watcher._service_name}',
        f'async-ring-watch:{watcher._ring._key}',
    )


async def _teardown_service(watcher: AsyncServiceWatcher, client: FakeEtcdClient) -> None:
    await watcher.stop()
    for fw in client.all_watchers:
        fw.close()
    names = _watch_thread_names(watcher)
    await _wait_until(lambda: not (set(names) & _live_thread_names()))


# --- AsyncServiceWatcher -----------------------------------------------------


@pytest.mark.asyncio
async def test_recovers_after_empty_full_sync():
    """An empty FullSyncRecursive wipes the set; the same generator repopulates it.

    This is the incident: a key-deleted / index_cleared full sync emptied the instance
    set. With continuous consumption the subsequent incremental upsert (yielded by the
    SAME generator) restores the instance, so select() recovers without a restart.
    """
    svc = _svc('shortlist-a', 1001)
    watcher = FakeWatcher(initial_event=_full_sync([svc]))
    client = FakeEtcdClient(recursive_watchers=[watcher])

    asw = AsyncServiceWatcher(client, '/discovery', 'smite_shortlist')
    await asw.ensure_initialized()
    assert set(asw._instances) == {svc.id}

    # Empty full sync wipes the set (no guard — the set self-repopulates).
    watcher.push(_full_sync([]))
    assert await _wait_until(lambda: len(asw._instances) == 0)
    with pytest.raises(ServiceUnavailable):
        asw.select()

    # Incremental upsert on the same continuous generator repopulates the set.
    watcher.push(_upsert(svc))
    assert await _wait_until(lambda: svc.id in asw._instances)
    assert asw.select() is not None  # no longer raises "No service"

    await _teardown_service(asw, client)


@pytest.mark.asyncio
async def test_incremental_upsert_and_delete_applied():
    """Plain upsert/delete events flow through the queue and mutate the set."""
    a, b = _svc('a', 1), _svc('b', 2)
    watcher = FakeWatcher(initial_event=_full_sync([a]))
    client = FakeEtcdClient(recursive_watchers=[watcher])

    asw = AsyncServiceWatcher(client, '/discovery', 'smite_shortlist')
    await asw.ensure_initialized()
    assert set(asw._instances) == {a.id}

    watcher.push(_upsert(b))
    assert await _wait_until(lambda: set(asw._instances) == {a.id, b.id})

    watcher.push(_delete(a))
    assert await _wait_until(lambda: set(asw._instances) == {b.id})

    await _teardown_service(asw, client)


@pytest.mark.asyncio
async def test_stop_teardown():
    """stop() sets the stop event and cancels the consumer; threads unwind on next poll."""
    svc = _svc('a', 1)
    watcher = FakeWatcher(initial_event=_full_sync([svc]))
    client = FakeEtcdClient(recursive_watchers=[watcher])

    asw = AsyncServiceWatcher(client, '/discovery', 'smite_shortlist')
    await asw.ensure_initialized()

    consumer = asw._consumer_task
    ring_consumer = asw._ring._consumer_task
    names = set(_watch_thread_names(asw))
    assert names <= _live_thread_names()  # both watch threads are running

    await asw.stop()

    assert asw._stop_event.is_set()
    assert asw._ring._stop_event.is_set()
    assert consumer.done()
    assert ring_consumer.done()

    # stop() does NOT join the daemon threads — they unwind on their next poll/timeout.
    # Simulate that next poll by releasing the blocked continue_watching() generators.
    for fw in client.all_watchers:
        fw.close()
    assert await _wait_until(
        lambda: not (names & _live_thread_names())
    ), 'watch threads did not unwind after their next poll'


# --- _AsyncHashRing ----------------------------------------------------------


@pytest.mark.asyncio
async def test_ring_recovers_after_key_deleted():
    """Same latent bug on the SCALAR ring: a key-deleted full sync wipes members,
    and continuous consumption repopulates them from a later FullSyncOne."""
    key = '/discovery/svc/ring'
    ring_watcher = FakeWatcher(initial_event=_ring_full_sync(['a:1', 'b:2']))
    client = FakeEtcdClient(scalar_watchers=[ring_watcher])

    ring = _AsyncHashRing(client, key)
    await ring.ensure_initialized()
    assert len(ring._members) == 2

    # Key deleted -> ring wiped.
    ring_watcher.push(FullSyncOneNoKey(key=key))
    assert await _wait_until(lambda: len(ring._members) == 0)

    # Later value repopulates the ring on the same continuous generator.
    ring_watcher.push(FullSyncOne(key=key, value=json.dumps(['c:3'])))
    assert await _wait_until(lambda: len(ring._members) == 1)

    await ring.stop()
    ring_watcher.close()
    await _wait_until(lambda: f'async-ring-watch:{key}' not in _live_thread_names())
