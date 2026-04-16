"""Tests for the async pigeon client."""

import asyncio
from contextvars import copy_context
from unittest.mock import AsyncMock, MagicMock, patch

import grpc
import pytest

from osprey.async_worker.lib.pigeon.skip_rate_limit import skip_rate_limit_context


# --- skip_rate_limit contextvars ---


def test_skip_rate_limit_default_false():
    assert skip_rate_limit_context.skip is False


def test_skip_rate_limit_set_and_get():
    skip_rate_limit_context.skip = True
    assert skip_rate_limit_context.skip is True
    skip_rate_limit_context.skip = False
    assert skip_rate_limit_context.skip is False


def test_skip_rate_limit_property_api():
    """Uses .skip property, matching the gevent.local API."""
    skip_rate_limit_context.skip = True
    assert skip_rate_limit_context.skip is True
    skip_rate_limit_context.skip = False


# --- RoutingType constants ---


def test_routing_type_values():
    from osprey.async_worker.lib.pigeon.client import RoutingType

    assert RoutingType.CHUNKED == 1
    assert RoutingType.SCALAR == 2
    assert RoutingType.ROUND_ROBIN == 3
    assert RoutingType.ENVOY == 4
    assert len(RoutingType.ALL) == 4


# --- GRPC HTTP code translation ---


def test_grpc_http_translations():
    from osprey.async_worker.lib.pigeon.client import _GRPC_HTTP_CODE_TRANSLATIONS

    assert _GRPC_HTTP_CODE_TRANSLATIONS[grpc.StatusCode.OK] == 200
    assert _GRPC_HTTP_CODE_TRANSLATIONS[grpc.StatusCode.NOT_FOUND] == 404
    assert _GRPC_HTTP_CODE_TRANSLATIONS[grpc.StatusCode.INTERNAL] == 500
    assert _GRPC_HTTP_CODE_TRANSLATIONS[grpc.StatusCode.UNAVAILABLE] == 503
    assert _GRPC_HTTP_CODE_TRANSLATIONS[grpc.StatusCode.DEADLINE_EXCEEDED] == 504


# --- RetryPolicy ---


def test_retry_policy_type():
    from osprey.async_worker.lib.pigeon.client import RetryPolicy

    policy: RetryPolicy = {
        'retryable_grpc_status_codes': {grpc.StatusCode.UNAVAILABLE},
        'max_secondaries_to_retry': 2,
    }
    assert grpc.StatusCode.UNAVAILABLE in policy['retryable_grpc_status_codes']
    assert policy['max_secondaries_to_retry'] == 2


# --- ServiceDefinition ---


def test_service_definition_type():
    from osprey.async_worker.lib.pigeon.client import ServiceDefinition

    sd: ServiceDefinition = {'address': 'localhost', 'ip': '127.0.0.1', 'port': 5000}
    assert sd['address'] == 'localhost'
    assert sd['port'] == 5000


class _TestStub:
    def __init__(self, channel):
        self.Predict = AsyncMock(return_value='ok')


class _TestService:
    def __init__(self, connection_key=(('localhost', None), 5000), name='test_service'):
        self.connection_key = connection_key[0]
        self.grpc_port = connection_key[1]
        self.connection_address = 'localhost'
        self.name = name


def test_idle_timeout_env_parsing(monkeypatch):
    from osprey.async_worker.lib.pigeon.client import _get_idle_timeout_ns

    monkeypatch.setenv('OSPREY_PIGEON_CHANNEL_IDLE_TIMEOUT_SECONDS', '2.5')
    assert _get_idle_timeout_ns() == 2_500_000_000


@pytest.mark.parametrize(
    ('env_value', 'expected'),
    [
        (None, 0),
        ('0', 0),
        ('-1', 0),
        ('not-a-number', 0),
    ],
)
def test_idle_timeout_env_parsing_disabled_values(monkeypatch, env_value, expected):
    from osprey.async_worker.lib.pigeon.client import _get_idle_timeout_ns

    if env_value is None:
        monkeypatch.delenv('OSPREY_PIGEON_CHANNEL_IDLE_TIMEOUT_SECONDS', raising=False)
    else:
        monkeypatch.setenv('OSPREY_PIGEON_CHANNEL_IDLE_TIMEOUT_SECONDS', env_value)

    assert _get_idle_timeout_ns() == expected


def test_evict_idle_clients_closes_only_idle(monkeypatch):
    from osprey.async_worker.lib.pigeon.client import RoutedClient, RoutingType

    stale_channel = MagicMock()
    stale_channel.close = AsyncMock()
    active_channel = MagicMock()
    active_channel.close = AsyncMock()
    monkeypatch.setenv('OSPREY_PIGEON_CHANNEL_IDLE_TIMEOUT_SECONDS', '1')

    with patch('osprey.async_worker.lib.pigeon.client.grpc.aio.insecure_channel') as insecure_channel:
        insecure_channel.side_effect = [stale_channel, active_channel]
        client = RoutedClient(
            'test_service',
            read_timeout=1,
            stub_cls=_TestStub,
            routing_type=RoutingType.ENVOY,
            envoy_endpoint={'address': 'localhost', 'port': 5000},
        )

        stale_service = _TestService((('localhost', None), 5001))
        active_service = _TestService((('localhost', None), 5002))
        stale_key = client._get_service_key(stale_service)
        active_key = client._get_service_key(active_service)

        client._get_client(stale_service)
        client._get_client(active_service)
        client._client_last_used_ns[stale_key] = 0
        client._client_last_used_ns[active_key] = 0
        client._client_active_requests[active_key] = 1

        client._evict_idle_clients()

        assert stale_key not in client._clients
        assert active_key in client._clients
        assert stale_key not in client._open_channels


def test_chunked_request_holds_service_activity_across_chunks(monkeypatch):
    from osprey.async_worker.lib.pigeon.client import RoutedClient, RoutingType

    class ChunkedRequest:
        def __init__(self):
            self.ids = [1, 2, 3]

        def ClearField(self, name):
            setattr(self, name, [])

    class ChunkedResponse:
        def __init__(self):
            self.merged = 0

        def MergeFrom(self, other):
            self.merged += 1

    class ChunkedStub:
        def __init__(self, channel):
            self.Predict = AsyncMock(side_effect=[ChunkedResponse(), ChunkedResponse(), ChunkedResponse()])

    async def run_test() -> None:
        service = _TestService()
        monkeypatch.setenv('OSPREY_PIGEON_CHANNEL_IDLE_TIMEOUT_SECONDS', '1')

        with patch('osprey.async_worker.lib.pigeon.client.grpc.aio.insecure_channel') as insecure_channel:
            insecure_channel.return_value = MagicMock()
            client = RoutedClient(
                'test_service',
                read_timeout=1,
                stub_cls=ChunkedStub,
                request_field='ids',
                routing_type=RoutingType.ENVOY,
                chunk_size=1,
                envoy_endpoint={'address': 'localhost', 'port': 5000},
            )
            key = client._get_service_key(service)

            with patch.object(client, '_generate_routed_calls', return_value={service: [1, 2, 3]}):
                original_invoke = client._invoke_service_method
                observed_counts = []

                async def wrapped_invoke(*args, **kwargs):
                    observed_counts.append(client._client_active_requests.get(key, 0))
                    return await original_invoke(*args, **kwargs)

                client._invoke_service_method = wrapped_invoke  # type: ignore[method-assign]
                response = await client._chunked_request('Predict', ChunkedRequest(), 'ids')

            assert isinstance(response, ChunkedResponse)
            assert observed_counts == [1, 1, 1]
            assert key not in client._client_active_requests

    asyncio.run(run_test())


def test_release_service_activity_after_cleanup_does_not_go_negative(monkeypatch):
    from osprey.async_worker.lib.pigeon.client import RoutedClient, RoutingType

    async def run_test() -> None:
        monkeypatch.setenv('OSPREY_PIGEON_CHANNEL_IDLE_TIMEOUT_SECONDS', '1')

        with patch('osprey.async_worker.lib.pigeon.client.grpc.aio.insecure_channel') as insecure_channel:
            channel = MagicMock()
            channel.close = AsyncMock()
            insecure_channel.return_value = channel
            client = RoutedClient(
                'test_service',
                read_timeout=1,
                stub_cls=_TestStub,
                routing_type=RoutingType.ENVOY,
                envoy_endpoint={'address': 'localhost', 'port': 5000},
            )
            service = _TestService()
            key = client._get_service_key(service)

            async def wait_for_release(*args, **kwargs):
                client._cleanup_client(key)
                return 'ok'

            client._get_client(service).Predict.side_effect = wait_for_release
            await client._invoke_service_method(service, 'Predict', message=MagicMock())

            assert key not in client._client_active_requests

    asyncio.run(run_test())
