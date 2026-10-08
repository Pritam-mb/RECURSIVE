"""
API contract regression tests.

These guard the specific regression where the frontend called
`POST /api/simulate` while the route decorator was missing, so every scenario
load silently 404'd. The route table is asserted explicitly, and the frontend
API call sites are cross-checked against the registered routes.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from fastapi.routing import APIRoute

REPO_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_SRC = REPO_ROOT / "frontend" / "src"


@pytest.fixture(scope="module")
def app_module():
    import sys

    backend_root = str(Path(__file__).resolve().parents[1])
    if backend_root not in sys.path:
        sys.path.insert(0, backend_root)

    import main  # noqa: PLC0415

    return main


def _all_routes(routes):
    """Flatten included routers.

    FastAPI >= 0.13x keeps ``include_router`` results as lazy ``_IncludedRouter``
    wrappers in ``app.routes`` instead of copying each APIRoute, so walk into
    ``original_router`` to see the real routes. (Routers here carry their own
    prefix and are included without an extra one, so paths are unchanged.)
    """
    for route in routes:
        inner = getattr(route, "original_router", None)
        if inner is not None:
            yield from _all_routes(inner.routes)
        else:
            yield route


def _route_keys(app) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    for route in _all_routes(app.routes):
        if isinstance(route, APIRoute):
            for method in route.methods:
                if method in {"HEAD", "OPTIONS"}:
                    continue
                keys.add((method, route.path))
    return keys


class TestSimulateRouteRegistration:
    def test_post_and_delete_simulate_are_registered(self, app_module):
        keys = _route_keys(app_module.app)
        assert ("POST", "/api/simulate") in keys
        assert ("DELETE", "/api/simulate") in keys

    def test_scenario_endpoints_are_registered(self, app_module):
        keys = _route_keys(app_module.app)
        assert ("GET", "/api/scenarios") in keys
        assert ("POST", "/api/preflight") in keys
        assert ("POST", "/api/maneuver") in keys

    def test_simulate_route_is_reachable_by_name(self, app_module):
        simulate_routes = [
            route
            for route in _all_routes(app_module.app.routes)
            if isinstance(route, APIRoute) and route.path == "/api/simulate"
        ]
        methods = {
            method
            for route in simulate_routes
            for method in route.methods
            if method not in {"HEAD", "OPTIONS"}
        }
        assert methods == {"POST", "DELETE"}


class TestFrontendBackendParity:
    """
    Every literal API path used by the frontend must exist on the backend.

    Template-literal paths (e.g. ``/api/satellites/${id}``) are matched by
    their static prefix.
    """

    _CALL_PATTERN = re.compile(
        r"""(?:apiGet|apiPost|apiDelete|fetch)\(\s*([`'"])(.*?)\1""",
        re.S,
    )
    _API_PATH_PATTERN = re.compile(r"(/api/[^`'\"\s)]*)")

    def _frontend_paths(self) -> set[str]:
        paths: set[str] = set()
        for js_file in FRONTEND_SRC.rglob("*.js*"):
            if "node_modules" in js_file.parts:
                continue
            content = js_file.read_text(encoding="utf-8", errors="ignore")
            for call in self._CALL_PATTERN.finditer(content):
                literal = call.group(2)
                for raw in self._API_PATH_PATTERN.findall(literal):
                    # Normalise template segments:
                    # /api/satellites/${id}/telemetry -> /api/satellites/telemetry
                    path = re.sub(r"\$\{[^}]*\}", "", raw)
                    path = path.split("?", 1)[0]  # query strings are not part of the route
                    path = re.sub(r"/{2,}", "/", path)
                    if not path.endswith("/"):
                        path += "/"
                    paths.add(path)
        return paths

    def _backend_paths(self, app_module) -> set[str]:
        paths: set[str] = set()
        for route in _all_routes(app_module.app.routes):
            if not isinstance(route, APIRoute):
                continue
            # Strip FastAPI path params so they line up with the frontend
            # normalisation: /api/satellites/{norad_id}/telemetry
            # -> /api/satellites/telemetry
            path = re.sub(r"\{[^}]*\}", "", route.path)
            path = re.sub(r"/{2,}", "/", path)
            if not path.endswith("/"):
                path += "/"
            paths.add(path)
        return paths

    def test_no_frontend_api_path_is_missing_on_the_backend(self, app_module):
        frontend_paths = self._frontend_paths()
        assert frontend_paths, "expected to find frontend API call sites"

        backend_paths = self._backend_paths(app_module)
        # The WebSocket endpoint is not an APIRoute.
        missing = {
            path for path in frontend_paths
            if path not in backend_paths
            and path.rstrip("/") not in {p.rstrip("/") for p in backend_paths}
        }

        assert not missing, (
            "frontend calls endpoints the backend does not register: "
            + ", ".join(sorted(missing))
        )
