from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_fluxtube_gk import (
    GEOMETRY_NORMALIZATION,
    GEOMETRY_SCHEMA_VERSION,
    BoozerSurface,
    FieldLineSpec,
    GeometryMetadata,
    GeometryProvenance,
    GeometryProvider,
    GeometryRequest,
    GeometryResult,
    build_boozer_parallel_grid,
    build_geometry_metadata,
    build_physical_flux_tube_geometry_from_arrays,
    cache_path_is_external,
    internal_geometry_from_result,
    load_geometry_result_cache,
    resolve_geometry,
    sample_boozer_field_line,
    validate_geometry_result,
    write_geometry_result_cache,
)


class _SyntheticProvider:
    def __init__(self, amplitude=0.1):
        self.amplitude = amplitude

    def get_geometry(self, request: GeometryRequest) -> GeometryResult:
        parallel = build_boozer_parallel_grid(n_z=request.n_z, n_turns=1)
        surface = BoozerSurface(iota=0.7, B0=1.0)
        field_line = sample_boozer_field_line(
            surface,
            FieldLineSpec(
                rho=request.radial_value,
                alpha0=request.alpha,
                radial_coordinate=request.radial_coordinate,
            ),
            parallel,
        )
        ones = jnp.ones_like(parallel.z)
        zeros = jnp.zeros_like(parallel.z)
        physical = build_physical_flux_tube_geometry_from_arrays(
            field_line=field_line,
            B=1.0 + self.amplitude * jnp.cos(parallel.z),
            b_dot_grad_z=ones,
            grad_psi_sq=ones,
            grad_alpha_sq=1.2 * ones,
            grad_psi_dot_grad_alpha=zeros,
            B_cross_gradB_dot_grad_psi=zeros,
            B_cross_gradB_dot_grad_alpha=zeros,
            b_cross_kappa_dot_grad_psi=zeros,
            b_cross_kappa_dot_grad_alpha=zeros,
            source="synthetic:cosine-B",
            provider="synthetic",
        )
        metadata = build_geometry_metadata(
            request,
            provenance=GeometryProvenance(
                provider="synthetic",
                provider_version="1",
                revision="builtin",
                configuration=request.configuration,
            ),
            differentiable=True,
        )
        return GeometryResult(parallel_grid=parallel, physical=physical, metadata=metadata)


def test_public_provider_protocol_returns_valid_versioned_result():
    request = GeometryRequest(configuration="cosine-B", n_z=24, alpha=0.3)
    provider = _SyntheticProvider()

    assert isinstance(provider, GeometryProvider)
    result = resolve_geometry(provider, request)

    assert result.metadata.schema_version == GEOMETRY_SCHEMA_VERSION
    assert result.metadata.normalization.name == GEOMETRY_NORMALIZATION
    assert result.metadata.provenance.provider == "synthetic"
    assert result.metadata.differentiable is True
    np.testing.assert_allclose(result.physical.alpha, request.alpha, atol=1.0e-14)
    assert result.physical.iota.shape == ()
    assert result.physical.nfp == 1
    assert result.physical.endpoint_policy == "exclude"


def test_solver_coefficients_are_derived_only_after_provider_result():
    result = resolve_geometry(_SyntheticProvider(), GeometryRequest(n_z=20))
    internal = internal_geometry_from_result(result)

    np.testing.assert_allclose(internal.B, result.physical.B)
    np.testing.assert_allclose(internal.F, result.physical.b_dot_grad_z)
    assert not hasattr(result.physical, "D_x")
    assert internal.D_x.shape == result.parallel_grid.z.shape


def test_provider_arrays_remain_differentiable_while_metadata_is_static():
    request = GeometryRequest(configuration="gradient-test", n_z=16)

    def objective(amplitude):
        result = resolve_geometry(_SyntheticProvider(amplitude), request, validate=False)
        internal = internal_geometry_from_result(result)
        return jnp.sum(internal.B**2)

    amplitude = 0.12
    derivative = jax.jit(jax.grad(objective))(amplitude)
    expected = jnp.sum(2.0 * (1.0 + amplitude * jnp.cos(_grid_z(16))) * jnp.cos(_grid_z(16)))

    assert jnp.isfinite(derivative)
    np.testing.assert_allclose(derivative, expected, rtol=1.0e-12, atol=1.0e-12)


def test_validation_reports_nonfinite_and_normalization_errors():
    result = resolve_geometry(_SyntheticProvider(), GeometryRequest(n_z=12))
    nonfinite = replace(result.physical, B=result.physical.B.at[3].set(jnp.nan))
    with pytest.raises(ValueError, match="field 'B' contains non-finite"):
        validate_geometry_result(replace(result, physical=nonfinite))

    incompatible = replace(result.physical, normalization="foreign_normalization")
    with pytest.raises(ValueError, match="incompatible physical normalization"):
        validate_geometry_result(replace(result, physical=incompatible))


def test_validation_rejects_duplicate_periodic_endpoint_and_grid_size_mismatch():
    request = GeometryRequest(n_z=12)
    result = resolve_geometry(_SyntheticProvider(), request)
    duplicate_z = jnp.linspace(request.z_min, request.z_max, request.n_z)
    duplicate_grid = replace(result.parallel_grid, z=duplicate_z)
    duplicate_physical = replace(result.physical, z=duplicate_z)

    with pytest.raises(ValueError, match="duplicate upper endpoint"):
        validate_geometry_result(
            replace(result, parallel_grid=duplicate_grid, physical=duplicate_physical)
        )

    with pytest.raises(ValueError, match="request requires n_z=13"):
        validate_geometry_result(result, request=replace(request, n_z=13))


def test_geometry_metadata_rejects_unknown_schema_version():
    provenance = GeometryProvenance(provider="synthetic")
    with pytest.raises(ValueError, match="unsupported geometry schema version"):
        GeometryMetadata(provenance=provenance, schema_version=GEOMETRY_SCHEMA_VERSION + 1)


def test_cache_paths_must_stay_outside_source_tree(tmp_path: Path):
    repository = tmp_path / "repository"
    repository.mkdir()

    with pytest.raises(ValueError, match="outside the source repository"):
        cache_path_is_external(repository / ".geometry-cache", repository_root=repository)

    external = cache_path_is_external(tmp_path / "geometry-cache", repository_root=repository)
    assert external == (tmp_path / "geometry-cache").resolve()


def test_geometry_cache_roundtrip_preserves_schema_and_arrays(tmp_path: Path):
    repository = tmp_path / "repository"
    repository.mkdir()
    cache = tmp_path / "cache" / "cosine-B"
    request = GeometryRequest(configuration="cached-cosine-B", n_z=16)
    result = resolve_geometry(_SyntheticProvider(), request)

    written = write_geometry_result_cache(
        result,
        cache,
        repository_root=repository,
    )
    loaded = load_geometry_result_cache(written, request=request)

    assert written == cache.resolve()
    assert (written / "metadata.json").is_file()
    assert (written / "arrays.npz").is_file()
    assert loaded.metadata.schema_version == result.metadata.schema_version
    assert loaded.metadata.provenance == result.metadata.provenance
    assert loaded.metadata.differentiable is False
    np.testing.assert_allclose(loaded.parallel_grid.D_z, result.parallel_grid.D_z)
    for name in result.physical._dynamic_fields:
        np.testing.assert_allclose(getattr(loaded.physical, name), getattr(result.physical, name))
    np.testing.assert_allclose(
        internal_geometry_from_result(loaded).D_y,
        internal_geometry_from_result(result).D_y,
    )

    with pytest.raises(FileExistsError, match="already exists"):
        write_geometry_result_cache(result, cache, repository_root=repository)


def _grid_z(n_z: int):
    return build_boozer_parallel_grid(n_z=n_z, n_turns=1).z
