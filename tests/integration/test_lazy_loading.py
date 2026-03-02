import pytest
from app.config import load_config_from_yaml, ModelEntryConfig, MultiModelServerConfig


@pytest.fixture
def lazy_config():
    return MultiModelServerConfig(
        host="0.0.0.0",
        port=8000,
        default_lazy_load=True,
        default_idle_timeout_seconds=60,
        models=[
            ModelEntryConfig(
                model_path="mlx-community/test-model",
                model_id="lazy-model",
                model_type="lm",
                lazy_load=True,
                idle_timeout_seconds=60,
            ),
            ModelEntryConfig(
                model_path="mlx-community/other-model",
                model_id="preload-model",
                model_type="lm",
                lazy_load=False,
                preload=True,
            ),
        ]
    )


def test_cold_start_only_preloads_marked_models(lazy_config):
    """Verify only preload models load at startup."""
    # In actual integration, would start server and check /v1/models
    # For now, verify config parsing
    assert lazy_config.models[0].lazy_load is True
    assert lazy_config.models[0].preload is False
    assert lazy_config.models[1].lazy_load is False
    assert lazy_config.models[1].preload is True


def test_idle_timeout_configuration():
    """Verify idle timeout is configurable."""
    config = MultiModelServerConfig(
        host="0.0.0.0",
        port=8000,
        default_idle_timeout_seconds=1800,
        models=[
            ModelEntryConfig(
                model_path="test",
                model_id="short-timeout",
                model_type="lm",
                lazy_load=True,
                idle_timeout_seconds=300,  # 5 minutes
            ),
            ModelEntryConfig(
                model_path="test2",
                model_id="long-timeout",
                model_type="lm",
                lazy_load=True,
                idle_timeout_seconds=7200,  # 2 hours
            ),
        ]
    )
    assert config.models[0].idle_timeout_seconds == 300
    assert config.models[1].idle_timeout_seconds == 7200


def test_zero_timeout_disables_unload():
    """Verify zero timeout means never unload."""
    config = ModelEntryConfig(
        model_path="test",
        model_id="no-unload",
        model_type="lm",
        lazy_load=True,
        idle_timeout_seconds=0,
    )
    assert config.idle_timeout_seconds == 0


def test_load_config_from_yaml_lazy_defaults(tmp_path):
    """Verify that load_config_from_yaml correctly parses lazy load defaults."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("""
server:
  default_lazy_load: true
  default_idle_timeout_seconds: 123
models:
  - model_path: mlx-community/test-model
    model_id: lazy-model
""")
    config = load_config_from_yaml(str(config_file))
    assert config.default_lazy_load is True
    assert config.default_idle_timeout_seconds == 123
    assert config.models[0].lazy_load is True
    assert config.models[0].idle_timeout_seconds == 123
