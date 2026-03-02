# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `lazy_load`, `idle_timeout_seconds`, and `preload` configuration fields for models.
- `LazyHandlerProxy` class for on-demand model loading and automatic idle unloading.
- `status` field to the `/v1/models` endpoint indicating if a model is "ready" or "unloaded".
- Support for concurrent requests during model spawn in lazy mode.

### Changed
- Multi-handler mode now supports a mix of lazy and eager loaded models.
- Default behavior can be configured globally with `default_lazy_load` and `default_idle_timeout_seconds`.
