"""Compatibility for the Versioneer copy vendored by the pinned DESC fork."""

import configparser


if not hasattr(configparser, "SafeConfigParser"):
    configparser.SafeConfigParser = configparser.ConfigParser  # type: ignore[attr-defined]
