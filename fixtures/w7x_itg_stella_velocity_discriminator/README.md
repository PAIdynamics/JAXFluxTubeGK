# W7-X stella Velocity Discriminator

Regenerate from the repository root with:

```bash
uv run python scripts/run_w7x_stella_velocity_discriminator.py
```

The discriminator holds the stella geometry, kx=0/n_kx=1,
`ky=(0.1,0.2,0.3)`, species gradients, and t=200 late-half
growth window fixed. It varies only velocity-space resolution
and backend, then reports whether the remaining ky=0.3
frequency/profile mismatch is velocity-sensitive.
