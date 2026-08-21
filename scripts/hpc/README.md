# HPC facility assets

Facility-specific environment, monitoring, and launch assets are organized as:

```text
scripts/hpc/<facility>/<system>/
```

For example, `olcf/frontier/environments` contains interactive Frontier setup
scripts, while `olcf/frontier/omnistat` contains the corresponding Omnistat
collector configurations and `olcf/frontier/installation` contains installation
scripts. Facility-wide helpers, such as `olcf/proxy-env.sh`, live at the facility
level. Keep portable HydraGNN utilities outside this hierarchy.
