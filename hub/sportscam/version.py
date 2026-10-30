"""Single source of truth for the product version (house rule: one version per release)."""

PRODUCT = "sports-cam"          # working name until Victor picks one
DISPLAY_NAME = "VICCOM Sports Cam"
SEMVER = "0.1.0"
BUILD_NUMBER = 1                # only ever goes up
CHANNEL = "alpha"               # alpha | beta | production
DISTRIBUTION = "direct"         # direct | steam | msstore | itch (store builds disable own updater)
