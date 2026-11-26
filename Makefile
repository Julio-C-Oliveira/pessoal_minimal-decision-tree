# Root Makefile forwarding targets to code/Makefile
.PHONY: all lib test clean bench bench-triple bench-drift bench-regression bench-memory bench-main bench-main-federated bench-prep bench-esp32 bench-tune bench-federated bench-fed-net directories version

all lib test clean bench bench-triple bench-drift bench-regression bench-memory bench-main bench-main-federated bench-prep bench-esp32 bench-tune bench-federated bench-fed-net directories version:
	@$(MAKE) -C code $@ $(if $(CFLAGS_EXTRA),CFLAGS_EXTRA="$(CFLAGS_EXTRA)",)
