# Root Makefile forwarding targets to code/Makefile
.PHONY: all lib test clean bench bench-triple bench-drift bench-regression directories version

all lib test clean bench bench-triple bench-drift bench-regression directories version:
	@$(MAKE) -C code $@ $(if $(CFLAGS_EXTRA),CFLAGS_EXTRA="$(CFLAGS_EXTRA)",)
