# Root Makefile forwarding targets to code/Makefile
.PHONY: all lib test clean bench bench-triple bench-drift bench-regression bench-memory bench-main directories version

all lib test clean bench bench-triple bench-drift bench-regression bench-memory bench-main directories version:
	@$(MAKE) -C code $@ $(if $(CFLAGS_EXTRA),CFLAGS_EXTRA="$(CFLAGS_EXTRA)",)
