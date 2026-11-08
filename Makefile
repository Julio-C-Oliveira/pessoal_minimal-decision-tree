# Root Makefile forwarding targets to code/Makefile
.PHONY: all lib test clean bench bench-triple bench-drift bench-regression bench-memory bench-main bench-esp32 directories version

all lib test clean bench bench-triple bench-drift bench-regression bench-memory bench-main bench-esp32 directories version:
	@$(MAKE) -C code $@ $(if $(CFLAGS_EXTRA),CFLAGS_EXTRA="$(CFLAGS_EXTRA)",)
