# Root Makefile forwarding targets to code/Makefile
.PHONY: all lib test clean bench bench-triple directories

all lib test clean bench bench-triple directories:
	@$(MAKE) -C code $@ $(if $(CFLAGS_EXTRA),CFLAGS_EXTRA="$(CFLAGS_EXTRA)",)
