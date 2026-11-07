# Root Makefile forwarding targets to code/Makefile
.PHONY: all lib test clean bench directories

all lib test clean bench directories:
	@$(MAKE) -C code $@ $(if $(CFLAGS_EXTRA),CFLAGS_EXTRA="$(CFLAGS_EXTRA)",)
