CC      ?= cc
CFLAGS  ?= -O2 -std=c11 -Wall -Wextra
BUILD   := build
INC     := -Iinclude

CORE := src/go_rules.c src/nn_encode.c src/rng.c src/puct.c
LIBSRC := src/go_rules.c src/nn_encode.c src/ffi.c

UNAME_S := $(shell uname -s)
# The NN binaries embed Python (nn_eval.c) and run the real checkpoint on
# the io thread. macOS uses the CLT Python3.framework (the venv's ABI);
# Linux asks python3-config. PYBIN must be the interpreter owning torch;
# the default finds the chess project's venv (sibling checkout).
PYBIN ?= $(if $(wildcard ../chessgui/.venv/bin/python3),\
            ../chessgui/.venv/bin/python3,\
         $(if $(wildcard $(HOME)/chessgui/.venv/bin/python3),\
            $(HOME)/chessgui/.venv/bin/python3,python3))
ifeq ($(UNAME_S),Darwin)
  PYFW := /Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.9
  PYINCLUDE := -I$(PYFW)/Headers
  PYLIBDIR  := -L$(PYFW)/lib
  PYLIBS    := $(PYLIBDIR) -lpython3.9 -Wl,-rpath,$(abspath $(PYFW)/../../..)
else
  PYINCLUDE := $(shell python3-config --includes)
  PYLIBS    := $(shell python3-config --ldflags --embed) $(shell $(PYBIN) -c "import sysconfig; d = sysconfig.get_config_var('LIBDIR'); print(f'-Wl,-rpath,{d}' if d else '')")
endif
PURELIB   := $(shell $(PYBIN) -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])' 2>/dev/null)
PYFLAGS   = $(PYINCLUDE) -DPUCT_REPO_ROOT='"$(abspath ..)"' -DPUCT_PURELIB='"$(PURELIB)"'
PYLD      = $(PYLIBS)

all: $(BUILD)/test_go_rules $(BUILD)/test_nn_encode $(BUILD)/test_sp_dump \
     $(BUILD)/test_puct $(BUILD)/test_selfplay_e2e $(BUILD)/selfplay \
     $(BUILD)/libgo.dylib

nn: $(BUILD)/selfplay_nn $(BUILD)/gtp $(BUILD)/engine_mux

check: all
	./$(BUILD)/test_go_rules
	./$(BUILD)/test_nn_encode
	./$(BUILD)/test_sp_dump
	./$(BUILD)/test_puct
	PUCT_TASKS=4 ./$(BUILD)/test_puct
	./$(BUILD)/test_selfplay_e2e

$(BUILD)/test_go_rules: src/go_rules.c tests/test_go_rules.c include/go.h | $(BUILD)
	$(CC) $(CFLAGS) $(INC) -o $@ src/go_rules.c tests/test_go_rules.c

$(BUILD)/test_nn_encode: src/go_rules.c src/nn_encode.c tests/test_nn_encode.c \
                         include/go.h include/go_nn.h | $(BUILD)
	$(CC) $(CFLAGS) $(INC) -o $@ src/go_rules.c src/nn_encode.c tests/test_nn_encode.c

$(BUILD)/test_sp_dump: src/sp_dump.c tests/test_sp_dump.c include/go_sp_dump.h | $(BUILD)
	$(CC) $(CFLAGS) $(INC) -o $@ src/sp_dump.c tests/test_sp_dump.c

$(BUILD)/test_puct: src/go_rules.c src/nn_encode.c src/rng.c src/puct.c \
                    tests/test_puct.c include/go.h include/go_nn.h \
                    include/go_rng.h include/go_puct.h | $(BUILD)
	$(CC) $(CFLAGS) $(INC) -o $@ src/go_rules.c src/nn_encode.c src/rng.c \
	       src/puct.c tests/test_puct.c -pthread -lm

$(BUILD)/test_selfplay_e2e: src/go_rules.c tests/test_selfplay_e2e.c \
                            include/go.h include/go_sp_dump.h | $(BUILD)
	$(CC) $(CFLAGS) $(INC) -o $@ src/go_rules.c tests/test_selfplay_e2e.c

$(BUILD)/selfplay: src/selfplay.c src/match_util.c src/nn_stub.c src/sp_dump.c \
                   $(CORE) include/go.h include/go_nn.h include/go_rng.h \
                   include/go_puct.h include/go_match.h include/go_nn_eval.h \
                   include/go_sp_dump.h | $(BUILD)
	$(CC) $(CFLAGS) $(INC) -o $@ src/selfplay.c src/match_util.c \
	       src/nn_stub.c src/sp_dump.c $(CORE) -pthread -lm

$(BUILD)/libgo.dylib: $(LIBSRC) include/go.h include/go_nn.h include/go_ffi.h | $(BUILD)
	$(CC) $(CFLAGS) $(INC) -shared -fPIC -o $@ $(LIBSRC)

# NN binaries: embedded-Python evaluator (nn_eval.c). Manual targets —
# `make nn`. device "dummy" works in all three (no net needed).
$(BUILD)/selfplay_nn: src/selfplay.c src/match_util.c src/nn_eval.c src/sp_dump.c \
                      $(CORE) include/go_nn_eval.h | $(BUILD)
	$(CC) $(CFLAGS) $(INC) $(PYFLAGS) -o $@ src/selfplay.c src/match_util.c \
	       src/nn_eval.c src/sp_dump.c $(CORE) -pthread -lm $(PYLD)

$(BUILD)/gtp: src/gtp.c src/nn_eval.c src/match_util.c $(CORE) include/go_nn_eval.h | $(BUILD)
	$(CC) $(CFLAGS) $(INC) $(PYFLAGS) -o $@ src/gtp.c src/nn_eval.c \
	       src/match_util.c $(CORE) -pthread -lm $(PYLD)

$(BUILD)/engine_mux: src/engine_mux.c src/nn_eval.c src/match_util.c $(CORE) include/go_nn_eval.h | $(BUILD)
	$(CC) $(CFLAGS) $(INC) $(PYFLAGS) -o $@ src/engine_mux.c src/nn_eval.c \
	       src/match_util.c $(CORE) -pthread -lm $(PYLD)

$(BUILD):
	mkdir -p $(BUILD)

# --- wasm build (web GUI; the gui branch) ------------------------------------
# The whole GTP engine in one module: JS is the net evaluator
# (wasm_glue.c under Asyncify — the stack suspends inside puct_think
# while onnxruntime-web runs). Single-threaded WASM: no pthreads, no
# SharedArrayBuffer, any static host. The worker captures printf via the
# module's print hook and drives GTP with gtp_exec.
WASM_DIR    = build/wasm
EMS        ?= emcc
WASM_SRCS   = src/gtp.c src/wasm_glue.c $(CORE)
WASM_FLAGS  = -O2 -std=c11 -Iinclude \
  -sMODULARIZE=1 -sEXPORT_ES6=1 -sEXPORT_NAME=createGoEngine -sENVIRONMENT=web,worker,node \
  -sASYNCIFY=1 -sALLOW_MEMORY_GROWTH=1 -sINITIAL_MEMORY=96MB \
  -sMAXIMUM_MEMORY=2GB -sSTACK_SIZE=1MB \
  -sEXPORTED_FUNCTIONS=_malloc,_free,_gtp_wasm_init,_gtp_exec,_gtp_interrupt,_gtp_wasm_attach_net,_gtp_wasm_set_sims \
  -sEXPORTED_RUNTIME_METHODS=ccall,cwrap,UTF8ToString,stringToUTF8,HEAPU8,HEAPF32

$(WASM_DIR):
	mkdir -p $(WASM_DIR)

$(WASM_DIR)/go_engine.js: $(WASM_SRCS) include/go_nn_eval.h Makefile | $(WASM_DIR)
	$(EMS) -Wall -Wextra $(WASM_FLAGS) $(WASM_SRCS) -o $@

wasm: $(WASM_DIR)/go_engine.js

# The referee worker's rules module: the same FFI surface the Python
# cffi binding uses (go_game_new/push/board/legal/termination), no
# search, no net — tiny, instant to instantiate.
WASM_RULES_FLAGS = -O2 -std=c11 -Iinclude \
  -sMODULARIZE=1 -sEXPORT_ES6=1 -sEXPORT_NAME=createGoRules -sENVIRONMENT=web,worker,node \
  -sALLOW_MEMORY_GROWTH=1 -sINITIAL_MEMORY=16MB \
  -sEXPORTED_FUNCTIONS=_malloc,_free,_go_init,_go_game_new,_go_game_free,_go_side,_go_ply,_go_board_size,_go_n_pass,_go_board,_go_legal,_go_push,_go_push_fast,_go_termination,_go_encode \
  -sEXPORTED_RUNTIME_METHODS=ccall,HEAPU8,HEAPF32,HEAPU16

$(WASM_DIR)/go_rules.js: src/ffi.c src/go_rules.c src/nn_encode.c \
                        include/go_ffi.h include/go.h include/go_nn.h \
                        Makefile | $(WASM_DIR)
	$(EMS) -Wall -Wextra $(WASM_RULES_FLAGS) src/ffi.c src/go_rules.c \
	       src/nn_encode.c -o $@

wasm-rules: $(WASM_DIR)/go_rules.js

clean:
	rm -rf $(BUILD)

.PHONY: all check clean nn wasm wasm-rules
