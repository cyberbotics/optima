CARD ?= MAX5C
CC := g++

OBJDIR := obj/$(CARD)
LIBDIR := lib/$(CARD)
BINDIR := bin/$(CARD)
INCDIR := inc/$(CARD)
MAXINCDIR := $(INCDIR)/max
MAXOBJDIR := $(OBJDIR)/max

CFLAGS := -O3 -fPIC -std=c++11 -Wall -Wextra
LFLAGS :=

EXE := fixed-forward-test
SONAME := lib$(shell echo $EXE | tr '[:upper:]' '[:lower:]').so

TRIMMEDUSER := $(shell echo $(USER) | head -c 10)
TRIMMEDPROJECT := $(shell echo $(EXE) | head -c 25)

SIMNAME ?= $(CARD)$(TRIMMEDUSER)$(TRIMMEDPROJECT)Sim

USE_SLIC := 1
include $(MAXCOMPILERDIR)/lib/Makefile.include
include .cproperties
include .maxproperties

#TEST_INC = -I$(MAXCOMPILERDIR_QUOTE)/include/slic

CFLAGS := $(CFLAGS) -I$(MAXINCDIR) $(MAXCOMPILER_INC) $(MACROS) $(INCPATH) 
LFLAGS = $(LIBPATH) $(LIBS) $(MAXCOMPILER_LIBS)

ifeq ($(MAXFILES), )
	BUILD_LOCATIONS += $(foreach build_loc, $(BUILD_FILES), $(shell tail -n 1 $(build_loc)))
	BUILD_LOCATIONS += $(BUILD_DIRS)
	MAXFILES += $(foreach build_loc, $(BUILD_LOCATIONS), $(wildcard $(build_loc)/results/*.max))
	MAXFILES += $(SELECTED_MAXFILES)
endif

MAX = $(patsubst %, $(MAXOBJDIR)/%.o, $(basename $(notdir $(MAXFILES))))
MAXINC = $(patsubst %, $(MAXINCDIR)/%.h, $(basename $(notdir $(MAXFILES))))

_OBJ = $(shell find $(SRCPATH) -name '*.c' -o -name '*.cpp')
OBJ = $(patsubst %.c, $(OBJDIR)/%.o, $(_OBJ)) $(MAX)

$(OBJDIR)/%.o: %.c
	@mkdir -p $(@D)
	$(CC) -c -o $@ $< $(CFLAGS)

$(MAX): $(MAXFILES)
	$(eval MAXFILE = $(filter %$(basename $(@F)).max, $(MAXFILES)))
	$(if $(wildcard $(MAXFILE)), , $(error $(ACTIVE_BUILD_NOT_SELECTED)))
	@mkdir -p $(@D)
	"$(MAXCOMPILERDIR)/bin/sliccompile" "$(MAXFILE)" $(@)

$(MAXINCDIR)/%.h:
	$(eval MAXFILE = $(filter %$(basename $(@F)).max, $(MAXFILES)))
	$(if $(wildcard $(MAXFILE)), , $(error $(ACTIVE_BUILD_NOT_SELECTED)))
	@mkdir -p $(@D)
	"$(MAXCOMPILERDIR)/bin/sliccompile" -t cheader -d $(MAXINCDIR) "$(MAXFILE)"

$(BINDIR)/$(EXE): $(MAXINC) $(OBJ)
	@mkdir -p $(@D)
	$(CC) -o $(@) $(OBJ) $(CFLAGS) $(LFLAGS)

$(LIBDIR)/$(SONAME): $(MAXINC) $(OBJ)
	@mkdir -p $(@D)
	$(CC) -shared -o $(@) $(OBJ) $(CFLAGS) $(LFLAGS)

generate_maxfiles_header:
	@rm -f $(INCDIR)/Maxfiles.h
	@printf	"\
	/* File Maxfiles.h  */ \n \
	#ifndef MAXFILES_H     \n \
	#define MAXFILES_H     \n \
                               \n \
	/**                    \n \
	 * Add .max files here \n \
	 */                    \n" >> $(INCDIR)/Maxfiles.h
	$(foreach header, $(MAXINC), printf "#include \"$(notdir $(header))\"\n" >> $(INCDIR)/Maxfiles.h;)
	@printf "\n#endif /* !MAXFILES_H */ \n" >> $(INCDIR)/Maxfiles.h 

all: build

build: $(BINDIR)/$(EXE)

create_so: $(LIBDIR)/$(SONAME)

run: build ; @$(BINDIR)/$(EXE)
run_dfe: export MAXELEROSDIR=/opt/maxeler/maxeleros
run_dfe: export LD_LIBRARY_PATH := $(MAXELEROSDIR)/lib:$(LD_LIBRARY_PATH)
run_dfe: ; @make run
run_sim: export MAXELEROSDIR=$(MAXCOMPILERDIR)/lib/maxeleros-sim
run_sim: export LD_LIBRARY_PATH := $(MAXCOMPILERDIR)/lib/maxeleros-sim/lib:$(LD_LIBRARY_PATH)
run_sim: export SLIC_CONF = use_simulation=$(SIMNAME)
run_sim: ; @make start_sim run stop_sim

setup_event_monitoring:
	nohup maxeventlog --name ${DEFAULT_EVENTLOG_SERVER} -v4 &

end_event_monitoring:
	$(shell [ -d ${RESULT_DIR} ] || mkdir -p ${RESULT_DIR})
	cd ${RESULT_DIR} && maxeventlogrendersvg -n ${DEFAULT_EVENTLOG_SERVER}
	maxeventlog --name ${DEFAULT_EVENTLOG_SERVER} --stop

%_with_event_monitoring: setup_event_monitoring % end_event_monitoring; @echo

start_sim: ; @'$(MAXCOMPILERDIR)/bin/maxcompilersim' -n $(SIMNAME) -c $(CARD) restart
stop_sim: ; @'$(MAXCOMPILERDIR)/bin/maxcompilersim' -n $(SIMNAME) stop
restart_sim: stop_sim start_sim

help: ; @echo "$$USAGE"

define safe_rm
	@if [ -d "$(strip $(1))" ]; then rm -r $(1); fi
endef

clean:
	$(call safe_rm, $(BINDIR))
	$(call safe_rm, $(MAXOBJDIR))
	$(call safe_rm, $(OBJDIR))
	@rm *~ core* 2> /dev/null || true

.PHONY: all build clean create_so help restart_sim run
.PHONY: run_dfe run_sim start_sim stop_sim $(MAX)

define ACTIVE_BUILD_NOT_SELECTED_IN_MAXIDE

[ERROR] No Maxfile $(MAXFILE) found.
[HINT] Please create the Maxfile by building the DFE Project associated with it
[HINT] (this can be done by clicking in the 'Build <target>' button in the DFE project),
[HINT] or go to Maxfiles Manager and select existing maxfiles (this can be done
[HINT] by right clicking on the CPU project and selecting 'Set up Maxfiles' option)

endef
export ACTIVE_BUILD_NOT_SELECTED_IN_MAXIDE

define ACTIVE_BUILD_NOT_SELECTED_IN_TERMINAL

[ERROR] No Maxfile $(MAXFILE) found.
[HINT] To specify different Maxfile location
[HINT] set MAXFILES environmental variable.

endef
export ACTIVE_BUILD_NOT_SELECTED_IN_TERMINAL

ifeq ($(MAXIDE),true)
  ACTIVE_BUILD_NOT_SELECTED = $(ACTIVE_BUILD_NOT_SELECTED_IN_MAXIDE)
else
  ACTIVE_BUILD_NOT_SELECTED = $(ACTIVE_BUILD_NOT_SELECTED_IN_TERMINAL)
endif

define USAGE
Usage:
    make                     Builds Project using .max file of specified type
                             This target is equivalent to:
                             make build
    make build               Builds Project using .max file of specified type
    make clean               Cleans Project
    make create_so           Creates shared library using .max file
                             of specified type.
    make help                Prints Help message
    make restart_sim         Restart Simulation
    make run                 Run created exe file
    make run_dfe             Run Project using dfe .max file
    make run_sim             Run Project using sim .max file inside Simulation
    make start_sim           Start Simulation
    make stop_sim            Stop Simulation

Options:
    CARD                     The simulated card model
    SIMNAME                  The simulation name
    MAXFILES                 The location of the .max file
endef
export USAGE
