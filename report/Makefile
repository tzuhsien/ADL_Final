SOURCES := report.tex

TARGETS = $(subst .tex,.pdf,$(SOURCES))

all: $(TARGETS)

%.pdf: %.tex %.bib
	latexmk -pdf -xelatex $<

clean: $(TARGETS)
	latexmk -C $(TARGETS)
