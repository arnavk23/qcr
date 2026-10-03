# Build all manuscripts: make -C manuscripts   (needs a TeX Live installation with beamer)
LATEX = pdflatex -interaction=nonstopmode -halt-on-error

.PHONY: all notes reproduction extension talk clean
all: notes reproduction extension talk

notes:
	cd notes && $(LATEX) notes.tex >/dev/null && $(LATEX) notes.tex >/dev/null
reproduction:
	cd reproduction && $(LATEX) reproduction_report.tex >/dev/null && $(LATEX) reproduction_report.tex >/dev/null
extension:
	cd extension && $(LATEX) extension_paper.tex >/dev/null && $(LATEX) extension_paper.tex >/dev/null
talk:
	cd talk && $(LATEX) talk.tex >/dev/null && $(LATEX) talk.tex >/dev/null

clean:
	find . \( -name '*.aux' -o -name '*.log' -o -name '*.out' -o -name '*.toc' -o -name '*.nav' \
	  -o -name '*.snm' -o -name '*.vrb' \) -delete
