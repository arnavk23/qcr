# Build all manuscripts: make -C manuscripts   (needs a TeX Live installation with beamer)
LATEX = pdflatex -interaction=nonstopmode -halt-on-error

.PHONY: all notes reproduction paper talk clean
all: notes reproduction paper talk

notes:
	cd notes && $(LATEX) notes.tex >/dev/null && $(LATEX) notes.tex >/dev/null
reproduction:
	cd reproduction && $(LATEX) reproduction_report.tex >/dev/null && $(LATEX) reproduction_report.tex >/dev/null
paper:
	cd paper && $(LATEX) paper.tex >/dev/null && bibtex paper >/dev/null && $(LATEX) paper.tex >/dev/null && $(LATEX) paper.tex >/dev/null
talk:
	cd talk && $(LATEX) talk.tex >/dev/null && $(LATEX) talk.tex >/dev/null
	cd talk && $(LATEX) prof_talk.tex >/dev/null && $(LATEX) prof_talk.tex >/dev/null

clean:
	find . \( -name '*.aux' -o -name '*.log' -o -name '*.out' -o -name '*.toc' -o -name '*.nav' \
	  -o -name '*.snm' -o -name '*.vrb' \) -delete
