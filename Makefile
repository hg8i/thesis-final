postUrl:=http://www.hg8i.com/thesisPageCount/?pass=4&pageupdate=

default: thesis.tex 
	lualatex thesis.tex $<
	# ==================================================
	# Finished:
	# * lualatex
	# (To run bibtex also, "make full")
	# ==================================================

part: thesis.tex 
	lualatex thesis.tex $<
	-bibtex  $(basename $<)
	lualatex thesis.tex $<
	lualatex thesis.tex $<
	# --------------------------------------------------
	pdfinfo thesis.pdf | grep Pages
	# --------------------------------------------------

full: thesis.tex 
	lualatex thesis.tex $<
	-bibtex  $(basename $<)
	lualatex thesis.tex $<
	lualatex thesis.tex $<
	# Post thesis length to website!
	# pdfinfo thesis.pdf | grep Pages | awk '{print "http://www.hg8i.com/thesisPageCount/?pass=4&pageupdate=" $$2}' | xargs curl --silent --output /dev/null $$1
	# --------------------------------------------------
	pdfinfo thesis.pdf | grep Pages
	# --------------------------------------------------

	# ==================================================
	# Finished:
	# * lualatex
	# * bibtex
	# * Post thesis length to http://www.hg8i.com/thesisPageCount
	# ==================================================


clean:
	-rm *.dvi *.toc *.aux *.log *.out \
		*.bbl *.blg *.brf *.bcf *-blx.bib *.run.xml \
		*.cb *.ind *.idx *.ilg *.inx \
		*.synctex.gz *~ *.fls *.fdb_latexmk .*.lb spellTmp



