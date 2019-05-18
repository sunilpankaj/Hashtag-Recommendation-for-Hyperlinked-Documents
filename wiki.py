import wikipedia as wiki
try:
	camp = wiki.page("sosoosos")
	print "camp"
	print len(camp.content)
except wiki.exceptions.DisambiguationError as e:
	print e.options
	camp2 = wiki.page(e.options[0])
	print "camp2"
	print len(camp2.content)
except wiki.exceptions.PageError:
    print " no wiki page"