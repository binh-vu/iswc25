from __future__ import annotations

from rdflib import RDFS

# set of properties that telling a data node contains entities (e.g., rdfs:label)
IDENT_PROPS = {str(RDFS.label)}
