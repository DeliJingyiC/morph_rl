import sys
from collections import *

def strCell(cell):
    return ".".join(list(cell))

class Clusters:
    def __init__(self, points):
        self.members = {}
        for pi in points:
            self.members[pi] = set([pi])
        self.pointToID = {}
        for pi in points:
            self.pointToID[pi] = pi

    def merge(self, pi, pj):
        idI = self.trace(pi)
        idJ = self.trace(pj)
        if idI == idJ:
            return
        self.pointToID[pi] = idJ
        self.members[idJ].update(self.members[idI])
        del self.members[idI]

    def trace(self, pi):
        parent = pi
        while self.pointToID[parent] != parent:
            parent = self.pointToID[parent]
        return parent

    def exclude(self, counts, pi, pj):
        excludingLemmas = set()
        allMem = set()
        for xx in self.members.get(self.trace(pi), []):
            allMem.add(xx)
        for xx in self.members.get(self.trace(pj), []):
            allMem.add(xx)
        allMem = list(allMem)
        for ii, xi in enumerate(allMem):
            for xj in allMem[ii + 1:]:
                excludingLemmas.update(counts[xi][xj])

        # print("exclusion count", strCell(pi), strCell(pj),
        #       len(excludingLemmas), [strCell(xx) for xx in allMem])
        # if len(excludingLemmas) < 20:
        #     print(excludingLemmas)

        return len(excludingLemmas)

    def __str__(self):
        def strfy(feats):
            return ";".join(feats)

        return "\n".join(
            [f"{strfy(key)}\t:\t\t {',   '.join([strfy(vi) for vi in vals])}"
             for (key, vals) in
             self.members.items()])
    
def findPossSyncretism(pos2form):
    sync = defaultdict(set)
    for pi, fi in pos2form.items():
        sync[fi.lower()].add(pi)

    return sync

def collapseSyncretism(lemmaPosForm, cutoff):
    allPos = set()
    for lemma, pos2form in lemmaPosForm.items():
        for pos in pos2form:
            allPos.add(pos)

    syncsExcluded = defaultdict(lambda: defaultdict(set))
    syncsAllowed = defaultdict(lambda: defaultdict(set))
    
    for lemma, pos2form in lemmaPosForm.items():
        possSyncs = findPossSyncretism(pos2form)

        # for pi, fi in pos2form.items():
        #     print(pi, fi)

        # print()

        # for fi, syncset in possSyncs.items():
        #     if len(syncset) > 1:
        #         print(fi, syncset)

        # print()

        for pi, fi in pos2form.items():
            syncset = possSyncs[fi]
            for pj in syncset:
                syncsAllowed[pi][pj].add(lemma)
                syncsAllowed[pj][pi].add(lemma)

            for pj in pos2form:
                if pj not in syncset:
                    syncsExcluded[pi][pj].add(lemma)
                    syncsExcluded[pj][pi].add(lemma)

    # print("exclusions")
    # for kk, vv in syncsExcluded.items():
    #     print(kk, vv)
    # assert(0)

    cells = Clusters(allPos)
    for pi in sorted(allPos, key=len, reverse=True):
        for pj, ct in sorted(syncsAllowed[pi].items(),
                             key=lambda xx: len(xx[1]), reverse=True):
            if pi == pj:
                continue
            if cells.exclude(syncsExcluded, pi, pj) < cutoff:
                # print("merge", strCell(pi), strCell(pj))
                cells.merge(pi, pj)
                break

    return cells

def readConll(fh, multiword=False,
              discard={"Typo" : "Yes", "Abbr" : "Yes"},
              lowercase=True):
    eos = True
    for line in fh:
        if not line.strip() or line.startswith("#"):
            if not eos:
                yield (None, None, None, None)
                eos = True

            continue

        eos = False
        flds = line.split("\t")
        idn, word, lemma, uPos, aPos, posFeats = flds[0:6]
        if "-" in idn and not multiword:
            continue
        posFeats = [xx.split("=") for xx in posFeats.split("|") if xx != "_"]
        posFeats = dict(posFeats)

        skip = False
        for ki, vi in discard.items():
            if posFeats.get(ki) == vi:
                skip = True
        if skip:
            continue

        posFeats = frozenset(["=".join(xx) for xx in posFeats.items()]) #frozenset(posFeats.values())
            
        if lowercase:
            lemma = lemma.lower()
            word = word.lower()

        yield (word, lemma, uPos, posFeats)

def readUDVocab(path, posTarget=None, addLemma=False, returnCounts=False):
    lemmaPosForm = defaultdict(dict)
    lemmaPosFormCount = defaultdict(lambda: defaultdict(Counter))

    with open(path) as fh:
        for (word, lemma, uPos, posFeats) in readConll(fh):
            if posTarget != None and uPos != posTarget:
                continue

            lemmaPosForm[lemma][posFeats] = word.lower()
            lemmaPosFormCount[lemma][posFeats][word.lower()] += 1
            if addLemma:
                lemmaPosForm[lemma][frozenset(["lemma"])] = lemma.lower()

    if returnCounts:
        return lemmaPosForm, lemmaPosFormCount
    return lemmaPosForm

def fullCollapseSyncretism(lemmaPosForm, discardCell=5, cutoff=5):
    cellCounts = Counter()
    for li, pos2form in lemmaPosForm.items():
        for pos in pos2form:
            cellCounts[pos] += 1

    # for cell, ct in cellCounts.most_common():
    #     print(strCell(cell), "\t", ct)

    #get rid of cells that don't look reasonable
    for li, pos2form in lemmaPosForm.items():
        newPos2Form = dict([ (key, val) for (key, val) in pos2form.items()
                             if cellCounts[key] >= discardCell])
        lemmaPosForm[li] = newPos2Form

    cells = collapseSyncretism(lemmaPosForm, cutoff=cutoff)

    # for cell, values in cells.members.items():
    #     print(cell, "\t", values)
    #     print()
        
    cellNames = {}
    for cell, vals in cells.members.items():
        if vals is not None:
            # if these cells share some features, assign those as a name
            name = frozenset.intersection(*vals)
            if len(name) == 0:
                # otherwise assign a name arbitrarily
                name = min(list(vals), key=len)

            for vi in vals:
                cellNames[vi] = name

    # for each syncretism set, print all matching forms and the name
    for cell, vals in cells.members.items():
        if vals is not None:
            print(", ".join([strCell(xx) for xx in vals]), "belong to", cellNames[cell])
            print()

    cellToCanon = {}
    for cell, vals in cells.members.items():
        if vals is not None:
            for vi in vals:
                cellToCanon[vi] = cellNames[cell]

    return cellToCanon

def relabelCells(lemmaPosForm, cellToCanon):
    delLemma = set()

    for lemma in lemmaPosForm:
        pos2Form = lemmaPosForm[lemma]
        newPos2Form = {}
        for pos, form in pos2Form.items():
            canon = cellToCanon.get(pos)
            if canon != None:
                if canon in newPos2Form and newPos2Form.get(canon) != form:
                    if lemma not in delLemma:
                        print("Deleting lemma", lemma, "because variant forms can't be collapsed to", canon)
                        print("\tCanonical form", newPos2Form.get(canon), "variant", pos, form)
                        delLemma.add(lemma)
                else:
                    newPos2Form[canon] = form

            lemmaPosForm[lemma] = newPos2Form

    for lemma in delLemma:
        del lemmaPosForm[lemma]

def relabelCounts(lemmaPosFormCount, cellToCanon):
    for lemma in lemmaPosFormCount:
        pos2Form2Count = lemmaPosFormCount[lemma]
        newPos2Form2Count = defaultdict(Counter)
        for pos, form2count in pos2Form2Count.items():
            canon = cellToCanon.get(pos)
            for form, count in form2count.items():
                newPos2Form2Count[canon][form] += count

        lemmaPosFormCount[lemma] = newPos2Form2Count    

if __name__ == "__main__":
    lemmaPosForm = readUDVocab(sys.argv[1], "VERB")
    cellToCanon = fullCollapseSyncretism(lemmaPosForm)
