import re
import itertools
import pynini
from collections import defaultdict

#https://stackoverflow.com/questions/48651891/longest-common-subsequence-in-python
def lcs(s1, s2):
    if len(s1) == 0 or len(s2) == 0:
        return ""

    matrix = [["" for x in range(len(s2))] for x in range(len(s1))]
    for i in range(len(s1)):
        for j in range(len(s2)):
            if s1[i] == s2[j]:
                if i == 0 or j == 0:
                    matrix[i][j] = s1[i]
                else:
                    matrix[i][j] = matrix[i-1][j-1] + s1[i]
            else:
                matrix[i][j] = max(matrix[i-1][j], matrix[i][j-1], key=len)

    cs = matrix[-1][-1]

    return cs

def diffExe(src, targ, cache):
    if (src, targ) in cache:
        return cache[(src, targ)]

    lc = lcs(src, targ)
    diff = multitapeMatch([lc, targ], targ).split()
    diffStr = diffToString(diff)
    # print("diffing", src, "with", targ)
    # print("lcs:", lc)
    # print("edit seq", diff)
    # print("diff str", diffStr)
    cache[(src, targ)] = src, diffStr
    return src, diffStr

def formatInputs(lemma, form, exemplars, cache):
    diffedExes = []
    for (src, targ) in exemplars:
        src, diffStr = diffExe(src, targ, cache)
        diffedExes.append((src, diffStr))

    analysis = multitapeMatch([lemma,] + [d1 for (d0, d1) in diffedExes], form, machineType=2)
    analysis = analysis.split()
    edits = convertToEdits(analysis)
    # print("the analysis of", lemma, form, exemplars)
    # print("the diffed exes", diffedExes)
    # print(analysis)
    # print("the edit seq")
    # print(edits)
    # print()

    return "".join(edits), diffedExes

def diffToString(syms):
    res = []
    reading = 0
    for si in syms:
        if si.startswith("<S>"):
            reading = si.replace("<S>", "")

        elif reading == "1" and not si.startswith("-"):
            res.append(si)

    return "".join(res)

def convertToEdits(syms):
    res = []
    lastValid = defaultdict(int)

    reading = 0
    for ind, si in enumerate(syms):
        if si.startswith("<S>"):
            reading = si.replace("<S>", "")

        if not (si.startswith("+") or si.startswith("-")):
            lastValid[reading] = ind

    reading = 0
    for ind, si in enumerate(syms):
        if si.startswith("<S>"):
            reading = si.replace("<S>", "")

        else:
            if si.startswith("-"):
                if ind < lastValid[reading]:
                    res.append("-" + reading)
            elif si.startswith("+"):
                res.append(si[1:])
            else:
                res.append(reading)

    return res

def convertFromEdits(pred, lemma, diffed):
    sources = [lemma,] + [diff for (src, feats, diff) in diffed]
    iters = [itertools.chain(list(xx), itertools.repeat(None)) for xx in sources]
    predC = re.split("([+-]?.)", pred)
    predC = [xx for xx in predC if xx != ""]
    res = []
    for ci in predC:
        if ci.startswith("-") and len(ci) == 2:
            try:
                digit = int(ci[1:])
                next(iters[digit])
            except (ValueError, IndexError):
                pass #malformed deletion or attempt to read invalid source
        else:
            try:
                digit = int(ci)
                nxt = next(iters[digit])
                if nxt != None:
                    res.append(nxt)
            except IndexError:
                pass #attempt to read invalid source
            except ValueError:
                #non-digit interpreted as insertion
                res.append(ci)

    # print("converting", pred, "from", sources)
    # print("full diffs:", diffed)
    # print(res)

    return "".join(res)

#desired:
#given s1, s2...sn and target t, produce t by copying characters
#st:
#cost inserting characters > copying with switch between si > copying
#emit explicit deletions for characters from si which are skipped
#deleting chars from si should increase in cost as fn of i
def multitapeMatch(strs, target, machineType=1):
    if machineType == 1:
        machine = makeMachine(strs, extraChars=target)
    else:
        machine = makeMachine2(strs, extraChars=target)
    machine.arcsort("ilabel")
    syms = machine.input_symbols()
    #syms.write_text("symtab.tsv") #debug
    
    acc = pynini.accep(" ".join(target), token_type=syms)
    acc.set_input_symbols(syms)
    acc.set_output_symbols(syms)
    # print(acc)
    
    comp = pynini.compose(acc, machine)
    # print(comp)

    # print("shortest path")
    spath = pynini.shortestpath(comp)
    # print(spath)

    # print("--")
    spath = spath.topsort().project("output")
    # print(spath)
    # print(spath.string(syms))
    return spath.string(syms)
    
def crossproduct(vs):
    if len(vs) == 0:
        yield []
    else:
        for xp in crossproduct(vs[1:]):
            for vi in vs[0]:
                yield [vi,] + xp

def getnext(state):
    vec, active = state
    nxt = list(vec)
    if active == len(nxt):
        #cannot advance pointer in insert state
        return None, None
    nxt[active] += 1
    nextState = (tuple(nxt), active)
    return nextState, nxt[active] - 1
                
def makeMachine(strs, extraChars=""):
    #symbols will be chars
    syms = pynini.SymbolTable()
    syms.add_symbol("<eps>")
    realChars = set()
    for sind, si in enumerate(strs):
        syms.add_symbol("<S>" + str(sind))
        for ci in si:
            realChars.add(ci)
            syms.add_symbol(ci)
            syms.add_symbol("-" + ci)
            syms.add_symbol("+" + ci)

    for ci in extraChars:
        realChars.add(ci)
        syms.add_symbol(ci)
        syms.add_symbol("-" + ci) #vacuous
        syms.add_symbol("+" + ci)
            
    #states in our machine will be vectors of string indices
    #plus an active string designation
    #active string n-1 is the insert state
    stateToInd = { "START" : 0 }
    for vec in crossproduct([list(range(len(xx) + 1)) for xx in strs]):
        vec = tuple(vec)
        for si in range(len(strs) + 1):
            stateToInd[(vec, si)] = len(stateToInd)

    # print(stateToInd)

    #the machine will accept the target string on the input side and
    #transduce it to a derivation
    
    fst = pynini.Fst()
    fst.set_input_symbols(syms)
    fst.set_output_symbols(syms)
    fst.add_states(len(stateToInd))

    fst.set_start(0)

    #add COPY arcs which output the next character in each string
    for state, ind in stateToInd.items():
        if state == "START":
            continue

        nextState, nextT = getnext(state)
        if nextState in stateToInd:
            nextInd = stateToInd[nextState]
            vec, active = state
            nextChar = strs[active][nextT]
            arc = pynini.Arc(syms.find(nextChar),
                             syms.find(nextChar),
                             0,
                             nextInd)
            fst.add_arc(ind, arc)

    #add SKIP arcs which delete the next character in each string
    for state, ind in stateToInd.items():
        if state == "START":
            continue

        nextState, nextT = getnext(state)
        if nextState in stateToInd:
            nextInd = stateToInd[nextState]
            vec, active = state
            nextChar = strs[active][nextT]
            arc = pynini.Arc(syms.find("<eps>"),
                             syms.find("-" + nextChar),
                             .01 * (len(strs) - active),
                             nextInd)
            fst.add_arc(ind, arc)

    #add INSERT arcs which consume a random character without
    #advancing the pointer
    #but only from the insert state
    for state, ind in stateToInd.items():
        if state == "START":
            continue

        (vec, active) = state
        if active != len(strs):
            continue

        for char in realChars:
            arc = pynini.Arc(syms.find(char),
                             syms.find("+" + char),
                             3,
                             ind)
            fst.add_arc(ind, arc)

    #add SWITCH arcs which change the active string we're working on
    for state, ind in stateToInd.items():
        if state == "START":
            continue

        (vec, active) = state
        for ai in range(len(strs) + 1):
            if ai != active:
                nextState = (vec, ai)
                nextInd = stateToInd[nextState]
                if ai < len(strs):
                    arc = pynini.Arc(syms.find("<eps>"),
                                     syms.find("<S>" + str(ai)),
                                     1,
                                     nextInd)
                else:
                    #don't bother outputting a switch for insertions
                    arc = pynini.Arc(syms.find("<eps>"),
                                     syms.find("<eps>"),
                                     1,
                                     nextInd)
                fst.add_arc(ind, arc)

    #connect the start state to all 0 states
    for active in range(len(strs)):
        state = tuple([0 for xi in strs]), active
        nextInd = stateToInd[state]
        arc = pynini.Arc(syms.find("<eps>"),
                         syms.find("<S>" + str(active)),
                         0,
                         nextInd)
        fst.add_arc(0, arc)

    #designate all states at which the entire string is consumed as final
    for active in range(len(strs) + 1):
        state = tuple([len(xi) for xi in strs]), active
        nextInd = stateToInd[state]
        fst.set_final(nextInd, weight=0)
                
    # print("--machine dump--")
    # print(fst)
    # print("--")

    return fst

def makeMachine2(strs, extraChars=""):
    #symbols will be chars
    syms = pynini.SymbolTable()
    syms.add_symbol("<eps>")
    realChars = set()
    for sind, si in enumerate(strs):
        syms.add_symbol("<S>" + str(sind))
        for ci in si:
            realChars.add(ci)
            syms.add_symbol(ci)
            syms.add_symbol("-" + ci)
            syms.add_symbol("+" + ci)

    for ci in extraChars:
        realChars.add(ci)
        syms.add_symbol(ci)
        syms.add_symbol("-" + ci) #vacuous
        syms.add_symbol("+" + ci)

    #states in our machine will be vectors of string indices
    #plus an active string designation
    #active string n-1 is the insert state
    stateToInd = { "START" : 0 }
    for vec in crossproduct([list(range(len(xx) + 1)) for xx in strs]):
        vec = tuple(vec)
        for si in range(len(strs) + 1):
            stateToInd[(vec, si)] = len(stateToInd)

    # print(stateToInd)

    #the machine will accept the target string on the input side and
    #transduce it to a derivation
    
    fst = pynini.Fst()
    fst.set_input_symbols(syms)
    fst.set_output_symbols(syms)
    fst.add_states(len(stateToInd))

    fst.set_start(0)

    #add COPY arcs which output the next character in each string
    for state, ind in stateToInd.items():
        if state == "START":
            continue

        nextState, nextT = getnext(state)
        if nextState in stateToInd:
            nextInd = stateToInd[nextState]
            vec, active = state
            nextChar = strs[active][nextT]
            arc = pynini.Arc(syms.find(nextChar),
                             syms.find(nextChar),
                             0,
                             nextInd)
            fst.add_arc(ind, arc)

    #add SKIP arcs which delete the next character in each string
    for state, ind in stateToInd.items():
        if state == "START":
            continue

        nextState, nextT = getnext(state)
        if nextState in stateToInd:
            nextInd = stateToInd[nextState]
            vec, active = state
            nextChar = strs[active][nextT]
            arc = pynini.Arc(syms.find("<eps>"),
                             syms.find("-" + nextChar),
                             1,
                             nextInd)
            fst.add_arc(ind, arc)

    #add INSERT arcs which consume a random character without
    #advancing the pointer
    #but only from the insert state
    for state, ind in stateToInd.items():
        if state == "START":
            continue

        (vec, active) = state
        if active != len(strs):
            continue

        for char in realChars:
            arc = pynini.Arc(syms.find(char),
                             syms.find("+" + char),
                             1,
                             ind)
            fst.add_arc(ind, arc)

    #add SWITCH arcs which change the active string we're working on
    for state, ind in stateToInd.items():
        if state == "START":
            continue

        (vec, active) = state
        for ai in range(len(strs) + 1):
            if ai != active:
                nextState = (vec, ai)
                nextInd = stateToInd[nextState]
                if ai < len(strs):
                    arc = pynini.Arc(syms.find("<eps>"),
                                     syms.find("<S>" + str(ai)),
                                     1,
                                     nextInd)
                else:
                    #don't bother outputting a switch for insertions
                    arc = pynini.Arc(syms.find("<eps>"),
                                     syms.find("<eps>"),
                                     1,
                                     nextInd)
                fst.add_arc(ind, arc)

    #connect the start state to all 0 states
    for active in range(len(strs)):
        state = tuple([0 for xi in strs]), active
        nextInd = stateToInd[state]
        arc = pynini.Arc(syms.find("<eps>"),
                         syms.find("<S>" + str(active)),
                         0,
                         nextInd)
        fst.add_arc(0, arc)

    #end anywhere, regardless of string consumption
    for state, ind in stateToInd.items():
        if state == "START":
            continue

        fst.set_final(ind, weight=0)
                
    # print("--machine dump--")
    # print(fst)
    # print("--")

    return fst

if __name__ == "__main__":
    print(multitapeMatch(["dress", "es"], "dresses"))
    print(multitapeMatch(["dress", "er", "hairy"], "hairdressers"))
    print(multitapeMatch(["undresses", "er", "hairy"], "hairdressers"))
    print(multitapeMatch(["undresses", "er", "hairy"], "hairdresers"))

    print(formatInputs("foot", "football", [["base", "baseball"]]))
    print(formatInputs("foot", "football", [["base", "baseman"]]))
    print(formatInputs("foot", "football", [["base", "basement"]]))
    print(formatInputs("foot", "footballer", [["base", "baseball"], ["hold", "holder"]]))
    print(formatInputs("foot", "footballer", [["base", "baseball"], ["hold", "holders"]]))
    print(formatInputs("foot", "footballer", [["base", "baseballing"], ["hold", "holders"]]))
    print(formatInputs("foot", "footballer", [["base", "baseballinger"], ["hold", "holders"]]))
    print(formatInputs("duman", "dumanlarımızı", [["duman", "dumanları"], ["duman", "dumanımız"]]))
    print(formatInputs("duman", "dumanımızı", [["duman", "dumanları"], ["duman", "dumanımız"]]))
