import pynini

#https://stackoverflow.com/questions/48651891/longest-common-subsequence-in-python
def lcs(s1, s2):
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

#desired:
#given s1, s2...sn and target t, produce t by copying characters
#st:
#cost inserting characters > copying with switch between si > copying
#emit explicit deletions for characters from si which are skipped
#deleting chars from si should increase in cost as fn of i
def multitapeMatch(strs, target):
    machine = makeMachine(strs, extraChars=target)
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
    stateToInd = { "START" : 0 }
    for vec in crossproduct([list(range(len(xx) + 1)) for xx in strs]):
        vec = tuple(vec)
        for si in range(len(strs)):
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
    #cost 3 makes a single insert worse than two switches and a copy
    for state, ind in stateToInd.items():
        if state == "START":
            continue

        (vec, active) = state
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
        for ai in range(len(strs)):
            if ai != active:
                nextState = (vec, ai)
                nextInd = stateToInd[nextState]
                arc = pynini.Arc(syms.find("<eps>"),
                                 syms.find("<S>" + str(ai)),
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
    for active in range(len(strs)):
        state = tuple([len(xi) for xi in strs]), active
        nextInd = stateToInd[state]
        fst.set_final(nextInd, weight=0)
                
    # print("--machine dump--")
    # print(fst)
    # print("--")

    return fst            

print(multitapeMatch(["dress", "es"], "dresses"))
print(multitapeMatch(["dress", "er", "hairy"], "hairdressers"))
print(multitapeMatch(["undresses", "er", "hairy"], "hairdressers"))
print(multitapeMatch(["undresses", "er", "hairy"], "hairdresers"))
