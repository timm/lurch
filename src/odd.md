#!/usr/bin/env python3
# vim: ft=python:nospell:sta:et:sw=2:ts=2:sts=2
#--------- --------- --------- --------- --------- ---------

# <img width=300 
#  src="https://raw.githubusercontent.com/timm/raised/master/etc/img/face.png?token=AAAHECZST5PO4FLNU5OM7DK54HNT4">

# "Round up the usual suspects."     
# - Capt. Renault, Casablanca
import subprocess,traceback,random,math,sys,re,os

class o:
  "Objects support quick inits and pretty print of contents."
  def __init__(i,**d) : 
    i.__dict__.update(**d)
  def __repr__(i):
    return i.__class__.__name__+'{'+d2s(i.__dict__)+'}'

#--------- --------- --------- --------- --------- ---------
# # Config Options

THE= o(
  misc = o( dummy= 0,
            seed = 1,
            hide = '_'),
  dist = o( some    = 256,
            tries   = 10,
            balanced= 0.1,
            far      = 0.8),# 0.999),
  gen  = o( cap     = True,
            retries = 10,
            steps   = 10,
            cf      = 0.3,
            f       = 0.3),
  some = o( most = 256),
  tree = o( rpMin = 0.5,
            minObs=4,
            wait = 0,
            verbose=True,
            rnd   =2),
  char = o( sep = ",",
            num = "$",
            less = "<",
            more = ">",
            skip = "?",
            klass= "!",
            doomed = r'([\n\t\r ]|#.*)'),
  eg   = o( run=r".*"),
  row  = o( p = 2),
  div  = o( trivial = 1.025,
            cohen   = 0.3,
            min     = 0.5)
)
random.seed(THE.misc.seed)

#--------- --------- --------- --------- --------- ---------
# # Utilities
# ## How to Run Examples

class eg(o):
  """
  Manager for examples/tests. Add test functions using:
  
  ```
  @eg
  def xxx(): ...
  ```

  Usage:
 
  - `eg.run("xxx")`   : run one example fuction
  - `THE.eg.run='f*'  : set a regx to select functions to run
  - `print(eg.run())` : run all selected functions.
  """
  all = {}  # Place to store demos/tests.
  n   = 0   # Number of called demos/tests.
  y   = 0   # Number of demos/tests with assert failers"
  tell= "#Tests: tries 1 fails 1 passed%s 100"
  def __init__(i,f): eg.all[f.__name__] = f
  def run():         
    [eg.run1(one) for one in eg.all.values()]
    return eg.tell
  def run1(f):
    want=THE.eg.run
    if re.match(want, f.__name__):
      print(f"#### {f.__name__}")
      if f.__doc__: 
        print("# "+ re.sub(r'\n[ \t]*',"\n# ", f.__doc__))
      eg.y += 1
      try:    
        random.seed(THE.misc.seed) 
        f()
      except: 
        eg.n += 1
        y,n  = eg.y, eg.n
        eg.tell = f"#Tests: tries {y} fails {n} passed%% %s" % (
                    round(100*y/(y+n+0.0000000001)))
        return print(traceback.format_exc())

@eg
def _eg1(): assert 1==2,"this test should fail"
@eg
def _eg2(): assert eg.tell=="#Tests: tries 1 fails 1 passed% 50"

# ## Utilities 

def d2s(d):
  "Keys are sorted alphabetically; hide 'private' keys"
  hide = THE.misc.hide
  kvs  = sorted([(k,v) for k,v in d.items()])
  a    = [('%s=%s' % (k,v)) for k,v in kvs if k[0] != hide]
  return ", ".join(a)

isa = lambda x,y: isinstance(x,y)
same = lambda z:z
r    = random.random
any  = random.choice

def sample(l,n=1):
  "Sample with replacement."
  return random.sample(l, min(len(l),n))

def now(ok,m=None):
  "maybe send a complaint to standard error, then exit"
  if not ok:
    sys.stderr.write(f"#E> {m or 'sigh'}\n")
    sys.exit()

def atom(x):
  "coerce x into the right kind of atom"
  try: return int(x)
  except:
    try: return float(x)
    except: return x
@eg 
def _utilities(): 
  assert 'a=1, b=2'==d2s(dict(a=1,b=2,_c=3))
  assert 't' == any('string')
  assert 0.567 < r() < 0.57
  assert ['a','b','d','c'] == sample('abcd',5)
  assert atom('56.1') == 56.1
  assert atom('x56')  == 'x56'

# ## Rows 

def rows(file, no=THE.char.skip):
  "Read csv, Skip columns with a row1 name containing 'no'."
  use = None
  with open(file) as fp:
    for n,line in enumerate(fp): 
      a   = re.sub(THE.char.doomed, '', line.strip())
      a   = a.split(THE.char.sep)
      use = use or [n for n,x in enumerate(a) if not no in x]
      yield [a[n] for n in use]

@eg
def _rows():
  assert [a for a in rows('odd.py')][-1] == ['eg.run()']

# ## Command-line Processing 

def cli(d=THE, a = [atom(x) for x in sys.argv[1:]]):
  """
  Update nested dictionary from array 'a'. E.g.

  ```
  python3 odd.py dist -far 0.5 tree -wait 2 -verbose
  ```

  translates to
  
  ```
  d["dist"]["far"] = 0.5
  d["tree"]["wait"] = 2
  d["tree"]["verbose"] = not d["tree"]["verbose"]
  ```

  Note that booleans can be flipped just by mentioning them.

  Note also that this procedure terminates the program if:

  - the new val is a different type to the old one
  - a flag in the array is not in the dictionary
  
  """
  what = {}
  while a:
    arg = a.pop(0)
    if arg in d.__dict__:
      what = d.__dict__[arg].__dict__
    else:
      now(isa(arg,str) and arg[0] == "-", f"bad flag '{arg}'")
      arg1 = arg[1:]
      now(arg1 in what, f"{arg1}: unknown group")
      old = what[arg1]
      if isa(old, bool):
        what[arg1] = not what[arg1]
      else:
        x = a.pop(0)
        now(type(old) == type(x), f"{x} unlike '{type(old)}'")
        what[arg1] = x

@eg
def _cli():
  cli(THE,["misc", "-dummy", 1,'-dummy', 10, '-dummy',100])
  assert 100==THE.misc.dummy

#--------- --------- --------- --------- --------- ---------
# # Things

class Thing(o):
  "`Thing`s are either `Num`s or `Sym`s."
  def __init__(i,inits=[],pos=0,txt="",key=same):
    i.n, i.pos, i.txt, i.key, i.w = 0, pos, txt, key, 1
    if THE.char.less in txt: i.w = -1
    [i+x for x in inits]
  def prep(i,x): return x
  def __add__(i,x0):
    x1 = i.key(x0)
    if x1 != THE.char.skip: 
      x1 = i.prep(x1)
      i.n += 1
      i.add(x1)
    return x1

# ## `Num` Things

class Num(Thing):
  "`Num`s track streams of numerics." 
  def __init__(i,**a):
    i.mu = i.m2 = i.sd = 0
    i.lo, i.hi = math.inf, -math.inf
    super().__init__(**a)
  def prep(i,x): return float(x)
  def sd0(i):
    if i.n  < 2: return 0
    if i.m2 < 0: return 0
    return (i.m2/(i.n - 1))**0.5
  def add(i,x):
    i.hi  = max(x,i.hi)
    i.lo  = min(x,i.lo)
    d     = x - i.mu
    i.mu += d/i.n
    i.m2 += d*(x - i.mu)
    i.sd = i.sd0()
  def norm(i,x): 
    return (x - i.lo) / (i.hi - i.lo + 10**-32)
  def diff(i,x,y,   no=THE.char.skip):
   if x==no and y==no: 
     return 1
   if x==no: 
     y = i.norm(y); x = 0 if y > 0.5 else 1
   elif y==no: 
     x = i.norm(x); y = 0 if x > 0.5 else 1
   else:
     x,y = i.norm(x), i.norm(y)
   return abs(x-y)
   
@eg
def _num():
  n = Num([9, 2,  5, 4, 12,  7, 8, 11, 9, 3, 
           7, 4, 12, 5,  4, 10, 9,  6, 9, 4])
  now(3.06 < n.sd < 3.07,"bad sd")
  now(int(n.mu) == 7,    "bad mu")

# ## `Sym` Things

class Sym(Thing):
  "`Num`s track streams of symbols." 
  def __init__(i,**lst):
    i.mode, i.most, i._ent, i.seen = None,0,None,{}
    super().__init__(**lst)
  def add(i,x):
    i._ent = None
    new = i.seen.get(x,0) + 1
    i.seen[x] = new
    if new > i.most:
      i.mode, i.most = x, new
  def diff(i,x,y,   no=THE.char.skip):
    if x==no and y==no: return 1
    return 0 if x == y else 1
  def entropy(i):
    if i._ent is None:
      i._ent = 0
      for _,v in i.seen.items():
        p = v/i.n
        i._ent -= p*math.log(p,2)
    return i._ent

@eg
def _sym():
  s = Sym('yyn')
  assert 0.918 < s.entropy() < 0.919

#--------- --------- --------- --------- --------- ---------
# # `Col`umns

class Cols(o):
  def __init__(i,words=[]):
    i.words = []
    i.nums, i.syms, i.all = [],[],[]
    i.x = o(nums=[], syms=[], all=[])
    i.y = o(nums=[], syms=[], all=[], 
            goals=[], less=[], more=[], klass=None)
    i + words
  def clone(i,lst):
    t= Tbl().head(i.words)
    [t.row(x) for x in lst]
    return t

  def __add__(i,a):
    i.words = a
    [i.add(pos,s) for pos,s in enumerate(a)]
  def add(i,pos,s):
    c = (Num if nump(s) else Sym)(pos=pos,txt=s)
    if klassp(s) : i.y.klass  =  c
    if goalp(s)  : i.y.goals += [c]
    if True      : i.add1(s, c, i.all,  i.x.all,  i.y.all)
    if nump(s)   : i.add1(s, c, i.nums, i.x.nums, i.y.nums)
    else         : i.add1(s, c, i.syms, i.x.syms, i.y.syms)
  def add1(i, s, c, all, x, y):
    all += [c]
    (y if yp(s) else x).append(c)

def goalp(s)  : return THE.char.less in s or THE.char.more in s
def nump(s)   : return THE.char.num  in s or goalp(s) 
def yp(s)     : return klassp(s) or goalp(s)
def klassp(s) : return THE.char.klass in s

@eg
def _cols():
  c = Cols(["$peace","love","<understanding"])
  [print(x) for x in c.all]

#--------- --------- --------- --------- --------- ---------
# # Tables

"""
When we collect data:

- The columns of those tables are either independent or
dependent variables (which, for the sake of brevity,  we
call `x` and `y`).
- These, in turn, divide into lists of numeric `Num` columns
and symbolic `Sym` columns.
- The dependent `y` columns also divide into lists of
`klass`es and goals that we want `more` or `less`.

Note that when we create a new column, we add it to all the
relevant lists.
"""

# ## `Row` hold data

class Row(o):
  id=0
  def __init__(i,cells=[]):
    i.cells, i.cooked = cells,[]
    i.id = Row.id = Row.id + 1
  def __add__(i,a): i.cells = a

# ## `Tbl` hold hold many `Row`s

class Tbl(o):
  def __init__(i): 
    i.cols, i.rows = None, []
  def head(i,a)  : 
    i.cols = Cols(a); return i
  def row(i,a)   : 
    a = [col + x for col,x in zip(i.cols.all,a)] 
    i.rows += [Row(a)]
  def read(i,file):
    for a in rows(file):
      i.row(a) if i.cols else i.head(a)
    return i
  def dist(i,j,k,cols=None):
    d,n,p = 0, 10**-32, THE.tbl.p
    for col in cols or i.cols.x:
      tmp = col.dict(j.cells[col.pos], k.cells[col.pos])
      d  += d**p
      n  += 1
    return d**(1/p) / n**(1/p)
  def pivots(i,cols=None,rows=None):
    "Pivots are remote items in the `rows`."
    rows    = rows or sample(i.rows,THE.tbl.some)
    cols    = cols or i.cols.x
    w       = any(rows)
    x,_,_   = w.far(i,cols,rows)
    y,c,mid = x.far(i,cols,rows)
    return x,y,c,mid
  def far(i,j, cols,rows):
    a   = sorted([(j.dist(k,i,cols),j) for j in rows])
    one = lambda z: a[int(len(a)*z)]
    far = THE.tbl.far
    return [ one(far)[1], one(far)[0], one(far/2)[0] ]

@eg
def _tbl(): 
  t=Tbl().read('../data/weather.csv')
  [print(x) for x in t.cols.y.all]

#--------- --------- --------- --------- --------- ---------
# ## `Col`umns

class RpTree(o):
  def __init__(i,t,lst=None,up=None,cols=None,lvl=0):
    print('|..'*lvl, end="")
    lst = lst or t.rows
    cols = cols or t.cols.x
    super().__init__(leaf=None,kids=[],t=t,_up=up,cols=cols,n=len(lst))
    if len(lst) < 2*(len(t.rows)**THE.tree.rpMin):
      i.leaf = t.cols.clone(lst)
      print(i.n)
    else:
      i.x, i.y, _, i.mid = t.pivots()
      left, right = [],[]
      for row in lst:
        what = left if t.dist(i.x,row,cols) < i.mid else right
        what += [row]
      i.kids = [ RpTree(t, left,  i, cols, lvl+1)
               , RpTree(t, right, i, cols, lvl+1)]
  def relevant(i,row):
    if i.leaf: return i.leaf
    elif i.t.dist(i.x,row,i.cols) < i.mid :
      return i.kids[0].relevant(row)
    else:
      return i.kids[1].relevant(row)

  def show(i, lvl=0):
    print(('|..'*lvl) + str(i.n))
    if i.kids:
      i.show(i.kids[0],lvl+1)
      i.show(i.kids[1],lvl+1)

#--------- --------- --------- --------- --------- ---------
# # Start up Actions

@eg
def _html():
  com=['/usr/local/bin/pycco','-d','../../../../tmp','odd.py']
  print(f'# {com}')
  subprocess.run(com)

if __name__ == "__main__":
  cli()
  eg.run()
